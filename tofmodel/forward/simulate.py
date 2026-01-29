# -*- coding: utf-8 -*-

import time
import numpy as np
from multiprocessing import Pool
import os
import logging
from functools import partial
from tofmodel.forward.fresignal import fre_signal_array as fre_signal
import math


def simulate_inflow(tr, te, npulse, w, fa, t1, t2, nslice, alpha, multi_factor, x_func, 
                    dx=0.01, offset_fact=1, varysliceprofile=True, X_given=None, ncpu=1, enable_logging=False):
    
    """ Routine for simulating inflow signals

    Parameters
    ----------
    tr : float
        repetition time (s)

    te : float
        repetition time (s)
        
    npulse : int
        total number of TR cycles to simulate
        
    w : float
        slice thickness (cm)
        
    fa : float
        flip angle (degrees)
        
    t1 : float
        T1 time constant for flowing fluid (s)
        
    t2 : float
        T2 time constant for flowing fluid (s)  
        
    nslice : int
        number of imaging slices
        
    alpha : list
        slice timing for each slice which should have length same as nslice (s)
        
    multi_factor : int
        multi-band factor
        
    x_func : func
        position (cm) as a function of time (s) and initial position (cm)
    
    dx : float, optional
        distance between initalized protons (cm), by default 0.01
    
    offset_fact : int, optional
        factor multiplied to the steady state signal contribution, by default 1

    rfprofile : str, optional
        method for rf slice profile, either 'ideal' or 'gaussian' (default)
           
    X_given : numpy.ndarray, optional
        array of proton positions (cm) directly supplied instead of computing in routine, by default None

    multithread : bool, optional
        use mutliple CPU cores for both position and signal computation, by default True
    
    enable_logging : bool, optional
        turn on logging messages; if off, only log critical messages, by default False
        
    Returns
    -------
    signal : numpy.ndarray
        matrix of signal timeseries (a.u.) for each TR/slice
    
    """
    
    # set up logging configuration
    if enable_logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.disable(logging.CRITICAL)  # Disable all logging
    logger = logging.getLogger(__name__)
    fa = fa * np.pi / 180
    alpha = np.array(alpha, ndmin=2).T
    assert np.size(alpha) == nslice, 'Warning: size of alpha should be nslice'
    timings_with_repeats, pulse_slice = get_pulse_targets(tr, nslice, npulse, alpha)
    
    # determine number of cores to use
    cpu_count = len(os.sched_getaffinity(0))
    if ncpu == -1:
        use_cores = cpu_count
    else:
        use_cores = int(ncpu)
        
    if X_given is None:
        tstart_pos = time.time()
        lower_bound, upper_bound = get_init_position_bounds(x_func, np.unique(timings_with_repeats), w, nslice)
        if use_cores > 1:
            num_cores = len(os.sched_getaffinity(0))
            X = compute_position_parallel(x_func, timings_with_repeats, lower_bound, upper_bound, dx, num_cores)
        else:
            X = compute_position(x_func, timings_with_repeats, lower_bound, upper_bound, dx)
        logger.info(f"trimming protons that never touch slices")
        mask = np.any((X > 0) & (X < w * nslice), axis=1)
        X = X[mask]
        logger.info(f"trimmed position bounds: ({X[0, 0]:.3f}, {X[-1, 0]:.3f}) cm")
        X = increase_proton_density(X, npulse, nslice, w, multi_factor, dx, min_proton_count=5, uptoslc=10, enable_logging=enable_logging)
        logger.info(f'position calculation time: {time.time() - tstart_pos:.2f} seconds')
    else:
        X = np.array(X_given)
        logger.info('using given proton positions. Skipping calculation...')
    nproton = X.shape[0]
    logger.info('running simulation with ' + str(nproton) + ' protons...')
    pulse_tr_actual = match_pulse_to_tr(npulse, nslice)
    tstart_sim = time.time()
    signal = np.zeros([npulse, nslice])
    params = ((npulse, nslice, X[iproton, :], multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual, offset_fact, varysliceprofile)
                  for iproton in range(nproton))
    if use_cores > 1:
        tasks_per_worker = 4
        if nproton > 0:
            optimal_chunksize = max(1, math.ceil(nproton / (use_cores * tasks_per_worker)))
        else:
            optimal_chunksize = 1
        logger.info(f"Using {use_cores} cores for {nproton} protons (chunksize: {optimal_chunksize})")
        with Pool(processes=use_cores) as pool:
            result = pool.starmap(
                compute_proton_signal_contribution, 
                enumerate(params), 
                chunksize=optimal_chunksize
            )
        for s in result:
            signal += s
    else:
        logger.info(f"using 1 cpu core")
        for iproton, p in enumerate(params):
            signal += compute_proton_signal_contribution(iproton, p)           
    logger.info(f'total simulation time: {time.time() - tstart_sim:.2f} seconds')
    num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
    signal = np.divide(signal, num_proton_in_slice, out=np.zeros_like(signal), where=num_proton_in_slice != 0)
    try:
        assert not np.isnan(signal[:, :nslice]).any(), 'Warning: NaN values found in signal array'
        assert not np.isinf(signal[:, :nslice]).any(), 'Warning: Inf values found in signal array'
    except AssertionError as error:
        print(error)
    else:
        return signal

def compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor):
    num_proton_in_slice = np.zeros((npulse, nslice), dtype=int)
    stride = nslice // multi_factor
    pos_at_end_of_tr_cycles = X[:, 0::stride]
    proton_slice = np.floor(pos_at_end_of_tr_cycles / w).astype(int)
    for ipulse in range(npulse):
        slices_at_tr = proton_slice[:, ipulse]
        mask = (slices_at_tr >= 0) & (slices_at_tr < nslice)
        if np.any(mask):
            num_proton_in_slice[ipulse, :] = np.bincount(slices_at_tr[mask], minlength=nslice)
    return num_proton_in_slice

def increase_proton_density(X, npulse, nslice, w, multi_factor, dx, min_proton_count=5, uptoslc=10, maxiter=200, enable_logging=False):
    if enable_logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.disable(logging.CRITICAL)
    logger = logging.getLogger(__name__)
    num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
    ind_sparse = np.where(num_proton_in_slice[:, :uptoslc] < min_proton_count)
    if ind_sparse[0].size > 0:
        logger.info(f"initial proton count is {X.shape[0]}")
        logger.info(f"found {ind_sparse[0].size} TR cycles with less than {min_proton_count} protons within the first {uptoslc} slices")
        logger.info(f"running density increase algorithm...")
        count = 0
        while ind_sparse[0].size > 0:
            count += 1
            if count > maxiter:
                logger.info(f"WARNING: reached max count iter of {maxiter}")
                logger.info(f"{ind_sparse[0].size} TR cycles have less than {min_proton_count} protons")
                logger.info(f"minimum TR cycle proton count is {num_proton_in_slice.min()}")
                return X
            thres = w / min_proton_count # threshold distance between adjacent position curves for increasing proton density
            X = increase_position_matrix_density(X, thres) 
            num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
            ind_sparse = np.where(num_proton_in_slice < min_proton_count)
        logger.info(f"finished after {count} iterations")
        logger.info(f"{ind_sparse[0].size} TR cycles have less than {min_proton_count} protons")
        logger.info(f"minimum TR cycle proton count is {num_proton_in_slice.min()}")
    return X

def compute_position(x_func, timings_with_repeats, lower_bound, upper_bound, dx):
    x0 = np.arange(lower_bound, upper_bound, dx)
    X = x_func(np.unique(timings_with_repeats), x0)
    return X

def compute_position_parallel(x_func, timings, lower_bound, upper_bound, dx, num_cores=None):
    if num_cores is None:
        num_cores = len(os.sched_getaffinity(0))
    x0_full = np.arange(lower_bound, upper_bound, dx)
    chunks = np.array_split(x0_full, num_cores)
    unique_timings = np.unique(timings)
    worker_func = partial(x_func, unique_timings)
    with Pool(processes=num_cores) as pool:
        results = pool.map(worker_func, chunks)
    return np.vstack(results)
    
def increase_position_matrix_density(X, thres):
    diffs = np.diff(X, axis=0)
    max_dx_for_pair = diffs.max(axis=1)
    ind_gap = np.where(max_dx_for_pair > thres)[0]
    if ind_gap.size == 0:
        return X
    X_new_curves = (X[ind_gap] + X[ind_gap + 1]) / 2.0
    Xnew = np.vstack([X, X_new_curves])
    Xnew_sorted = Xnew[np.argsort(Xnew[:, 0])]
    return Xnew_sorted

def get_init_position_bounds(x_func, timings, w, nslice):
    """ Optimally define bounds of proton initial positions 

    Parameters
    ----------
    x_func : func
        position (cm) as a function of time (s) and initial position (cm)
        
    timings : numpy.ndarray
        array of times to evaluate position
        
    w : float
        slice thickness (cm)
        
    nslice : int
        number of imaging slices

    Returns
    -------
    lower_bound, upper_bound: float
        values for the lowest and highest initial proton positions 
    """

    def does_x_touch_slices(x):
        return ((x < w*nslice) & (x > 0)).any()

    max_iter = 200
    dt = 0.1
    timings = np.arange(timings.min(), timings.max(), dt)
    upper_bound = w*nslice + 0.01
    x = x_func(timings, np.array(upper_bound, ndmin=1))
    counter = 0
    while does_x_touch_slices(x):
        dx_downward = x.min() - upper_bound
        upper_bound += np.abs(dx_downward) / 5
        x = x_func(timings, np.array(upper_bound, ndmin=1))
        counter += 1
        assert counter < max_iter, f'Warning: counter={counter} for finding upper bound exceeded limit'
    lower_bound = -w*nslice
    x = x_func(timings, np.array(lower_bound, ndmin=1))
    counter = 0
    while does_x_touch_slices(x):
        dx_upward = x.max() - lower_bound
        lower_bound -= np.abs(dx_upward) / 5
        x = x_func(timings, np.array(lower_bound, ndmin=1))
        counter += 1
        assert counter < max_iter, f'Warning: counter={counter} for finding lower bound exceeded limit'
    upper_bound *= 1.5
    lower_bound *= 1.5
    return lower_bound, upper_bound

def compute_proton_signal_contribution(iproton, params):
    npulse_total, nslice, Xproton, multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual, offset_fact, varysliceprofile = params
    s_proton_contribution = np.zeros([npulse_total, nslice])
    
    proton_pos = np.repeat(Xproton, multi_factor)
    proton_slice = np.floor(proton_pos / w).astype(int)

    if varysliceprofile:
        # Define the three zones that can affect the proton
        w_offsets = [-w, 0, w]
        pulse_lists = [
            np.where(proton_slice == pulse_slice - 1)[0], # behind
            np.where(proton_slice == pulse_slice)[0],     # target
            np.where(proton_slice == pulse_slice + 1)[0]  # front
        ]
        
        # Only keep 'front' pulses if the proton is still within the stack
        mask = proton_slice[pulse_lists[2]] < nslice
        pulse_lists[2] = pulse_lists[2][mask]
    else:
        pulse_lists = [np.where(proton_slice == pulse_slice)[0]]
        w_offsets = [0]

    # Main Signal Loop
    for pulse_indices, w_offset in zip(pulse_lists, w_offsets):
        tprev = -1.0
        dt_list = np.zeros(pulse_indices.size)
        
        for count, pulse_id in enumerate(pulse_indices):
            t_curr = timings_with_repeats[pulse_id]
            dt_list[count] = t_curr - tprev if count > 0 else 0.0
            tprev = t_curr

            n_seen = count + 1
            
            if varysliceprofile:
                pos_in_slice = (proton_pos[pulse_id] % w) + w_offset
                dist_from_center = pos_in_slice - (w / 2)
                # Gaussian RF scaling
                k = np.exp(-0.5 * (dist_from_center / (w / 2))**2)
                s = fre_signal(n_seen, k * fa, tr, te, t1, t2, dt_list[:n_seen], offset_fact=offset_fact)
            else:
                s = fre_signal(n_seen, fa, tr, te, t1, t2, dt_list[:n_seen], offset_fact=offset_fact)
            target_tr = pulse_tr_actual[pulse_id]
            target_slc = proton_slice[pulse_id]
    
            if 0 <= target_slc < nslice:
                s_proton_contribution[target_tr, target_slc] += s
                
    return s_proton_contribution

def get_pulse_targets(tr, nslice, npulse, alpha):
    tr_vect = np.arange(npulse) * tr
    timing_array = tr_vect + alpha 
    pulse_timing = timing_array.flatten()
    pulse_target_slice = np.repeat(np.arange(nslice), npulse)
    dtype = [('time', float), ('slice', int)]
    combined = np.empty(len(pulse_timing), dtype=dtype)
    combined['time'] = pulse_timing
    combined['slice'] = pulse_target_slice
    combined.sort(order='time')
    return combined['time'], combined['slice']

def match_pulse_to_tr(npulse, nslice):
    return np.repeat(np.arange(npulse), nslice).astype(int)


