# -*- coding: utf-8 -*-

import time
import numpy as np
from tofmodel.forward.fresignal import fre_signal_array as fre_signal
from multiprocessing import Pool
import os
import logging


def simulate_inflow(tr, te, npulse, w, fa, t1, t2, nslice, alpha, multi_factor, x_func, 
                    dx=0.01, offset_fact=1, varysliceprofile=True, X_given=None, multithread=False, return_pulse=False, enable_logging=False):
    
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
        method for rf slice profile, either 'ideal' or 'gaussian'
           
    X_given : numpy.ndarray, optional
        array of proton positions (cm) directly supplied instead of computing in routine, by default None

    multithread : bool, optional
        use mutliple CPU cores, by default True

    return_pulse : bool, optional
        return Nmatrix variable 
    
    enable_logging : bool, optional
        turn on logging messages; if off, only log critical messages, by default False
        
    Returns
    -------
    signal : numpy.ndarray
        matrix of signal timeseries (a.u.) for each TR/slice
    
    Nmatrix : numpy.ndarray, optional
        matrix of mean number of pulses for each TR/slice
    
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
    if X_given is None:
        tstart_pos = time.time()
        lower_bound, upper_bound = get_init_position_bounds(x_func, np.unique(timings_with_repeats), w, nslice)
        X = compute_position(x_func, timings_with_repeats, lower_bound, upper_bound, dx)
        logger.info(f"trimming protons that never touch slices")
        mask = np.apply_along_axis(lambda x: ((x < w*nslice) & (x > 0)).any(), 1, X)
        X = X[mask]
        logger.info(f"trimmed position bounds: ({X[0, 0]:.3f}, {X[-1, 0]:.3f}) cm")
        X = increase_proton_density(X, npulse, nslice, w, multi_factor, dx, min_proton_count=5, uptoslc=nslice, enable_logging=enable_logging)
        logger.info(f'position calculation time: {time.time() - tstart_pos:.2f} seconds')
    else:
        X = np.array(X_given)
        logger.info('using given proton positions. Skipping calculation...')
    nproton = X.shape[0]
    logger.info('running simulation with ' + str(nproton) + ' protons...')
    pulse_tr_actual = match_pulse_to_tr(npulse, nslice)
    tstart_sim = time.time()
    signal = np.zeros([npulse, nslice])
    if return_pulse:
        Nmatrix = np.zeros([npulse, nslice])
    params = [(npulse, nslice, X[iproton, :], multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual, offset_fact, varysliceprofile)
                for iproton in range(nproton)]
    if multithread:
        num_usable_cpu = len(os.sched_getaffinity(0))
        optimal_chunksize = int(nproton / (num_usable_cpu * 4))
        logger.info(f"using {num_usable_cpu} cpu cores to compute signal contributions")
        with Pool() as pool:
            result = pool.starmap(compute_proton_signal_contribution, enumerate(params), chunksize=optimal_chunksize)    
        for s in result:
            signal += s
        if return_pulse:
            with Pool() as pool:
                result_pulse = pool.starmap(compute_pulse_contribution, enumerate(params), chunksize=optimal_chunksize)    
            for N in result_pulse:
                Nmatrix += N
    else:
        logger.info(f"using 1 cpu core")
        for iproton in range(nproton):
            signal += compute_proton_signal_contribution(iproton, params[iproton])
        if return_pulse:  
            for iproton, param in enumerate(params):
                Nmatrix += compute_pulse_contribution(iproton, param)                
    logger.info(f'total simulation time: {time.time() - tstart_sim:.2f} seconds')
    num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
    signal /= num_proton_in_slice
    if return_pulse:
        Nmatrix /= num_proton_in_slice
    try:
        assert not np.isnan(signal[:, :nslice]).any(), 'Warning: NaN values found in signal array'
        assert not np.isinf(signal[:, :nslice]).any(), 'Warning: Inf values found in signal array'
    except AssertionError as error:
        print(error)
    if return_pulse:
        return signal, Nmatrix
    else:
        return signal


def compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor):
    num_proton_in_slice = np.zeros((npulse, nslice), dtype=int)
    stride = nslice // multi_factor
    if nslice % multi_factor != 0:
        raise ValueError("nslice must be divisible by multi_factor")
    pos_at_end_of_tr_cycles = X[:, 0::stride]
    proton_slice = np.floor(pos_at_end_of_tr_cycles / w)
    for ipulse in range(npulse):
        all_proton_slice_id = proton_slice[:, ipulse]
        for islice in range(nslice):
            num_proton_in_slice[ipulse, islice] = np.count_nonzero(all_proton_slice_id == islice)
    return num_proton_in_slice


def increase_position_matrix_density(X, thres):
    max_dx_for_pair = np.diff(X, axis=0).max(axis=1)
    ind_gap_above_thres = np.where(max_dx_for_pair > thres)[0]
    npulse = X.shape[1]
    X_new_curves = np.zeros((ind_gap_above_thres.size, npulse))
    for count, idx in enumerate(ind_gap_above_thres):
        X_new_curves[count] = X[idx:idx+2, :].T.mean(axis=1)
    Xnew = np.concatenate([X, X_new_curves])
    sorted_indices = np.argsort(Xnew[:, 0])[::-1]
    Xnew_sorted = Xnew[np.flipud(sorted_indices)] 
    return Xnew_sorted


def compute_position(x_func, timings_with_repeats, lower_bound, upper_bound, dx):
    x0 = np.arange(lower_bound, upper_bound, dx)
    X = x_func(np.unique(timings_with_repeats), x0)
    return X

            
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
    npulse, nslice, Xproton, multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual, offset_fact, varysliceprofile = params
    s_proton_contribution = np.zeros([npulse, nslice])
    proton_slice = np.floor(np.repeat(Xproton, multi_factor) / w)

    if varysliceprofile:
        # new approach - RF pulse effects target slice, and the one before/after     
        pulse_recieve_ind_behind = np.where(proton_slice == pulse_slice-1)[0]
        pulse_recieve_ind_target = np.where(proton_slice == pulse_slice)[0]
        pulse_recieve_ind_front = np.where(proton_slice == pulse_slice+1)[0]
        
        # catch pulses outside slices
        outside_ind = np.where(proton_slice[pulse_recieve_ind_front] >= nslice)[0]
        pulse_recieve_ind_front = np.delete(pulse_recieve_ind_front, outside_ind)
        pulse_recieve_list = [pulse_recieve_ind_behind, pulse_recieve_ind_target, pulse_recieve_ind_front]
        w_offsets = [-w, 0, w]
        
        # flip angle gaussian
        N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2)
        proton_pos = np.repeat(Xproton, multi_factor)
        
    else:
        # old approach - RF pulse only effects target slice
        pulse_recieve_ind_target = np.where(proton_slice == pulse_slice)[0]
        pulse_recieve_list = [pulse_recieve_ind_target]
        w_offsets = [0]

    
    for pulse_recieve_ind, w_offset in zip(pulse_recieve_list, w_offsets):
        tprev = float('nan')
        dt_list = np.zeros(np.size(pulse_recieve_ind))
        for count, pulse_id in enumerate(pulse_recieve_ind):
            dt_list[count] = timings_with_repeats[pulse_id] - tprev
            tprev = timings_with_repeats[pulse_id]
            npulse = count + 1
            
            if varysliceprofile:
                pos_in_slice = proton_pos[pulse_id] % w + w_offset
                k = N(pos_in_slice, w/2, w/2)
                fa_scaled = k * fa
                s = fre_signal(npulse, fa_scaled, tr, te, t1, t2, dt_list, offset_fact=offset_fact)
            else:
                s = fre_signal(npulse, fa, tr, te, t1, t2, dt_list, offset_fact=offset_fact)
            
            s_proton_contribution[pulse_tr_actual[pulse_id], proton_slice[pulse_id].astype(int)] += s
    return s_proton_contribution


def compute_pulse_contribution(iproton, params):
    npulse, nslice, Xproton, multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual, offset_fact, varysliceprofile = params
    pulse_contribution = np.zeros([npulse, nslice])
    proton_slice = np.floor(np.repeat(Xproton, multi_factor) / w)
    pulse_recieve_ind = np.where(proton_slice == pulse_slice)[0]
    tprev = float('nan')
    dt_list = np.zeros(np.size(pulse_recieve_ind))
    for count, pulse_id in enumerate(pulse_recieve_ind):
        dt_list[count] = timings_with_repeats[pulse_id] - tprev
        tprev = timings_with_repeats[pulse_id]
        npulse = count + 1
        pulse_contribution[pulse_tr_actual[pulse_id], proton_slice[pulse_id].astype(int)] += npulse
    return pulse_contribution


def get_pulse_targets(tr, nslice, npulse, alpha):
    tr_vect = np.array(range(npulse)) * tr
    timing_array = tr_vect + alpha
    timing_array = timing_array.T
    pulse_target_slice = np.repeat(np.array(range(nslice)), npulse)
    pulse_timing = np.reshape(timing_array.T, npulse * nslice)
    pulse_slice_tuple = [(pulse_timing[i], pulse_target_slice[i])
                         for i in range(0, len(pulse_timing))]
    pulse_slice_tuple = sorted(pulse_slice_tuple)
    pulse_slice_tuple_unpacked = list(zip(*pulse_slice_tuple))
    timings_with_repeats = np.array(pulse_slice_tuple_unpacked[0])
    pulse_slice = np.array(pulse_slice_tuple_unpacked[1])
    return timings_with_repeats, pulse_slice


def match_pulse_to_tr(npulse, nslice):
    pulse_tr_actual = []
    for ipulse in range(npulse):
        tr_block = ipulse * np.ones(nslice)
        pulse_tr_actual = np.append(pulse_tr_actual, tr_block)
    pulse_tr_actual = pulse_tr_actual.astype(int)
    return pulse_tr_actual


