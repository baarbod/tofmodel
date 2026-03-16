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
                    dx=0.005, offset_fact=1, varysliceprofile=True, X_given=None, ncpu=1, enable_logging=False):
    
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
    timings_with_repeats = timings_with_repeats.astype(np.float32)
    
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
        X = X.astype(np.float32)
        logger.info(f'position calculation time: {time.time() - tstart_pos:.2f} seconds')
    else:
        X = np.array(X_given, dtype=np.float32)
        logger.info('using given proton positions. Skipping calculation...')
    nproton = X.shape[0]
    
    matrix_size_gb = (npulse * nslice * 8) / (1024**3)
    total_estimated_gb = nproton * matrix_size_gb
    logger.info(f'Total estimate memory: {total_estimated_gb:.2f}GB')
    
    logger.info('running simulation with ' + str(nproton) + ' protons...')
    pulse_tr_actual = match_pulse_to_tr(npulse, nslice)
    tstart_sim = time.time()
    signal = np.zeros([npulse, nslice], dtype=np.float32)
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


# def compute_proton_signal_contribution(iproton, params):
#     npulse_total, nslice, Xproton, multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual, offset_fact, varysliceprofile = params
#     s_proton_contribution = np.zeros([npulse_total, nslice], dtype=np.float32)
    
#     proton_pos = np.repeat(Xproton, multi_factor)
#     proton_slice = np.floor(proton_pos / w).astype(np.int16)

#     if varysliceprofile:
#         # Define the three zones that can affect the proton
#         w_offsets = np.array([-w, 0, w], dtype=np.float32)
#         pulse_lists = [
#             np.where(proton_slice == pulse_slice - 1)[0], # behind
#             np.where(proton_slice == pulse_slice)[0],     # target
#             np.where(proton_slice == pulse_slice + 1)[0]  # front
#         ]
        
#         mask = proton_slice[pulse_lists[2]] < nslice
#         pulse_lists[2] = pulse_lists[2][mask]
#     else:
#         pulse_lists = [np.where(proton_slice == pulse_slice)[0]]
#         w_offsets = [0]

#     for pulse_indices, w_offset in zip(pulse_lists, w_offsets):
#         tprev = -1.0
#         dt_list = np.zeros(pulse_indices.size)
        
#         for count, pulse_id in enumerate(pulse_indices):
#             t_curr = timings_with_repeats[pulse_id]
#             dt_list[count] = t_curr - tprev if count > 0 else 0.0
#             tprev = t_curr
#             n_seen = count + 1
            
#             if varysliceprofile:
#                 pos_in_slice = (proton_pos[pulse_id] % w) + w_offset
#                 dist_from_center = pos_in_slice - (w / 2)
                
#                 # --- SINC PULSE IMPLEMENTATION ---
#                 # Normalize x so that the first zero-crossing is at the slice edge (w/2)
#                 x = (np.pi * dist_from_center) / (w / 2)
                
#                 if dist_from_center == 0:
#                     sinc_val = 1.0
#                 else:
#                     sinc_val = np.sin(x) / x
                
#                 # Apply a Hamming window to truncate tails (standard in MRI)
#                 # This window spans 3 slices total (-1.5w to 1.5w)
#                 window = 0.54 + 0.46 * np.cos(np.pi * dist_from_center / (1.5 * w))
                
#                 k = sinc_val * window
#                 # --------------------------------
                
#                 s = fre_signal(n_seen, k * fa, tr, te, t1, t2, dt_list[:n_seen], offset_fact=offset_fact)
#             else:
#                 s = fre_signal(n_seen, fa, tr, te, t1, t2, dt_list[:n_seen], offset_fact=offset_fact)
            
#             target_tr = pulse_tr_actual[pulse_id]
#             target_slc = proton_slice[pulse_id]
    
#             if 0 <= target_slc < nslice:
#                 s_proton_contribution[target_tr, target_slc] += np.float32(s)
                
#     return s_proton_contribution


def compute_proton_signal_contribution(iproton, params):
    # Unpack parameters
    npulse_total, nslice, Xproton, multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual, offset_fact, varysliceprofile = params
    
    # Initialize output
    s_proton_contribution = np.zeros([npulse_total, nslice], dtype=np.float32)
    
    # Proton Positioning
    proton_pos = np.repeat(Xproton, multi_factor)
    proton_slice = np.floor(proton_pos / w).astype(np.int16)

    # Define Zones (Target slice + 1 neighbor on each side)
    if varysliceprofile:
        w_offsets = np.array([-w, 0, w], dtype=np.float32)
        pulse_lists = [
            np.where(proton_slice == pulse_slice - 1)[0], # behind
            np.where(proton_slice == pulse_slice)[0],     # target
            np.where(proton_slice == pulse_slice + 1)[0]  # front
        ]
        
        # Safety mask for the 'front' slice to ensure we don't exceed array bounds
        mask = proton_slice[pulse_lists[2]] < nslice
        pulse_lists[2] = pulse_lists[2][mask]
        
        # --- BWTP PARAMETERS ---
        # BWTP 5.2 = 5.2 total zero crossings = 2.6 per side.
        # Slice width 'w' is defined by the 1st zero crossing (at w/2).
        # Cutoff distance = 2.6 * (w/2) = 1.3 * w.
        bwtp_cutoff_factor = 1.3 
    else:
        pulse_lists = [np.where(proton_slice == pulse_slice)[0]]
        w_offsets = [0]
        bwtp_cutoff_factor = 0 # Not used

    # Pre-calculate constant decay
    exp_te_t2 = np.exp(-te / t2)

    # --- MAIN PULSE LOOP ---
    for pulse_indices, w_offset in zip(pulse_lists, w_offsets):
        
        # Reset Magnetization for this set of pulses
        # We assume full relaxation (M0=1.0) before the first pulse in this list
        mz_current = 1.0 
        tprev = -1.0 

        for count, pulse_id in enumerate(pulse_indices):
            t_curr = timings_with_repeats[pulse_id]
            
            # 1. RELAXATION (Time Propagation)
            if count == 0:
                dt = 0.0 
            else:
                dt = t_curr - tprev
            
            # Apply T1 recovery: Mz(t) = 1 + (Mz_start - 1) * exp(-dt/T1)
            if dt > 0:
                exp_dt_t1 = np.exp(-dt / t1)
                mz_current = 1.0 + (mz_current - 1.0) * exp_dt_t1
            
            # 2. DETERMINE FLIP ANGLE (Slice Profile)
            # if varysliceprofile:
            #     # Calculate distance from the center of the current pulse's slice
            #     pos_in_slice = (proton_pos[pulse_id] % w) + w_offset
            #     dist_from_center = pos_in_slice - (w / 2)
                
            #     # Check if proton is within the Bandwidth-Time Product cutoff
            #     cutoff_dist = bwtp_cutoff_factor * w
                
            #     if abs(dist_from_center) < cutoff_dist:
            #         # Sinc Calculation
            #         if dist_from_center == 0:
            #             sinc_val = 1.0
            #         else:
            #             # Normalize x so first zero is at w/2
            #             x = (np.pi * dist_from_center) / (w / 2)
            #             sinc_val = np.sin(x) / x
                    
            #         # Hamming Window 
            #         # Scaled to go to the edge of the cutoff (2.6 lobes)
            #         window = 0.54 + 0.46 * np.cos(np.pi * dist_from_center / cutoff_dist)
                    
            #         current_fa = sinc_val * window * fa
            #     else:
            #         current_fa = 0.0 # Outside the pulse bandwidth
            # else:
            #     current_fa = fa

            # 2. DETERMINE FLIP ANGLE (Slice Profile)
            if varysliceprofile:
                # Calculate distance from the center of the current pulse's slice
                # (Your existing relative positioning logic here is brilliant and works perfectly)
                pos_in_slice = (proton_pos[pulse_id] % w) + w_offset
                dist_from_center = pos_in_slice - (w / 2)
                
                # --- SPATIAL FERMI FUNCTION ---
                # 'a' controls the edge sharpness (transition band). 
                # a = w / 20.0 closely mimics a Hamming-windowed Sinc with BWTP 5.2
                a = w / 20.0 
                
                # Calculate the normalized spatial profile (0.0 to 1.0)
                fermi_val = 1.0 / (1.0 + np.exp((abs(dist_from_center) - (w / 2.0)) / a))
                
                # Optimization: Only apply if the proton experiences a meaningful flip angle (> 0.1% of nominal)
                if fermi_val > 0.001:
                    current_fa = fermi_val * fa
                else:
                    current_fa = 0.0
            else:
                current_fa = fa

            # 3. COMPUTE SIGNAL & UPDATE STATE
            # Only process if there is a flip angle (optimization)
            if current_fa != 0:
                sin_alpha = np.sin(current_fa)
                cos_alpha = np.cos(current_fa)

                # Steady State Subtraction (Your specific requirement)
                # Note: Assuming pulse_tr_actual is accessible by pulse_id
                tr_val = pulse_tr_actual[pulse_id] if isinstance(pulse_tr_actual, np.ndarray) else tr
                exp_tr_t1_ss = np.exp(-tr_val / t1)
                
                denom = (1.0 - exp_tr_t1_ss * cos_alpha)
                if abs(denom) > 1e-12:
                    mz_ss = offset_fact * (1.0 - exp_tr_t1_ss) / denom
                else:
                    mz_ss = 0.0

                # Signal = Mz_available * sin(alpha) * T2_decay - SteadyState_term
                # Note: We apply the subtraction exactly as you had it: (Mz - Mz_ss)
                s = sin_alpha * exp_te_t2 * (mz_current - mz_ss)
                
                # Update Mz: Tip the remaining magnetization
                mz_current = mz_current * cos_alpha
            else:
                s = 0.0
                # Mz remains unchanged if FA is 0
            
            # Update history
            tprev = t_curr
            
            # 4. STORE RESULT
            # Map global pulse_id to the output array format
            target_tr_idx = int(pulse_tr_actual[pulse_id]) 
            target_slc = proton_slice[pulse_id]
    
            if 0 <= target_slc < nslice:
                s_proton_contribution[target_tr_idx, target_slc] += np.float32(s)
                
    return s_proton_contribution






## ORIGNAL

# def compute_proton_signal_contribution(iproton, params):
#     npulse_total, nslice, Xproton, multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual, offset_fact, varysliceprofile = params
#     s_proton_contribution = np.zeros([npulse_total, nslice], dtype=np.float32)
    
#     proton_pos = np.repeat(Xproton, multi_factor)
#     proton_slice = np.floor(proton_pos / w).astype(np.int16)

#     if varysliceprofile:
#         # Define the three zones that can affect the proton
#         w_offsets = np.array([-w, 0, w], dtype=np.float32)
#         pulse_lists = [
#             np.where(proton_slice == pulse_slice - 1)[0], # behind
#             np.where(proton_slice == pulse_slice)[0],     # target
#             np.where(proton_slice == pulse_slice + 1)[0]  # front
#         ]
        
#         # Only keep 'front' pulses if the proton is still within the stack
#         mask = proton_slice[pulse_lists[2]] < nslice
#         pulse_lists[2] = pulse_lists[2][mask]
#     else:
#         pulse_lists = [np.where(proton_slice == pulse_slice)[0]]
#         w_offsets = [0]

#     # Main Signal Loop
#     for pulse_indices, w_offset in zip(pulse_lists, w_offsets):
#         tprev = -1.0
#         dt_list = np.zeros(pulse_indices.size)
        
#         for count, pulse_id in enumerate(pulse_indices):
#             t_curr = timings_with_repeats[pulse_id]
#             dt_list[count] = t_curr - tprev if count > 0 else 0.0
#             tprev = t_curr

#             n_seen = count + 1
            
#             if varysliceprofile:
#                 pos_in_slice = (proton_pos[pulse_id] % w) + w_offset
#                 dist_from_center = pos_in_slice - (w / 2)
#                 # Gaussian RF scaling
#                 k = np.exp(-0.5 * (dist_from_center / (w / 2))**2)
#                 s = fre_signal(n_seen, k * fa, tr, te, t1, t2, dt_list[:n_seen], offset_fact=offset_fact)
#             else:
#                 s = fre_signal(n_seen, fa, tr, te, t1, t2, dt_list[:n_seen], offset_fact=offset_fact)
#             target_tr = pulse_tr_actual[pulse_id]
#             target_slc = proton_slice[pulse_id]
    
#             if 0 <= target_slc < nslice:
#                 s_proton_contribution[target_tr, target_slc] += np.float32(s)
                
#     return s_proton_contribution

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


