import time
import numpy as np
from multiprocessing import Pool
import os
import logging
import math
from functools import partial
import numba
from numba import njit, prange

def simulate_inflow(tr, te, npulse, w, fa, t1, t2, nslice, alpha, multi_factor, x_func, 
                    dx=0.005, offset_fact=0, varysliceprofile=True, X_given=None, ncpu=1, enable_logging=False):
    logger = logging.getLogger(__name__)
    if enable_logging:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.CRITICAL + 1)
    fa = fa * np.pi / 180.0
    alpha = np.array(alpha, ndmin=2).T
    assert np.size(alpha) == nslice, 'Warning: size of alpha should be nslice'
    timings_with_repeats, pulse_slice = get_pulse_targets(tr, nslice, npulse, alpha)
    timings_with_repeats = timings_with_repeats.astype(np.float32)
    cpu_count = len(os.sched_getaffinity(0))
    use_cores = cpu_count if ncpu == -1 else int(ncpu)
    numba.set_num_threads(use_cores)
    if X_given is None:
        tstart_pos = time.time()
        lower_bound, upper_bound = get_init_position_bounds(x_func, np.unique(timings_with_repeats), w, nslice)
        if use_cores > 1:
            X = compute_position_parallel(x_func, timings_with_repeats, lower_bound, upper_bound, dx, use_cores)
        else:
            X = compute_position(x_func, timings_with_repeats, lower_bound, upper_bound, dx)
        logger.info(f"Trimming protons that never touch slices")
        mask = np.any((X > 0) & (X < w * nslice), axis=1)
        X = X[mask]
        logger.info(f"Trimmed position bounds: ({X[0, 0]:.3f}, {X[-1, 0]:.3f}) cm")
        X = increase_proton_density(X, npulse, nslice, w, multi_factor, dx, min_proton_count=5, uptoslc=10, enable_logging=enable_logging)
        X = X.astype(np.float32)
        logger.info(f'Position calculation time: {time.time() - tstart_pos:.2f} seconds')
    else:
        X = np.array(X_given, dtype=np.float32)
        logger.info('Using given proton positions. Skipping calculation...')
    nproton = X.shape[0]
    logger.info(f'Running simulation with {nproton} protons using {use_cores} Numba threads...')
    X_expanded = np.repeat(X, multi_factor, axis=1).astype(np.float32)
    pulse_tr_actual = match_pulse_to_tr(npulse, nslice).astype(np.int32)
    pulse_slice = pulse_slice.astype(np.int32)
    tstart_sim = time.time()
    signals_all = compute_all_signals_numba(
        X_expanded, npulse, nslice, timings_with_repeats, 
        float(w), float(fa), float(tr), float(te), float(t1), float(t2), 
        pulse_slice, pulse_tr_actual, float(offset_fact), bool(varysliceprofile)
    )
    signal = np.sum(signals_all, axis=0)
    logger.info(f'Total simulation time: {time.time() - tstart_sim:.2f} seconds')
    num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
    signal = np.divide(
        signal, 
        num_proton_in_slice.astype(np.float32), 
        out=np.zeros_like(signal), 
        where=num_proton_in_slice != 0
    )
    if np.isnan(signal[:, :nslice]).any() or np.isinf(signal[:, :nslice]).any():
        raise ValueError('Invalid numerical bounds (NaN/Inf) detected in signal matrix.')
    return signal

@njit(parallel=True, fastmath=True)
def compute_all_signals_numba(X_expanded, npulse_total, nslice, timings_with_repeats, 
                              w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual, 
                              offset_fact, varysliceprofile):
    nproton = X_expanded.shape[0]
    N_targets = len(timings_with_repeats)
    signals_all = np.zeros((nproton, npulse_total, nslice), dtype=np.float32)
    exp_te_t2 = math.exp(-te / t2)
    exp_tr_t1_ss = math.exp(-tr / t1)
    for i in prange(nproton):
        proton_pos = X_expanded[i, :]
        t_entry = -1.0
        for p_id in range(N_targets):
            p_pos = proton_pos[p_id]
            p_slc = int(math.floor(p_pos / w))
            target_slc = pulse_slice[p_id]
            is_valid = False
            if varysliceprofile:
                if (p_slc == target_slc - 1) or (p_slc == target_slc) or (p_slc == target_slc + 1 and p_slc < nslice):
                    is_valid = True
            else:
                if p_slc == target_slc:
                    is_valid = True
                    
            if is_valid and (0 <= p_slc < nslice):
                t_entry = timings_with_repeats[p_id]
                break
        if t_entry < 0:
            continue
        mz_current = 1.0 
        tprev = -1.0
        for p_id in range(N_targets):
            p_pos = proton_pos[p_id]
            p_slc = int(math.floor(p_pos / w))
            target_slc = pulse_slice[p_id]
            w_offset = 0.0
            is_valid = False
            if varysliceprofile:
                if p_slc == target_slc - 1:
                    w_offset = -w
                    is_valid = True
                elif p_slc == target_slc:
                    w_offset = 0.0
                    is_valid = True
                elif p_slc == target_slc + 1 and p_slc < nslice:
                    w_offset = w
                    is_valid = True
            else:
                if p_slc == target_slc:
                    is_valid = True  
            if not is_valid:
                continue
            t_curr = timings_with_repeats[p_id]
            if t_curr < (t_entry - 3.0 * t1):
                continue
            dt = 0.0 if tprev < 0 else (t_curr - tprev)
            if dt > 0:
                mz_current = 1.0 + (mz_current - 1.0) * math.exp(-dt / t1)
            if varysliceprofile:
                pos_in_slice = (p_pos % w) + w_offset
                dist_from_center = pos_in_slice - (w / 2.0)
                a = w / 20.0 
                fermi_val = 1.0 / (1.0 + math.exp((abs(dist_from_center) - (w / 2.0)) / a))
                current_fa = fermi_val * fa if fermi_val > 0.001 else 0.0
            else:
                current_fa = fa 
            if current_fa != 0.0:
                sin_alpha = math.sin(current_fa)
                cos_alpha = math.cos(current_fa)
                denom = (1.0 - exp_tr_t1_ss * cos_alpha)
                mz_ss = offset_fact * (1.0 - exp_tr_t1_ss) / denom if abs(denom) > 1e-12 else 0.0
                s = sin_alpha * exp_te_t2 * (mz_current - mz_ss)
                mz_current = mz_current * cos_alpha
            else:
                s = 0.0  
            tprev = t_curr
            target_tr_idx = pulse_tr_actual[p_id]
            if 0 <= p_slc < nslice:
                signals_all[i, target_tr_idx, p_slc] += np.float32(s)
    return signals_all

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
    logger = logging.getLogger(__name__)
    if enable_logging:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.CRITICAL + 1)
    num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
    ind_sparse = np.where(num_proton_in_slice[:, :uptoslc] < min_proton_count)
    if ind_sparse[0].size > 0:
        logger.info(f"Initial proton count is {X.shape[0]}")
        count = 0
        while ind_sparse[0].size > 0:
            count += 1
            if count > maxiter:
                logger.info(f"WARNING: reached max count iter of {maxiter}")
                return X
            thres = w / min_proton_count
            X = increase_position_matrix_density(X, thres) 
            num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
            ind_sparse = np.where(num_proton_in_slice < min_proton_count)
        logger.info(f"Finished after {count} iterations")
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
    return Xnew[np.argsort(Xnew[:, 0])]

def get_init_position_bounds(x_func, timings, w, nslice):
    def does_x_touch_slices(x):
        return ((x < w*nslice) & (x > 0)).any()
    max_iter = 200
    dt = 0.25 
    timings = np.arange(timings.min(), timings.max(), dt)
    upper_bound = w*nslice + 0.01
    x = x_func(timings, np.array(upper_bound, ndmin=1))
    counter = 0
    while does_x_touch_slices(x):
        dx_downward = x.min() - upper_bound
        upper_bound += np.abs(dx_downward) / 5
        x = x_func(timings, np.array(upper_bound, ndmin=1))
        counter += 1
        if counter >= max_iter:
            raise RuntimeError(f'Counter={counter} for finding upper bound exceeded limit')  
    lower_bound = -w*nslice
    x = x_func(timings, np.array(lower_bound, ndmin=1))
    counter = 0
    while does_x_touch_slices(x):
        dx_upward = x.max() - lower_bound
        lower_bound -= np.abs(dx_upward) / 5
        x = x_func(timings, np.array(lower_bound, ndmin=1))
        counter += 1
        if counter >= max_iter:
            raise RuntimeError(f'Counter={counter} for finding lower bound exceeded limit')     
    return lower_bound * 1.5, upper_bound * 1.5

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