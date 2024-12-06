# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from tofmodel.forward.fresignal import fre_signal_array as fre_signal
from multiprocessing import Pool
import os


def simulate_inflow(tr, npulse, w, fa, t1, nslice, alpha, multi_factor, x_func, dx=0.01, X_given=None, multithread=False):
    """ Routine for simulating inflow signals

    Parameters
    ----------
    tr : float
        RF pulse repetition time (s)
        
    npulse : int
        total number of TR cycles to simulate
        
    w : float
        slice thickness (cm)
        
    fa : float
        flip angle (degrees)
        
    t1 : float
        T1 time constant for flowing fluid (s) 
        
    nslice : int
        number of imaging slices
        
    alpha : list
        slice timing for each slice which should have length same as nslice (s)
        
    multi_factor : int
        multi-band factor
    
    dx : float
        distance between initalized protons (cm), by default 0.01
        
    x_func : func
        position (cm) as a function of time (s) and initial position (cm)
    
    X_given : numpy.ndarray
        array of proton positions (cm) directly supplied instead of computing in routine, by default None

    multithread : bool, optional
        use mutliple CPU cores, by default True

    Returns
    -------
    signal : numpy.ndarray
        matrix of signal timeseries (a.u.) for each slice
    """
    
    # process scan parameters
    fa = fa * np.pi / 180
    alpha = np.array(alpha, ndmin=2).T

    assert np.size(alpha) == nslice, 'Warning: size of alpha should be nslice'
    
    # find the time and target slice of each RF pulse
    timings_with_repeats, pulse_slice = get_pulse_targets(tr, nslice, npulse, alpha)
    print(' ')
    # define proton positions over time
    if X_given is None:
        tstart_pos = time.time()
        lower_bound, upper_bound = get_init_position_bounds(x_func, np.unique(timings_with_repeats), w, nslice)
        X = compute_position(x_func, timings_with_repeats, lower_bound, upper_bound, dx)
        
        # trim initialized protons that don't touch slices
        print(f"trimming protons that never touch slices")
        def does_x_touch_slices(x):
            return ((x < w*nslice) & (x > 0)).any()
        proton_to_remove = []
        for iproton in range(X.shape[0]):
            if not does_x_touch_slices(X[iproton, :]):
                proton_to_remove.append(iproton)
        X = np.delete(X, proton_to_remove, axis=0)
        print(f"found {len(proton_to_remove)} protons to be trimmed")
        print(f"trimmed upper bound: {X[-1, 0]} cm")
        print(f"trimmed lower bound: {X[0, 0]} cm")
        
        X = increase_proton_density(X, npulse, nslice, w, multi_factor, dx, min_proton_count=5)
        print(f'position calculation time: {time.time() - tstart_pos:.2f} seconds')
    else:
        X = np.array(X_given)
        print('using given proton positions. Skipping calculation...')
    
    nproton = X.shape[0]
    print('running simulation with ' + str(nproton) + ' protons...')

    # match pulses to their RF cycles
    pulse_tr_actual = match_pulse_to_tr(npulse, nslice)

    tstart_sim = time.time()

    # initialize signal array
    signal = np.zeros([npulse, nslice])

    # setup parameters for each proton
    params = [(npulse, nslice, X[iproton, :], multi_factor, timings_with_repeats, w, fa, tr, t1, pulse_slice, pulse_tr_actual)
                for iproton in range(nproton)]
    
    # run simulation
    if multithread:
        num_usable_cpu = len(os.sched_getaffinity(0))
        optimal_chunksize = int(nproton / (num_usable_cpu * 4))
        
        print(f"using {num_usable_cpu} cpu cores")
        with Pool() as pool:
            result = pool.starmap(compute_proton_signal_contribution, enumerate(params), chunksize=optimal_chunksize)    
        
        for s in result:
            signal += s
    else:
        print(f"using 1 cpu core")
        for iproton in range(nproton):
            signal += compute_proton_signal_contribution(iproton, params[iproton])
                
    print(f'total simulation time: {time.time() - tstart_sim:.2f} seconds')
    print(' ')
    
    num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
    signal /= num_proton_in_slice
    assert not np.isnan(signal).any(), 'Warning: NaN values found in signal array'
    
    return signal


def compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor):
    # compute the number of spins in each slice over each tr
    num_proton_in_slice = np.zeros((npulse, nslice), dtype=int)
    pos_at_end_of_tr_cycles = X[:, 0::int(nslice/multi_factor)]
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


def increase_proton_density(X, npulse, nslice, w, multi_factor, dx, min_proton_count=1, maxiter=200):
    
    # increase density until no proton count is below the minimum allowance
    num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
    ind_sparse = np.where(num_proton_in_slice < min_proton_count)
    if ind_sparse[0].size > 0:
        print(f"initial proton count is {X.shape[0]}")
        print(f"found {ind_sparse[0].size} TR cycles with less than {min_proton_count} protons")
        print(f"running density increase algorithm...")
        count = 0
        while ind_sparse[0].size > 0:
            count += 1
            if count > maxiter:
                print(f"WARNING: reached max count iter of {maxiter}")
                print(f"{ind_sparse[0].size} TR cycles have less than {min_proton_count} protons")
                print(f"minimum TR cycle proton count is {num_proton_in_slice.min()}")
                return X
            thres = w / min_proton_count # threshold distance between adjacent position curves for increasing proton density
            X = increase_position_matrix_density(X, thres) 
            num_proton_in_slice = compute_slice_pulse_particle_counts(X, npulse, nslice, w, multi_factor)
            ind_sparse = np.where(num_proton_in_slice < min_proton_count)
        print(f"finished after {count} iterations")
        print(f"{ind_sparse[0].size} TR cycles have less than {min_proton_count} protons")
        print(f"minimum TR cycle proton count is {num_proton_in_slice.min()}")

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
    
    # need to make sure timings array is linear ramp because otherwise issues can arise.
    # need to update docstring. position during simulation only needs to be evaluated as pulses, 
    # but for determining position bounds we need to know in between positions in cases where pulses
    # are very spaced out
    dt = 0.1
    timings = np.arange(timings.min(), timings.max(), dt)
    
    # upper_bound = 2*w*nslice + 0.01
    upper_bound = w*nslice + 0.01
    x = x_func(timings, np.array(upper_bound, ndmin=1))
    counter = 0
    while does_x_touch_slices(x):
        dx_downward = x.min() - upper_bound
        upper_bound += np.abs(dx_downward) / 5
        x = x_func(timings, np.array(upper_bound, ndmin=1))
        counter += 1
        assert counter < max_iter, f'Warning: counter={counter} for finding upper bound exceeded limit'
    
    # lower_bound = -2*w*nslice
    lower_bound = -w*nslice
    x = x_func(timings, np.array(lower_bound, ndmin=1))
    counter = 0
    while does_x_touch_slices(x):
        dx_upward = x.max() - lower_bound
        lower_bound -= np.abs(dx_upward) / 5
        x = x_func(timings, np.array(lower_bound, ndmin=1))
        counter += 1
        assert counter < max_iter, f'Warning: counter={counter} for finding lower bound exceeded limit'
    
    # place buffers on bounds just to be sure
    upper_bound *= 1.5
    lower_bound *= 1.5
    print(f"initial upper bound: {upper_bound} cm")
    print(f"initial lower bound: {lower_bound} cm")
    
    return lower_bound, upper_bound


def compute_proton_signal_contribution(iproton, params):
    
    npulse, nslice, Xproton, multi_factor, timings_with_repeats, w, fa, tr, t1, pulse_slice, pulse_tr_actual = params
    s_proton_contribution = np.zeros([npulse, nslice])

    # convert absolute positions to slice location
    proton_slice = np.floor(np.repeat(Xproton, multi_factor) / w)
    pulse_recieve_ind = np.where(proton_slice == pulse_slice)[0]

    # # flip angle gaussian
    # N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2)
    # proton_pos = np.repeat(Xproton, multi_factor)

    tprev = float('nan')
    dt_list = np.zeros(np.size(pulse_recieve_ind))
    
    for count, pulse_id in enumerate(pulse_recieve_ind):
        dt_list[count] = timings_with_repeats[pulse_id] - tprev
        tprev = timings_with_repeats[pulse_id]

        # # slice selection profile hack
        # pos_in_slice = proton_pos[pulse_id] % w
        # k = N(pos_in_slice, w/2, w/3)
        # fa_scaled = k * fa
        
        npulse = count + 1  # correcting for zero-based numbering
        s = fre_signal(npulse, fa, tr, t1, dt_list)
        s_proton_contribution[pulse_tr_actual[pulse_id], proton_slice[pulse_id].astype(int)] += s
    return s_proton_contribution


def get_pulse_targets(tr, nslice, npulse, alpha):

    # Define list of tuples defining pulse timings and corresponding target slices
    tr_vect = np.array(range(npulse)) * tr
    timing_array = tr_vect + alpha
    timing_array = timing_array.T

    pulse_target_slice = np.repeat(np.array(range(nslice)), npulse)
    pulse_timing = np.reshape(timing_array.T, npulse * nslice)
    pulse_slice_tuple = [(pulse_timing[i], pulse_target_slice[i])
                         for i in range(0, len(pulse_timing))]
    pulse_slice_tuple = sorted(pulse_slice_tuple)

    # unpack timing-slice tuple
    pulse_slice_tuple_unpacked = list(zip(*pulse_slice_tuple))
    timings_with_repeats = np.array(pulse_slice_tuple_unpacked[0])
    pulse_slice = np.array(pulse_slice_tuple_unpacked[1])
    return timings_with_repeats, pulse_slice


def match_pulse_to_tr(npulse, nslice):
    # define array of TR cycles that each pulse belongs to
    pulse_tr_actual = []
    for ipulse in range(npulse):
        tr_block = ipulse * np.ones(nslice)
        pulse_tr_actual = np.append(pulse_tr_actual, tr_block)
    pulse_tr_actual = pulse_tr_actual.astype(int)
    return pulse_tr_actual


