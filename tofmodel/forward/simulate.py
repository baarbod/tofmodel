# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from tofmodel.forward.fresignal import fre_signal_array as fre_signal


def simulate_inflow(tr, npulse, w, fa, t1, nslice, alpha, multi_factor, x_func, X_given=None, multithread=False):
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
    timings, pulse_slice = get_pulse_targets(tr, nslice, npulse, alpha)

    # define proton positions over time
    dx = 0.01
    if X_given is None:
        tstart_pos = time.time()
        X = compute_positions_over_time(x_func, np.unique(timings), w, nslice, dx)
        elapsed = time.time() - tstart_pos
        elapsed = '{:3.2f}'.format(elapsed)
        print('position calculation time: ' + str(elapsed) + ' seconds')
    elif type(X_given) == np.ndarray:
        X = np.array(X_given)
        print('Using given proton positions. Skipping calculation...')
    print('=================================================================')
    print(' ')

    # repeated X array because positions are equal with simoultaneous RF pulses                    
    Xrepeated = np.repeat(X, multi_factor, axis=1)
    nproton = X.shape[0]
    nproton_per_slice = int(w / dx)
    print('running simulation with ' + str(nproton) + ' protons...')

    # associate each pulse to its RF cycle
    pulse_tr_actual = match_pulse_to_tr(npulse, nslice)

    # get the current time before starting the simulation
    tstart_sim = time.time()

    # initialize
    signal = np.zeros([npulse, nslice])

    # define params tuple the same for each proton
    params = [(npulse, nslice, Xrepeated, timings, w, fa, tr, t1, pulse_slice, pulse_tr_actual)] * nproton
    
    # run simulation
    if multithread:
        from multiprocessing import Pool
        import os
        
        num_usable_cpu = len(os.sched_getaffinity(0))
        protons_per_core = nproton / num_usable_cpu
        num_turnover = 4
        optimal_chunksize = int(protons_per_core / num_turnover)
        
        print(f"using {num_usable_cpu} cpu cores")
        
        with Pool() as pool:
            result = pool.starmap(compute_proton_signal_contribution, enumerate(params), chunksize=optimal_chunksize)    
        
        for s in result:
            signal += s
    else:
        print(f"using 1 cpu core")
        for iproton in range(nproton):
            signal += compute_proton_signal_contribution(iproton, params[0])

    elapsed = time.time() - tstart_sim

    elapsed = '{:3.2f}'.format(elapsed)
    print('total simulation time: ' + str(elapsed) + ' seconds')
    print('=================================================================')
    print(' ')

    # Divide after summing contributions; signal is the average of protons
    signal = signal / nproton_per_slice
    return signal


def compute_positions_over_time(x_func, timings, w, nslice, dx):
    """ Define range of protons to initialize and compute their positions over time

    Parameters
    ----------
    x_func : func
        position (cm) as a function of time (s) and initial position (cm)
        
    timings : numpy.ndarray
        array of RF pulse timings
        
    w : float
        slice thickness (cm)
        
    nslice : int
        number of imaging slices
        
    dx : float
        distance (cm) between adjacent initialized protons
    
    Returns
    -------
    X: numpy.ndarray
        array of positions (cm) for each proton at each RF pulse timing
    """
    
    def does_x_touch_slices(X):
        lower_above_slc1 = X.min(axis=1) > 0
        lower_below_slc_last = X.min(axis=1) < w * nslice
        upper_above_slc1 = X.max(axis=1) > 0
        upper_below_slc_last = X.max(axis=1) < w * nslice
        slice_within_lower = X.min(axis=1) < 0
        slice_within_upper = X.max(axis=1) > w * nslice
        condition1 = lower_above_slc1 & lower_below_slc_last
        condition2 = upper_above_slc1 & upper_below_slc_last
        condition3 = slice_within_lower & slice_within_upper
        does_touch = condition1 | condition2 | condition3
        return does_touch

    upper_bound = w*nslice
    does_touch = True
    x = x_func(timings, np.array(upper_bound, ndmin=1))
    while does_touch:
        dx_downward = x.min() - upper_bound
        dx_downward /= 10
        upper_bound += np.abs(dx_downward)
        does_touch = does_x_touch_slices(x)
        x = x_func(timings, np.array(upper_bound, ndmin=1))
        
    lower_bound = 0
    does_touch = True
    x = x_func(timings, np.array(lower_bound, ndmin=1))
    while does_touch:
        dx_upward = x.max() - lower_bound
        dx_upward /= 10
        lower_bound -= np.abs(dx_upward)
        x = x_func(timings, np.array(lower_bound, ndmin=1))
        does_touch = does_x_touch_slices(x)

    upper_bound += 2
    lower_bound -= 2
    
    x0test_range = np.arange(lower_bound, upper_bound, dx)
    X = x_func(timings, x0test_range)
    
    return X


def compute_proton_signal_contribution(iproton, params):
    
    npulse, nslice, Xrepeated, timings, w, fa, tr, t1, pulse_slice, pulse_tr_actual = params
    
    s_proton_contribution = np.zeros([npulse, nslice])

    # convert absolute positions to slice location
    proton_slice = np.floor(Xrepeated[iproton, :] / w)

    # find pulses that this proton received
    match_cond = pulse_slice == proton_slice
    pulse_recieve_ind = np.where(match_cond)[0]

    # loop through recieved pulses and compute flow signal
    tprev = float('nan')

    # loop through each pulse and compute signal
    dt_list = np.zeros(np.size(pulse_recieve_ind))
    for count, pulse_id in enumerate(pulse_recieve_ind):
        tnow = timings[pulse_id]
        dt = tnow - tprev  # correct dt behavior on 1st pulse
        tprev = tnow
        dt_list[count] = dt
        npulse = count + 1  # correcting for zero-based numbering

        s = fre_signal(npulse, fa, tr, t1, dt_list)

        current_tr = pulse_tr_actual[pulse_id]
        current_slice = proton_slice[pulse_id]
        current_slice = current_slice.astype(int)
        s_proton_contribution[current_tr, current_slice] += s 
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
    timings = np.array(pulse_slice_tuple_unpacked[0])
    pulse_slice = np.array(pulse_slice_tuple_unpacked[1])
    return timings, pulse_slice


def match_pulse_to_tr(npulse, nslice):
    # define array of TR cycles that each pulse belongs to
    pulse_tr_actual = []
    for ipulse in range(npulse):
        tr_block = ipulse * np.ones(nslice)
        pulse_tr_actual = np.append(pulse_tr_actual, tr_block)
    pulse_tr_actual = pulse_tr_actual.astype(int)
    return pulse_tr_actual


