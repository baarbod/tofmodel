# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from tofmodel.forward.fresignal import fre_signal_array as fre_signal


def simulate_inflow(tr, npulse, w, fa, t1, nslice, alpha, multi_factor, x_func, x0_array_given=None, progress=False, multithread=True):
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
    
    x0_array_given : numpy.ndarray
        array of proton inital positions (cm) directly supplied instead of computing in routine, by default None
        
    progress : bool, optional
        print information about simulation progress, by default False

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

    # set the proton initial positions
    dx = 0.01
    if x0_array_given is None:
        x0_array = set_init_positions(x_func, tr, w, npulse, nslice, dx, progress=progress)
    elif type(x0_array_given) == np.ndarray:
        x0_array = np.array(x0_array_given)
        
    nproton = np.size(x0_array)
    nproton_per_slice = int(w / dx)

    print('running simulation with ' + str(nproton) + ' protons...')

    # find the time and target slice of each RF pulse
    timings, pulse_slice = get_pulse_targets(tr, nslice, npulse, alpha)

    # associate each pulse to its RF cycle
    pulse_tr_actual = match_pulse_to_tr(npulse, nslice)

    # get the current time before starting the simulation
    t = time.time()

    # initialize
    signal = np.zeros([npulse, nslice])

    params = [(npulse, nslice, x0_array, x_func, timings, 
              multi_factor, w, fa, tr, t1, pulse_slice, pulse_tr_actual)] * nproton
    
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
        
    elapsed = time.time() - t

    elapsed = '{:3.2f}'.format(elapsed)
    print('total simulation time: ' + str(elapsed) + ' seconds')
    print('=================================================================')
    print(' ')

    # Divide after summing contributions; signal is the average of protons
    signal = signal / nproton_per_slice
    return signal


def set_init_positions(x_func, tr, w, npulse, nslice, dx, progress=False):
    """ Defines the protons to be simulated and their initial positions 

    Parameters
    ----------
    x_func : func
        position (cm) as a function of time (s) and initial position (cm)
        
    tr : float
        RF pulse repetition time (s)
        
    w : float
        slice thickness (cm)
        
    npulse : int
        total number of TR cycles to simulate
        
    nslice : int
        number of imaging slices
        
    dx : float
        distance (cm) between adjacent initialized protons
        
    progress : bool, optional
        print information about simulation progress, by default False

    Returns
    -------
    x0_array: numpy.ndarray
        array of initial positions (cm) for each proton
    """
    
    # initialize protons for simulation
    dummyt = np.arange(0, tr*npulse, tr/10)
    print('=================================================================')
    if progress:
        print('Finding initial proton positions...')

    # define range of potential positions
    x0test_range = np.arange(-1500, 1500, 1)

    # initialize
    arrmin = np.zeros(np.size(x0test_range))
    arrmax = np.zeros(np.size(x0test_range))
    protons_to_include = []

    # loop through potential initial positions
    for idx, x0 in enumerate(x0test_range):

        # solve position over time for a given initial position
        x = x_func(dummyt, x0)

        # compute the resulting min and max position
        xmin = np.min(x)
        xmax = np.max(x)

        # store in array
        arrmin[idx] = xmin
        arrmax[idx] = xmax

        # check if proton would flow within slices
        lower_within_bool = 0 < xmin < w * nslice
        upper_within_bool = 0 < xmax < w * nslice
        slice_within_bool = xmin < 0 and xmax > w * nslice
        if lower_within_bool or upper_within_bool or slice_within_bool:
            # record this proton as a candidate
            protons_to_include.append(idx)

    # define bounds with some padding
    xlower = x0test_range[min(protons_to_include) - 5]
    try:
        xupper = x0test_range[max(protons_to_include) + 5]
    except:
        xupper = x0test_range[-1]

    if xupper < xlower:
        print('WARNING: xupper is less than xlower. Check proton initialization')
    if progress:
        print('setting lower bound to ' + str(xlower) + ' cm')
        print('setting upper bound to ' + str(xupper) + ' cm')
    x0_array = np.arange(xlower, xupper, dx)
    return x0_array


def compute_proton_signal_contribution(iproton, params):
    
    npulse, nslice, x0_array, x_func, timings, \
    multi_factor, w, fa, tr, t1, pulse_slice, pulse_tr_actual = params
    
    s_proton_contribution = np.zeros([npulse, nslice])
    
    # get the initial position
    init_pos = x0_array[iproton]

    # compute position at each rf pulse
    proton_position_no_repeats = x_func(np.unique(timings), init_pos)
    proton_position = np.repeat(proton_position_no_repeats, multi_factor)

    # convert absolute positions to slice location
    proton_slice = np.floor(proton_position / w)

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


