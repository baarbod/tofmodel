# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from forward.fresignal import fre_signal_array as fre_signal


def run_tof_model(scan_param, x_func, showplot=False, progress=False):
    """Routine for simulating inflow signals

    Parameters
    ----------
    scan_param : dict
        Scanning parameters
        
    x_func : func
        Function of x and x0 used to define position dependance of spins
        
    showplot : bool, optional
        show plots related to spin position initialization, by default False
        
    progress : bool, optional
        print information about simulation progress, by default False

    Returns
    -------
    signal : numpy.ndarray
        matrix of signal timeseries for each slice
    """
    
    
    # define scan parameters
    w = scan_param['slice_width']
    tr = scan_param['repetition_time']
    fa = scan_param['flip_angle'] * np.pi / 180
    t1 = scan_param['t1_time']
    nslice = scan_param['num_slice']
    npulse = scan_param['num_pulse']
    multi_factor = scan_param['MBF']
    alpha = np.array(scan_param['alpha_list'], ndmin=2).T

    assert np.size(alpha) == nslice, 'Warning: size of alpha should be nslice'

    # set the proton initial positions
    dx = 0.01
    x0_array = set_init_positions(x_func, tr, w, npulse, nslice, dx, showplot=showplot, progress=progress)
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
    s_counter = np.zeros([npulse, nslice])

    # controls how often to display progress message
    fraction = nproton / 10
    
    # main loop which computes proton signal contributions
    for iproton in range(nproton):

        # display progress message
        if progress:
            if (iproton % fraction) == 0:
                tnow = time.time()-t
                tstr = '{:3.2f}'.format(tnow)
                string = str(iproton) + ' protons at ' + tstr + ' seconds'
                print(string)

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
        s_for_proton = []
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
            signal[current_tr, current_slice] += s
            s_counter[current_tr, current_slice] += 1
            s_for_proton.append(s)

    elapsed = time.time() - t

    elapsed = '{:3.2f}'.format(elapsed)
    print('total simulation time: ' + str(elapsed) + ' seconds')
    print('=================================================================')
    print(' ')

    # Divide after summing contributions; signal is the average of protons
    signal = signal / nproton_per_slice
    return signal


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


def set_init_positions(x_func, tr, w, npulse, nslice, dx,
                       showplot=True, progress=False):
    # initialize protons for simulation
    dummyt = np.arange(0, tr*npulse, tr/10)
    print('=================================================================')
    if progress:
        print('Finding initial proton positions...')

    if showplot:
        fig, ax = plt.subplots(nrows=1, ncols=1)

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

    if showplot:
        ax.plot(x0test_range, x0test_range, label='x0')
        ax.plot(x0test_range, arrmin, label='xmin')
        ax.plot(x0test_range, arrmax, label='xmax')
        for i in protons_to_include:
            ax.axvline(x0test_range[i], linestyle=':', color='black')
        ax.axhline(0, linestyle='--', color='gray')
        ax.axhline(w * nslice, linestyle='--', color='gray')
        ax.legend()

        ax.set_xlim(xlower, xupper)
        ax.set_ylim(xlower, xupper)
        plt.show()

    if xupper < xlower:
        print('WARNING: xupper is less than xlower. Check proton initialization')
    if progress:
        print('setting lower bound to ' + str(xlower) + ' cm')
        print('setting upper bound to ' + str(xupper) + ' cm')
    x0_array = np.arange(xlower, xupper, dx)
    return x0_array