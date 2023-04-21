# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:41:44 2023

@author: Baarbod
"""

import math
import time
import numpy as np
#from tof.fresignal import fre_signal
from tof.fresignal import fre_signal_array as fre_signal

def get_pulse_targets(scan_param):
    TR = scan_param['repetition_time']
    nslice = scan_param['num_slice']
    npulse = scan_param['num_pulse']
    alpha = np.array(scan_param['alpha_list'], ndmin=2).T
    
    # Define list of tuples defining pulse timings and corresponding target slices
    tr_vect = np.array(range(npulse))*TR
    T = tr_vect + alpha
    T = T.T
    pulse_target_slice = np.repeat(np.array(range(nslice)), npulse)
    pulse_timing = np.reshape(T.T, npulse*nslice)
    pulse_slice_tuple = [(pulse_timing[i], pulse_target_slice[i]) 
                         for i in range(0, len(pulse_timing))]
    pulse_slice_tuple = sorted(pulse_slice_tuple)
    
    # Unpack timing-slice tuple
    pulse_slice_tuple_unpacked = list(zip(*pulse_slice_tuple))
    timings = np.array(pulse_slice_tuple_unpacked[0])
    pulse_slice = np.array(pulse_slice_tuple_unpacked[1])
    return timings, pulse_slice

def match_pulse_to_tr(npulse, nslice):
    # Define array of TR cycles that each pulse belongs to
    pulse_tr_actual = []
    for ipulse in range(npulse):
        tr_block = ipulse*np.ones(nslice)
        pulse_tr_actual = np.append(pulse_tr_actual, tr_block)    
    pulse_tr_actual = pulse_tr_actual.astype(int)
    return pulse_tr_actual

def set_init_positions(Xfunc, TR, w, npulse, nslice, dx):
    # Initialize protons for simulation
    dummyt = np.arange(0, TR*npulse, TR/10)
    print('Finding initial proton positions...')
    forward_flag = 1
    
    # import matplotlib.pyplot as plt
    # fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
    x0test_range = np.arange(-100, 100)
    arrmin = np.zeros(np.size(x0test_range))
    arrmax = np.zeros(np.size(x0test_range))
    for idx, x0 in enumerate(x0test_range):
        # print(x0)
        X = Xfunc(dummyt, x0)
        xmin = np.min(X)
        xmax = np.max(X)
        arrmin[idx] = xmin
        arrmax[idx] = xmax
    # ax1.plot(x0test_range, x0test_range, label='x0')
    # ax1.plot(x0test_range, arrmin - x0test_range, label='xmin - x0')
    # ax1.plot(x0test_range, arrmax - x0test_range, label='xmax - x0')
    # ax2.plot(x0test_range, x0test_range, label='x0')
    # ax2.plot(x0test_range, arrmin, label='xmin')
    # ax2.plot(x0test_range, arrmax, label='xmax')
    # ax1.legend(); ax2.legend(); 
    # plt.show()
    
    offset_max = np.abs(np.mean(arrmax - x0test_range))
    if np.mean(offset_max) < 0.5:
        forward_flag = 0
    else:
        forward_flag = 1
    
    if forward_flag:
        a = np.sign(arrmax)
        b = a == 1
        res = next((i for i, j in enumerate(b) if j), None)
        print("The values till first True value : " + str(res))
        xlower = x0test_range[res] - 5
        xupper = 2*w*nslice
    else:
        min_above = arrmin > w*nslice
        max_within = arrmax > 0
        result = set(i for i, x in enumerate(max_within) if x and not min_above[i])
        xupper = x0test_range[max(result)]
        xlower = -2*w*nslice

    print('setting lower bound to ' + str(xlower) + ' cm')
    print('setting upper bound to ' + str(xupper) + ' cm')
    X0array = np.arange(xlower, xupper, dx)
    return X0array
    

def run_tof_model(scan_param, Xfunc):
    
    # Scan parameters
    w = scan_param['slice_width']
    TR = scan_param['repetition_time']
    fa = scan_param['flip_angle']*math.pi/180
    T1 = scan_param['t1_time']
    nslice = scan_param['num_slice']
    npulse = scan_param['num_pulse']
    MBF = scan_param['MBF']
    alpha = np.array(scan_param['alpha_list'], ndmin=2).T
    
    assert np.size(alpha) == nslice, 'Warning: size of alpha should be nslice'
    
    dx = 0.01
    X0array = set_init_positions(Xfunc, TR, w, npulse, nslice, dx)
    nproton = np.size(X0array)
    nproton_per_slice = int(w / dx)
    
    print('running simulation with ' + str(nproton) + ' protons...')
    
    # find the time and target slice of each RF pulse
    timings, pulse_slice = get_pulse_targets(scan_param)
    
    # associate each pulse to its RF cycle
    pulse_tr_actual = match_pulse_to_tr(npulse, nslice)
    
    t = time.time()

    signal = np.zeros([npulse, nslice])
    s_counter = np.zeros([npulse, nslice])
    halfway_flag = 0
    
    for iproton in range(nproton):
        
        if iproton/nproton > 0.5 and not halfway_flag:
            halfway_flag = 1
            print('half way done at ' + str(time.time()-t) + ' seconds')
        
        # Solve position at each pulse for this proton
        init_pos = X0array[iproton]
    
        proton_position_no_repeats = Xfunc(np.unique(timings), init_pos)
        proton_position = np.repeat(proton_position_no_repeats, MBF)
        
        # Convert absolute positions to slice location
        proton_slice = np.floor(proton_position/w)
        
        # Find pulses that this proton recieved
        match_cond = pulse_slice == proton_slice
        pulse_recieve_ind = np.where(match_cond)[0]
        
        # Loop through recieved pulses and compute flow signal
        tprev = float('nan')
        dt_list = np.zeros(np.size(pulse_recieve_ind))
        for count, pulse_id in enumerate(pulse_recieve_ind):
             tnow = timings[pulse_id]
             dt = tnow - tprev # correct dt behavior on 1st pulse
             tprev = tnow
             dt_list[count] = dt
             npulse = count+1 # correcting for zero-based numbering
             S = fre_signal(npulse, fa, TR, T1, dt_list)
             current_tr = pulse_tr_actual[pulse_id]
             current_slice = proton_slice[pulse_id]
             current_slice = current_slice.astype(int)
             signal[current_tr, current_slice] += S
             s_counter[current_tr, current_slice] += 1
    
    elapsed = time.time() - t
    print('total simulation time: ' + str(elapsed))
    print(' ')
    
    # # Check conservation of protons
    # err_statement = 'Warning: proton conservation failed - check s_counter.'
    # assert np.all(s_counter == nproton_per_slice), err_statement
        
    # Divide after summing contributions; signal is the average of protons
    signal = signal / nproton_per_slice    
    return signal






    