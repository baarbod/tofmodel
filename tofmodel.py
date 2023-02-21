# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:41:44 2023

@author: Baarbod
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
import posfunclib as pfl
from functools import partial 
import input_handler as ih
#from fre_signal import fre_signal
from fre_signal import fre_signal_array as fre_signal

def run_tof_model(scan_param, Xfunc):
    
    # Scan parameters
    w = scan_param['slice_width']
    TR = scan_param['repetition_time']
    fa = scan_param['flip_angle']*math.pi/180
    T1 = scan_param['t1_time']
    nslice = scan_param['num_slice']
    npulse = scan_param['num_pulse']
    alpha = np.array(scan_param['alpha_list'], ndmin=2).T
    
    assert np.size(alpha) == nslice, 'Warning: size of alpha should be nslice'
    
    # Initialize protons for simulation
    dummyt = np.arange(0, TR*npulse, TR/20)
    x0 = 0
    X = Xfunc(dummyt, x0)
    dx = 0.01
    X0array = np.arange(-max(X), nslice*w, dx)
    nproton = np.size(X0array)
    nproton_per_slice = int(w / dx)
    
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
    
    # Define array of TR cycles that each pulse belongs to
    pulse_tr_actual = []
    for ipulse in range(npulse):
        tr_block = ipulse*np.ones(nslice)
        pulse_tr_actual = np.append(pulse_tr_actual, tr_block)    
    pulse_tr_actual = pulse_tr_actual.astype(int)
    
    t = time.time()
    
    signal = np.zeros([npulse, nslice])
    s_counter = np.zeros([npulse, nslice])
    for iproton in range(nproton):
        
        # Solve position at each pulse for this proton
        init_pos = X0array[iproton]
        proton_position_no_repeats = Xfunc(np.unique(timings), init_pos)
        proton_position = np.repeat(proton_position_no_repeats, 2)
        
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
    print(elapsed)
    
    # # Check conservation of protons
    # err_statement = 'Warning: proton conservation failed - check s_counter.'
    # assert np.all(s_counter == nproton_per_slice), err_statement
        
    # Divide after summing contributions; signal is the average of protons
    signal = signal / nproton_per_slice    
    return signal


# Position functions
#Xfunc = partial(pfl.compute_position_constant, v0=0.1)
#Xfunc = partial(pfl.compute_position_sine, v1=-0.5, v2=0.5, w0=2*np.pi/5)
#Xfunc = partial(pfl.compute_position_sine, v1=0, v2=0.5, w0=2*np.pi/5)
Xfunc = partial(pfl.compute_position_sine_spatial, v1=-0.5, v2=0.5, w0=2*np.pi/5)

# An = [0.2, 0.1]
# Bn = [0*0.1]
# w0 = 2*np.pi/5
# Xfunc = partial(pfl.compute_position_fourier, An=An, Bn=Bn, w0=w0)

#scan_param = ih.input_from_json('test_scan.json')

# Dictionary containing model parameters
scan_param =	{
    'slice_width' : 0.25,
    'repetition_time' : 0.35,
    'flip_angle' : 47,
    't1_time' : 4,
    'num_slice' : 10,
    'num_pulse' : 100,
    'alpha_list' : [0.14, 0, 0.2075, 0.07, 0.2775, 0.14, 0, 0.2075, 0.07, 0.2775]}

signal = run_tof_model(scan_param, Xfunc)

# Plot slice signals
plt.plot(signal[:, 0:4])
plt.show()





    