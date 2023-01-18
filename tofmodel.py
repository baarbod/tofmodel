# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:41:44 2023

@author: Baarbod
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

# Position as a function of time
def compute_position(t, x0, v0):
    #X = v0*t + x0
    
    v1 = 0
    v2 = v0
    Amp = (v2-v1)/2
    A0 = (v1 + Amp)*2
    An = Amp
    w0 = 2*math.pi/5
    X = A0*t/2 + An/w0*np.sin(w0*t) + x0 
    return X

# Define equation for flow-enhanced fMRI signal
def fre_signal(n, fa, TR, T1, dt_list):
    # e1 = math.exp(-TR/T1)
    # S = math.sin(fa)*(e1*math.cos(fa))**(n-1)*(1 - (1-e1)/(1-e1*math.cos(fa)))
    
    M0 = 1
    C = np.cos(fa)
    Mzss = M0*(1 - math.exp(-TR/T1)) / (1 - math.exp(-TR/T1)*C)
    positive_series = np.zeros(n)
    negative_series = np.zeros(n)
    E = np.exp(-dt_list/T1)
    
    if n == 1:
        Mzn_pre = M0*C
    else:
        for m in range(n):
            positive_series[m] = C**m * np.prod(E[1:n-1])/np.prod(E[1:n-m-1]) 
        for m in range(n-1):
            negative_series[m] = C**m * np.prod(E[1:n-1])/np.prod(E[1:n-m-2])
        Mzn_pre = M0 * (np.sum(positive_series) - np.sum(negative_series))
    
    # if n == 1:
    #     Mzn_pre = M0*C
    # else:
    #     temp = np.empty(n)
    #     for m in range(n):
    #         # SOMETHING WRONG HERE. DOUBLE CHECK DERIVATIONS FIRST
    #         temp[m] = C**m * np.prod(E[n-m:n-1])*(1 - E[n-m-1])
    #     Mzn_pre = M0*C * np.sum(temp)
        
    S = np.sin(fa)*(Mzn_pre - Mzss)
    return S

def run_tof_model(v0, scan_param):
    
    # Scan parameters (should be function input)
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
    X = compute_position(dummyt, x0, v0)
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
        proton_position = compute_position(timings, init_pos, v0)
        
        # Convert absolute positions to slice location
        proton_slice = np.floor(proton_position/w)
        
        # Find pulses that this proton recieved (i.e. in a slice that recieved one)
        match_cond = pulse_slice == proton_slice
        pulse_recieve_ind = np.where(match_cond)[0]
        
        # Loop through recieved pulses and compute flow signal
        tprev = float('nan')
        dt_list = np.empty(np.size(pulse_recieve_ind))
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
    
    # Check conservation of protons
    err_statement = 'Warning: proton conservation failed - check s_counter.'
    assert np.all(s_counter == nproton_per_slice), err_statement
        
    # Divide after summing contributions; signal is the average of protons
    signal = signal / nproton_per_slice    
    return signal


# Velocity input amplitude
v0 = 0.7

# Dictionary containing model parameters
scan_param =	{
    'slice_width' : 0.25,
    'repetition_time' : 0.35,
    'flip_angle' : 47,
    't1_time' : 4,
    'num_slice' : 5,
    'num_pulse' : 100,
    'alpha_list' : [0.14, 0, 0.2075, 0.07, 0.2775]}

signal = run_tof_model(v0, scan_param)


# Plot slice signals
plt.plot(signal)
plt.show()





    