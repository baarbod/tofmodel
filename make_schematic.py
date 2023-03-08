# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:46:40 2023

@author: Baarbod
"""

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
from functools import partial 
from tof import posfunclib as pfl
from tof import tofmodel as tm
from tof.fresignal import fre_signal_array as fre_signal

plt.close('all')

def main():
    
    # dictionary containing model parameters
    scan_param =	{
        'slice_width' : 0.25,
        'repetition_time' : 0.387,
        'flip_angle' : 47,
        't1_time' : 4,
        'num_slice' : 5,
        'num_pulse' : 20,
        'MBF' : 1, 
        'alpha_list' : [0.14, 0, 0.2075, 0.07, 0.2775]}
    
    # find the time and target slice of each RF pulse
    timings, pulse_slice = tm.get_pulse_targets(scan_param)
    
    # setup the plot
    fig, ax = make_base_plot()
    
    # draw vertical lines for each RF pulse timing (assuming all slices same)
    trvect = timings
    draw_pulse_lines(ax, timings, pulse_slice, scan_param['slice_width'])
    
    # shade slices with a color
    w = 0.25
    xr = trvect[-1]
    add_slice_shade(ax, w, 1, xr, 'C0')
    add_slice_shade(ax, w, 2, xr, 'C1')
    add_slice_shade(ax, w, 3, xr, 'C3')
    add_slice_shade(ax, w, 4, xr, 'C4')
    add_slice_shade(ax, w, 5, xr, 'C5')
    
    # define position function
    Xfunc = partial(pfl.compute_position_sine, v1=-0.6, v2=0.6, w0=2*np.pi/3)
    
    # compute signal for each recieved RF pulse for each proton
    X0array = np.linspace(-0.2, 2)
    s_proton = run_protons_subroutine(scan_param, Xfunc, X0array)
        
    # draw position-time curves that fade based on signal evolution
    for x0, val_tuple in zip(X0array, s_proton):
        P = Xfunc(trvect, x0)
        draw_fading_curve(ax, trvect, P, val_tuple)
    
    
def draw_fading_curve(ax, x, y, val_tuple):

    xp = np.zeros(len(val_tuple))
    yp = np.zeros(len(val_tuple))

    for count, pair in enumerate(val_tuple):
        idx = pair[0]
        val = pair[1]
        xp[count] = idx
        yp[count] = val
    
    if np.size(xp) != 0:
        alpha_fade = np.interp(x, xp, yp)
    else:
        alpha_fade = np.ones(np.size(x))
        
    for i in range(len(x)):
        alpha = alpha_fade[i]/np.max(alpha_fade)
        if alpha < 0:
            alpha = float(0)
        ax.plot(x[i:i+2], y[i:i+2], 
                color='black', 
                alpha=alpha)   
        
    plt.show()
    
def make_base_plot():
    fig, ax = plt.subplots()
    return fig, ax

def add_slice_shade(ax, w, islice, xr, c):    
    
    y = [(islice-1)*w, islice*w]
    x1 = 0
    x2 = xr
    ax.fill_betweenx(y, x1, x2, step='post', color=c, alpha=0.3)
    
def draw_pulse_lines(ax, timings, pulse_slice, w):

    for timing, slc in zip(timings, pulse_slice):
        x1 = slc*w
        x2 = (slc+1)*w
        line = lines.Line2D([timing, timing], [x1, x2],
                            lw=1, alpha=0.5, color='black', axes=ax)
        ax.add_line(line)
    
def run_protons_subroutine(scan_param, Xfunc, X0array):
    # Scan parameters
    w = scan_param['slice_width']
    TR = scan_param['repetition_time']
    fa = scan_param['flip_angle']*np.pi/180
    T1 = scan_param['t1_time']
    nslice = scan_param['num_slice']
    npulse = scan_param['num_pulse']
    MBF = scan_param['MBF']
    alpha = np.array(scan_param['alpha_list'], ndmin=2).T
    
    assert np.size(alpha) == nslice, 'Warning: size of alpha should be nslice'
    
    nproton = np.size(X0array)
    
    # find the time and target slice of each RF pulse
    timings, pulse_slice = tm.get_pulse_targets(scan_param)
    
    s_proton = []
    
    for iproton in range(nproton):
        
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
        signal_at_time = []
        for count, pulse_id in enumerate(pulse_recieve_ind):
             tnow = timings[pulse_id]
             dt = tnow - tprev # correct dt behavior on 1st pulse
             tprev = tnow
             dt_list[count] = dt
             npulse = count+1 # correcting for zero-based numbering
             S = fre_signal(npulse, fa, TR, T1, dt_list)
             signal_at_time.append((tnow, S))
             
        s_proton.append(signal_at_time)
    return s_proton


if __name__ == "__main__":
   main()

