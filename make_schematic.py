# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:46:40 2023

@author: Baarbod
"""

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
from functools import partial 
import posfunclib as pfl
from tofmodel import get_pulse_targets
from tofmodel import set_init_positions
from fre_signal import fre_signal_array as fre_signal

plt.close('all')

def draw_fading_curve(ax, x, y, val_tuple):

    # x = np.linspace(0, 10, 100)
    # y = np.sin(x)

    # val_tuple = [(2, 0.8), (3, 0.35), (7, 0.3), (7.5, 0)]

    xp = np.zeros(len(val_tuple))
    yp = np.zeros(len(val_tuple))

    for count, pair in enumerate(val_tuple):
        idx = pair[0]
        val = pair[1]
        xp[count] = idx
        yp[count] = val
    
    alpha_fade = np.interp(x, xp, yp)
        
    for i in range(len(x)):
        alpha = alpha_fade[i]/np.max(alpha_fade)
        if alpha < 0:
            alpha = np.float(0)
        ax.plot(x[i:i+2], y[i:i+2], color='black', alpha=alpha)
        
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_title('Fading Line')
        
    plt.show()


def main():
    # Dictionary containing model parameters
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
    timings, pulse_slice = get_pulse_targets(scan_param)
    
    #trvect = scan_param['repetition_time'] * np.arange(scan_param['num_pulse'])
    #trvect = np.arange(scan_param['num_pulse'])
    trvect = timings
    fig, ax = make_base_plot()
    draw_pulse_lines(ax, trvect)
    
    w = 0.25
    xr = trvect[-1]
    add_slice_shade(ax, w, 1, xr, 'C0')
    add_slice_shade(ax, w, 2, xr, 'C1')
    add_slice_shade(ax, w, 3, xr, 'C3')
    
    
    Xfunc = partial(pfl.compute_position_sine, v1=-0.5, v2=0.5, w0=2*np.pi/5)
    #X0array = np.array([0, 0.25, 0.5])
    X0array = np.linspace(0, 2)
    s_proton = run_protons_subroutine(scan_param, Xfunc, X0array)
    #draw_proton_trajectories(ax, trvect, Xfunc, X0array)
    
    for x0, val_tuple in zip(X0array, s_proton):
        P = Xfunc(trvect, x0)
        draw_fading_curve(ax, trvect, P, val_tuple)
    
    

    
def make_base_plot():
    fig, ax = plt.subplots()
    return fig, ax

def add_slice_shade(ax, w, islice, xr, c):    
    
    y = [(islice-1)*w, islice*w]
    x1 = 0
    x2 = xr
    ax.fill_betweenx(y, x1, x2, step='post', color=c, alpha=0.3)
    
def draw_pulse_lines(ax, trlist):
    for tr in trlist:
        line = lines.Line2D([tr, tr], [0, 0.75],
                            lw=1, alpha=0.5, color='black', axes=ax)
        ax.add_line(line)

def draw_proton_trajectories(ax, t, Xfunc, init_positions):
    for x0 in init_positions:
        x = Xfunc(t, x0)
        ax.plot(t, x)
    return x
    
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
    
    # dx = 0.01
    # X0array = set_init_positions(Xfunc, TR, w, npulse, nslice, dx)
    nproton = np.size(X0array)
    
    # find the time and target slice of each RF pulse
    timings, pulse_slice = get_pulse_targets(scan_param)
    
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

