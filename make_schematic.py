# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.collections import LineCollection

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
        'repetition_time' : 0.5,
        'flip_angle' : 47,
        't1_time' : 4,
        'num_slice' : 6,
        'num_pulse' : 50,
        'MBF' : 1, 
        'alpha_list' : np.linspace(0, 0.5-0.05, 6)}
        
    # find the time and target slice of each RF pulse
    timings, pulse_slice = tm.get_pulse_targets(scan_param)
    
    # setup the plot
    fig, axes = make_base_plot()
    ax = axes[1]
    plt.subplots_adjust(hspace=0)
    
    # draw vertical lines for each RF pulse timing (assuming all slices same)
    trvect = timings
    draw_pulse_lines(ax, timings, pulse_slice, scan_param['slice_width'])
    
    # shade slices with a color
    w = 0.25
    xr = trvect[-1]
    add_slice_shade(ax, w, 1, xr, "slateblue")
    add_slice_shade(ax, w, 2, xr, "teal")
    add_slice_shade(ax, w, 3, xr, "orange")
    add_slice_shade(ax, w, 4, xr, "dimgrey")
    add_slice_shade(ax, w, 5, xr, "dimgrey")
    add_slice_shade(ax, w, 6, xr, "dimgrey")
    add_slice_shade(ax, w, 7, xr, "dimgrey")
    add_slice_shade(ax, w, 8, xr, "dimgrey")
    
    # define position function
    v1, v2, w0 = -0.5, 0.52, 2*np.pi/6
    Xfunc = partial(pfl.compute_position_sine, v1=v1, v2=v2, w0=w0)
    
    # compute signal for each recieved RF pulse for each proton
    X0array = np.arange(-20, 1.2, 0.1)
    s_proton = run_protons_subroutine(scan_param, Xfunc, X0array)
        
    # draw position-time curves that fade based on signal evolution
    for x0, val_tuple in zip(X0array, s_proton):
        P = Xfunc(trvect, x0)
        draw_fading_curve(ax, trvect, P, val_tuple)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(22) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_xlabel('Time (s)') 
    ax.set_ylabel('Position (cm)') 
    ax.set_xlim(4, 16)
    ax.set_ylim(-0.15, 3*w + 0.05)
    plt.tight_layout(pad=3)
    plt.show()
    
    axes = plot_velocity_and_signal(scan_param, Xfunc, 
                                         v1=v1, v2=v2, w0=w0, 
                                         axes=[axes[2],axes[0]])
    # axes[0].set_ylim(-0.01, 0.6)
    axes[0].set_xlim(4, 16)
    # axes[1].set_ylim(-0.01, 0.6)
    axes[1].set_xlim(4, 16)
    figname = 'results/' + 'detailed_schematic'
    fig.savefig(figname, bbox_inches="tight")
    
    X0array = np.array([-0.2])
    s_proton = run_protons_subroutine(scan_param, Xfunc, X0array)
    tup = s_proton[0]
    pulse_recieve_times = [pair[0] for pair in tup]
    dts = list(np.diff(pulse_recieve_times))
    dt_list = list([np.float('nan')])
    dt_list = np.array(dt_list + dts)
    fig, ax = plot_Mt_curve(scan_param, dt_list)
    ax.set_xlim(0, 12)
    ax.set_box_aspect(1)
    figname = 'results/' + 'single_proton_Mt_curve'
    fig.savefig(figname, bbox_inches="tight")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_ylim(-0.1, 3*w + 0.05)
    ax.set_xlim(0, 10)
    ax.set_box_aspect(1)
    # draw vertical lines for each RF pulse timing (assuming all slices same)
    trvect = timings
    draw_pulse_lines(ax, timings, pulse_slice, scan_param['slice_width'])
    
    # shade slices with a color
    w = 0.25
    xr = trvect[-1]
    add_slice_shade(ax, w, 1, xr, "slateblue")
    add_slice_shade(ax, w, 2, xr, "teal")
    add_slice_shade(ax, w, 3, xr, "orange")
    add_slice_shade(ax, w, 4, xr, "dimgrey")
    add_slice_shade(ax, w, 5, xr, "dimgrey")
    add_slice_shade(ax, w, 6, xr, "dimgrey")
    add_slice_shade(ax, w, 7, xr, "dimgrey")
    add_slice_shade(ax, w, 8, xr, "dimgrey")
    
    # draw position-time curves that fade based on signal evolution
    for x0, val_tuple in zip(X0array, s_proton):
        P = Xfunc(trvect, x0)
        draw_fading_curve(ax, trvect, P, val_tuple)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(22) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_xlabel('Time (s)') 
    ax.set_ylabel('Position (cm)') 
    plt.tight_layout(pad=3)
    plt.show()
    figname = 'results/' + 'single_proton_trajectory'
    fig.savefig(figname, bbox_inches="tight")
    
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
    alpha_fade /= np.max(alpha_fade)   
    alpha_fade[alpha_fade < 0] = 0
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, alpha=alpha_fade,
                        color='black',
                        linewidth=3)
    ax.add_collection(lc)
    
    # add pulse recieve dots
    indices = []
    for i, elem1 in enumerate(xp):
        for j, elem2 in enumerate(x):
            if elem1 == elem2:
                indices.append(j)
    
    for idx in indices:
        ax.scatter(x[idx], y[idx], 
                   color='black',
                   alpha=alpha_fade[idx])
    
    plt.show()
    
def make_base_plot():
    # fig, ax = plt.subplots(figsize=(10, 5))
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 10))
    return fig, axes

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


def plot_Mt_curve(scan_param, dt_list):
    fa = scan_param['flip_angle']*np.pi/180
    TR = scan_param['repetition_time']
    T1 = scan_param['t1_time']
    dt_list = np.array(dt_list)
    n = len(dt_list)

    # test against other formulation 
    S = np.zeros(n)
    for ipulse in range(n):
        S[ipulse] = fre_signal(ipulse+1, fa, TR, T1, dt_list)
          

    fig, ax = plt.subplots(figsize=(8, 8))
        
    # Plot slice signals
    pulsehistory = np.arange(1, np.size(S)+1)
    ax.plot(pulsehistory, S, linewidth=5, color='dimgrey')
        
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(22) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('# Pulse Recieved') 
    ax.set_ylabel('Transverse Magnetization') 
    # ax.set_xlim(10, 20)
    # ax.set_ylim(-0.15, scan_param['num_slice']*w + 0.15)
    plt.tight_layout(pad=3)
    plt.show()
    return fig, ax

def plot_velocity_and_signal(scan_param, Xfunc, v1=None, v2=None, w0=None, 
                             axes=None):
    
    if not axes:
        fig, [axleft, axright] = plt.subplots(nrows=1, ncols=1)
    else:
        fig = None
        axleft, axright = axes[0], axes[1]
    
    signal = tm.run_tof_model(scan_param, Xfunc, showplot=False)
    trvect_sim = scan_param['repetition_time'] * np.arange(scan_param['num_pulse'])
    
    c1, c2, c3 = ["slateblue", "teal", "orange"]

    axleft.plot(trvect_sim, signal[:, 0:3], linewidth=3)
    line_handles = axleft.get_lines()
    line_handles[0].set_color(c1)
    line_handles[1].set_color(c2)
    line_handles[2].set_color(c3)
    # axright = axleft.twinx()
    
    Amp = (v2-v1)/2
    A0 = (v1 + Amp)*2
    An = Amp
    v = A0/2 + An*np.cos(w0*trvect_sim) 
    axright.plot(trvect_sim, v,
                 linewidth=3, color='black')
    axright.axhline(y=0, color='grey', linestyle='dashed')
    
    axes = [axleft, axright]
    ylab = ['Inflow Signal', 'Velocity (cm/s)']
    for ax, ylabel in zip(axes, ylab):
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(22) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_xlabel('Time (s)') 
        ax.set_ylabel(ylabel) 
    plt.tight_layout(pad=4)
    plt.show()
    axes[0].set_xlabel('Time (s)') 
    return axes

if __name__ == "__main__":
   main()

