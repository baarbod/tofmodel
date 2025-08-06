# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.collections import LineCollection
import numpy as np
from tofmodel.forward import simulate as tm
from tofmodel.forward.fresignal import fre_signal_array as fre_signal


def plot_forward_sim_visual(w, tr, te, npulse, fa, nslice, alpha, t1, t2, multi_factor, x_func, xlim, ylim, axis=False, figsize=(2, 2)):
    
    if not axis:
        fig, ax = plt.subplots( figsize=figsize)
    else:
        fig = None
        ax = axis
    
    # find the time and target slice of each RF pulse
    timings_with_repeats, pulse_slice = tm.get_pulse_targets(tr, nslice, npulse, np.array(alpha, ndmin=2).T)
    
    # match pulses to their RF cycles
    pulse_tr_actual = tm.match_pulse_to_tr(npulse, nslice)
    
    # draw vertical lines for each RF pulse timing (assuming all slices same)
    trvect = timings_with_repeats
    draw_pulse_lines(ax, timings_with_repeats, pulse_slice, w)
    
    # shade slices with a color
    xr = trvect[-1]
    add_slice_shade(ax, w, 1, xr, "slateblue")
    add_slice_shade(ax, w, 2, xr, "teal")
    add_slice_shade(ax, w, 3, xr, "orange")
    add_slice_shade(ax, w, 4, xr, "dimgrey")
    add_slice_shade(ax, w, 5, xr, "dimgrey")
    add_slice_shade(ax, w, 6, xr, "dimgrey")
    add_slice_shade(ax, w, 7, xr, "dimgrey")
    add_slice_shade(ax, w, 8, xr, "dimgrey")
    
    y_tick_spacing = w
    y_tick_locator = plt.MultipleLocator(base=y_tick_spacing)
    ax.yaxis.set_major_locator(y_tick_locator)        

    dx = 0.1
    lower_bound = -20
    upper_bound = 1
    X_for_plot = tm.compute_position(x_func, timings_with_repeats, lower_bound, upper_bound, dx)
    
    nproton_for_plot = X_for_plot.shape[0]
    params = [(npulse, nslice, X_for_plot[iproton, :], multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual)
                for iproton in range(nproton_for_plot)]
    s_proton = []
    for iproton in range(nproton_for_plot):
        s_proton.append(compute_proton_signal_contribution(iproton, params[iproton]))
        
    # draw position-time curves that fade based on signal evolution
    for iproton, val_tuple in enumerate(s_proton):
        P = np.repeat(X_for_plot[iproton, :].flatten(), multi_factor)
        
        draw_fading_curve(ax, trvect, P, val_tuple, s=15, lw=0.35, marker='.')

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    return fig, ax

def draw_fading_curve(ax, tproton, xproton, val_tuple, lw=2, s=None, marker=None):

    if len(val_tuple) == 0:
        return False
    
    # add 1 to leave room for dummy event 
    pulse_event_times = np.zeros(len(val_tuple) + 1)
    pulse_event_signals = np.zeros(len(val_tuple) + 1)

    for count, pair in enumerate(val_tuple):
        time_at_pulse = pair[0]
        signal_at_pulse = pair[1]
        pulse_event_times[count] = time_at_pulse
        pulse_event_signals[count] = signal_at_pulse
    
    # add dummy pulse event to ramp signal to maximum after final pulse
    # this is to illustrate relaxation process
    T1 = 4
    time_to_max = 3*T1
    time_max_recovery = pulse_event_times[-2] + time_to_max
    pulse_event_times[-1] = time_max_recovery
    pulse_event_signals[-1] = pulse_event_signals.max()
    
    if np.size(pulse_event_times) != 0:
        alpha_fade = np.interp(tproton, pulse_event_times, pulse_event_signals)
    else:
        alpha_fade = np.ones(np.size(tproton))
    alpha_fade /= np.max(alpha_fade)   
    alpha_fade[alpha_fade < 0] = 0
    
    # CLAMP
    alpha_fade[alpha_fade < 0.15] = 0.15
    
    points = np.array([tproton, xproton]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, alpha=alpha_fade,
                        color='black',
                        linewidth=lw)
    ax.add_collection(lc)
    
    # add pulse recieve dots, skip last dummy event
    indices = []
    for i, elem1 in enumerate(pulse_event_times[:-1]):
        for j, elem2 in enumerate(tproton):
            if elem1 == elem2:
                indices.append(j)
    
    for idx in indices:
        ax.scatter(tproton[idx], xproton[idx], 
                   color='black',
                   alpha=alpha_fade[idx],
                   edgecolors='none',
                   s=s,
                   zorder=10,
                   marker=marker)
    
    # plt.show()


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
                            lw=0.5, alpha=0.5, color='black', axes=ax)
        ax.add_line(line)
    

def compute_proton_signal_contribution(iproton, params):
    
    npulse, nslice, Xproton, multi_factor, timings_with_repeats, w, fa, tr, te, t1, t2, pulse_slice, pulse_tr_actual = params
    
    # convert absolute positions to slice location
    proton_slice = np.floor(np.repeat(Xproton, multi_factor) / w)
    pulse_recieve_ind = np.where(proton_slice == pulse_slice)[0]

    tprev = float('nan')
    dt_list = np.zeros(np.size(pulse_recieve_ind))
    signal_at_time = []
    for count, pulse_id in enumerate(pulse_recieve_ind):
        dt_list[count] = timings_with_repeats[pulse_id] - tprev
        tprev = timings_with_repeats[pulse_id]

        npulse = count + 1  # correcting for zero-based numbering
        s = fre_signal(npulse, fa, tr, te, t1, t2, dt_list)
        signal_at_time.append((timings_with_repeats[pulse_id], s))
        
    return signal_at_time












