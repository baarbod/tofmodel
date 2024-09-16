# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import inflowan.spectrum as spec
import inflowan.utils as inf_utils
import os
import json

from config.path import ROOT_DIR
config_path = os.path.join(ROOT_DIR, "config", "config.json")
with open(config_path, "r") as jsonfile:
        param = json.load(jsonfile)

sampling_param = param['sampling']
exp_div = sampling_param['exp_div']
breath_fsd = sampling_param['breath_fsd']
cardiac_fsd = sampling_param['cardiac_fsd']
lower_bound = sampling_param['lower_bound']
BU_constant_add = sampling_param['BU_constant_add']
exp_u = sampling_param['exp_u']

def get_sampling_bounds(ff, f1, ff_card=1):
    
    N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2)
    t1 = np.exp(exp_u * -ff)/exp_div
    
    fsd = breath_fsd
    t2 = N(ff, f1, fsd) + N(ff, 2*f1, fsd)/3 + N(ff, 3*f1, fsd)/6
    t2 /= 3.33
    
    ff_cardsd = cardiac_fsd 
    t3 = N(ff, ff_card, ff_cardsd)
    t3 /= 3.33
    
    lower_val = lower_bound
    upper = t1 + t2 + t3 + BU_constant_add
    lower = np.ones(np.shape(ff))*lower_val

    bound_array = np.zeros((np.size(ff), 2))
    bound_array[:, 0] = lower
    bound_array[:, 1] = upper
    return bound_array


def load_velocity(subject, pcruns, subdir='data/measured'):
    if subject == 'ff07':
        v = inf_utils.combine_pc_runs(subject, pcruns, dofilter=False, delimiter=None, subdir=subdir)
    else:
        v = inf_utils.combine_pc_runs(subject, pcruns, dofilter=False, delimiter=',', subdir=subdir)
    return v


def routine(v, trpc):
    f_plot, v_spectrum = spec.compute_frequency_spectra(v - np.mean(v), trpc, method='naive', donorm=False)
    f_plot_up = upsample(f_plot, 1000, f_plot[1] - f_plot[0])
    v_spectrum_up = upsample(v_spectrum, 1000, f_plot[1] - f_plot[0])
    return f_plot_up, v_spectrum_up


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_freq_bounds(f_plot_up, vspectrum_array, scale=1):
    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(7, 4))
    plt.tight_layout(pad=1)
    ax1.plot(f_plot_up, vspectrum_array.T)
    ax1.set_xlabel('Hz')
    ax1.set_ylabel('cm/s')
    
    freq_min = np.min(vspectrum_array, axis=0)
    freq_max = np.max(vspectrum_array, axis=0)

    ax2.plot(f_plot_up, scale*freq_min)
    ax2.plot(f_plot_up, scale*freq_max)
    ax2.set_xlabel('Hz')
    ax2.set_ylabel('cm/s')
    return scale*freq_min, scale*freq_max
    

def upsample(y_input, n, tr):
    # increase sampling rate using linear interpolation
    if y_input.ndim == 1:
        y_input = np.expand_dims(y_input, 0).T
    npoints, ncols = np.shape(y_input)
    y_interp = np.zeros((n, ncols))
    x = tr * np.arange(npoints)
    for icol in range(ncols):
        y = y_input[:, icol]
        xvals = np.linspace(np.min(x), np.max(x), n)
        y_interp[:, icol] = np.interp(xvals, x, y)
    return y_interp


def downsample(data, new_size):
    old_size = np.size(data)
    new_indices = np.linspace(0, old_size - 1, new_size)
    downsampled_vector = np.interp(new_indices, np.arange(old_size), data)
    return downsampled_vector
