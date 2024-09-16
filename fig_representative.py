# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import tofinv.utils as utils
import tofinv.simulations as sim
from tofinv.models import TOFinverse

import inflowan.utils as inf_utils
import inflowan.spectrum as spec
import inflowan.processing as proc

from config.path import ROOT_DIR

from functools import partial
from tof import posfunclib as pfl
from tof import tofmodel as tm

from inflowan.plotting import set_default_rcParams
set_default_rcParams()
import matplotlib as mpl
mpl.rcParams['font.size'] = 10


def save_figure(name, fig):
    results_dir = os.path.join(ROOT_DIR, "results", "representative")
    isExist = os.path.exists(results_dir)
    if not isExist:
        os.makedirs(results_dir)
    figname = f"{name}.svg"
    figpath = os.path.join(results_dir, figname)
    fig.savefig(figpath, format='svg', dpi=300)


def routine(subject, pcruns, fmriruns, param, model_state, slc1=0, phantom=False, mctimes=None):

    fstart = param['data_simulation']['frequency_start']
    fend = param['data_simulation']['frequency_end']
    fspacing = param['data_simulation']['frequency_spacing']
    num_input = param['data_simulation']['num_input_features']
    input_size = param['data_simulation']['input_feature_size']
    output_size = param['data_simulation']['output_feature_size']
    model = TOFinverse(nfeatures=num_input,
                    feature_size=input_size,
                    output_size=output_size)
    model.load_state_dict(model_state)

    trepi = 0.504
    trpc = 1.00985

    if phantom:
        xarea = np.linspace(-3, 3, 200)
        area = 0.25*np.ones(np.size(xarea))
    else:
        filepath = os.path.join(ROOT_DIR, 'data', 'measured', subject, 'area.txt')
        area = np.loadtxt(filepath)
        xarea = 1 * np.arange(np.size(area))
        xarea_new = 0.1*np.linspace(0, 10*np.size(area), 200)
        area = np.interp(xarea_new, xarea, area)
        xarea = xarea_new - slc1
        xarea *= 0.1 # convert the depth to cm
    
    # load fmri and phase contrast signals
    s_raw, v = utils.load_data(subject, fmriruns, pcruns, mctimes=mctimes)
    # nan values from combined array
    s_raw = s_raw[~np.isnan(s_raw).any(axis=1)]
    s_raw = s_raw[20:, :]
    
    # preprocess fmri signal
    s_data = proc.scale_epi(s_raw)
    s_data_for_nn = s_data - np.mean(s_data, axis=0)
    
    f_plot, x1_true = spec.compute_frequency_spectra(s_data_for_nn[:, 0], trepi, method='welch', NW=2)
    f_plot, x2_true = spec.compute_frequency_spectra(s_data_for_nn[:, 1], trepi, method='welch', NW=2)
    f_plot, x3_true = spec.compute_frequency_spectra(s_data_for_nn[:, 2], trepi, method='welch', NW=2)
    
    # velocity_NN = utils.input_batched_signal_into_NN(s_data_for_nn, model)
    velocity_NN = utils.input_batched_signal_into_NN_area(s_data_for_nn, model, xarea, area)
    
    f_plot1, xv = spec.compute_frequency_spectra(v - np.mean(v), trpc, method='welch', NW=4)
    f_plot2, xvdummy = spec.compute_frequency_spectra(velocity_NN - np.mean(velocity_NN), trepi, method='welch', NW=2)
    
    param['scan_param']["num_pulse"] = velocity_NN.size

    # define the position function based on area and velocity
    tr_vect = trepi*np.arange(0, np.size(velocity_NN))
    x_func2 = partial(pfl.compute_position_numeric_spatial, tr_vect=tr_vect, vts=velocity_NN, xarea=xarea, area=area)
    s_raw = sim.run_model(x_func2, param['scan_param'], progress=True, showplot=False)
    s_raw = s_raw[20:, :]
    sdata = proc.scale_epi(s_raw)
    s_pred_demean = sdata - np.mean(sdata, axis=0)
    
    f_plot_model, x1_true_model = spec.compute_frequency_spectra(s_pred_demean[:, 0], trepi, method='welch', NW=2)
    f_plot_model, x2_true_model = spec.compute_frequency_spectra(s_pred_demean[:, 1], trepi, method='welch', NW=2)
    f_plot_model, x3_true_model = spec.compute_frequency_spectra(s_pred_demean[:, 2], trepi, method='welch', NW=2)
    
    return x1_true, x2_true, x3_true, f_plot, x1_true_model, x2_true_model, x3_true_model, f_plot_model, f_plot1, xv, f_plot2, xvdummy

config_path = '/om/user/bashen/repositories/tof-inverse/experiments/2024-02-11_09:35:24_training_run_fmv/config_used.json' 
with open(config_path, "r") as jsonfile:
    param = json.load(jsonfile)
state_filename = '/om/user/bashen/repositories/tof-inverse/experiments/2024-02-11_09:35:24_training_run_fmv/model_state_100000_samples.pth'
model_state = torch.load(state_filename)

# load config dictionary with run information   
config_path = os.path.join(ROOT_DIR, "config", "config_data.json")
with open(config_path, "r") as jsonfile:
        param_data = json.load(jsonfile)
inputs = param_data['human_data']

trepi = 0.504
trpc = 1.00985

def change_line_colors(ax):
    c1, c2, c3 = "slateblue", "teal", "orange"
    line_handles = ax.get_lines()
    line_handles[0].set_color(c1)
    line_handles[1].set_color(c2)
    line_handles[2].set_color(c3)

cm = 1/2.54/10  # inches to mm
width = 120 # mm
height = 100 # mm
figsize = (cm*width, cm*height)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize)
plt.subplots_adjust(hspace=0.4)
plt.subplots_adjust(wspace=0.8)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HUMAN BREATH 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
subject_num = 0
dset = inputs[subject_num]
subject = dset['name']
postype = dset['postype']
pcruns = dset['pc']
fmriruns = dset['fmri']
mctimes = dset['mctimes']
slc1 = dset['slc']

x1_true, x2_true, x3_true, f_plot, \
    x1_true_model, x2_true_model, x3_true_model, f_plot_model, \
        f_plot1, xv, f_plot2, xvdummy = routine(subject, pcruns, fmriruns, param, model_state, slc1=slc1, mctimes=mctimes)

ax = axes[2, 0]
X = np.column_stack((x1_true, x2_true, x3_true))
ax.plot(f_plot, X, linewidth=0.5)
ax.set_xticks([0, 0.5, 1]) 
change_line_colors(ax)

ax = axes[2, 2]
X = np.column_stack((x1_true_model, x2_true_model, x3_true_model))
ax.plot(f_plot_model, X, linewidth=0.5)
ax.set_xticks([0, 0.5, 1]) 
change_line_colors(ax)

ax = axes[2, 1]
ax.plot(f_plot1, xv, zorder=0, linewidth=0.5, color='black')
ax.plot(f_plot2, xvdummy, color='seagreen', linewidth=0.5)
ax.set_xticks([0, 0.5, 1]) 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HUMAN REST 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
subject_num = 2
dset = inputs[subject_num]
subject = dset['name']
postype = dset['postype']
pcruns = dset['pc_rest']
fmriruns = dset['fmri_rest']
mctimes = dset['mctimes_rest']
slc1 = dset['slc']

try:
    x1_true, x2_true, x3_true, f_plot, \
        x1_true_model, x2_true_model, x3_true_model, f_plot_model, \
            f_plot1, xv, f_plot2, xvdummy = routine(subject, pcruns, fmriruns, param, model_state, slc1=slc1, mctimes=mctimes)
except KeyError:
    pass

ax = axes[1, 0]
X = np.column_stack((x1_true, x2_true, x3_true))
ax.plot(f_plot, X, linewidth=0.5)
ax.set_xticks([0, 0.5, 1]) 
change_line_colors(ax)

ax = axes[1, 2]
X = np.column_stack((x1_true_model, x2_true_model, x3_true_model))
ax.plot(f_plot_model, X, linewidth=0.5)
ax.set_xticks([0, 0.5, 1]) 
change_line_colors(ax)

ax = axes[1, 1]
ax.plot(f_plot1, xv, zorder=0, linewidth=0.5, color='black')
ax.plot(f_plot2, xvdummy, color='seagreen', linewidth=0.5)
ax.set_xticks([0, 0.5, 1]) 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PHANTOM 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# load config dictionary with run information   
config_path = os.path.join(ROOT_DIR, "config", "config_data.json")
with open(config_path, "r") as jsonfile:
        param_data = json.load(jsonfile)
inputs = param_data['phantom_data']

subject_num = 0
dset = inputs[subject_num]
subject = dset['name']
pcruns = dset['pc']
fmriruns = dset['fmri']
freq = dset['freq']
vel = dset['velocity']

x1_true, x2_true, x3_true, f_plot, \
    x1_true_model, x2_true_model, x3_true_model, f_plot_model, \
        f_plot1, xv, f_plot2, xvdummy = routine(subject, pcruns, fmriruns, param, model_state, phantom=True, mctimes=None)

ax = axes[0, 0]
X = np.column_stack((x1_true, x2_true, x3_true))
ax.plot(f_plot, X, linewidth=0.5)
ax.set_xticks([0, 0.5, 1]) 
change_line_colors(ax)

ax = axes[0, 2]
X = np.column_stack((x1_true_model, x2_true_model, x3_true_model))
ax.plot(f_plot_model, X, linewidth=0.5)
ax.set_xticks([0, 0.5, 1]) 
change_line_colors(ax)

ax = axes[0, 1]
ax.plot(f_plot1, xv, zorder=0, linewidth=0.5, color='black')
ax.plot(f_plot2, xvdummy, color='seagreen', linewidth=0.5)
ax.set_xticks([0, 0.5, 1]) 


# #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

save_figure(f"TD_all_plots", fig)