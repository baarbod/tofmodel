# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import json

import os

from tofinv.models import TOFinverse
import tofinv.utils as utils

import inflowan.plotting as splt
import inflowan.processing as proc
import inflowan.utils as inf_utils
import inflowan.spectrum as spec

from config.path import ROOT_DIR

from inflowan.plotting import set_default_rcParams
set_default_rcParams()
import matplotlib as mpl
mpl.rcParams['font.size'] = 10


def save_figure(name, fig):
    results_dir = os.path.join(ROOT_DIR, "results", "summary")
    isExist = os.path.exists(results_dir)
    if not isExist:
        os.makedirs(results_dir)
    figname = f"{name}.svg"
    figpath = os.path.join(results_dir, figname)
    fig.savefig(figpath, format='svg', dpi=300)


def compute_error(subject, pcruns, fmriruns, param, model_state, mctimes=None):

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
    
    # load fmri and phase contrast signals
    s_raw, v = utils.load_data(subject, fmriruns, pcruns, mctimes=mctimes)
    # nan values from combined array
    s_raw = s_raw[~np.isnan(s_raw).any(axis=1)]
    s_raw = s_raw[20:, :]
    
    # preprocess fmri signal
    s_data = proc.scale_epi(s_raw)
    s_data_for_nn = s_data - np.mean(s_data, axis=0)

    velocity_NN = utils.input_batched_signal_into_NN(s_data_for_nn, model)
    
    f_plot1, xv = spec.compute_frequency_spectra(v - np.mean(v), trpc, method='welch', NW=4)
    f_plot2, xvdummy = spec.compute_frequency_spectra(velocity_NN - np.mean(velocity_NN), trepi, method='welch', NW=2)
    
    # error metrics
    ind = [i for i,value in enumerate(f_plot1) if value < 0.49]
    y_pred_interp = np.interp(f_plot1, f_plot2, xvdummy.squeeze())
    se = (np.abs(y_pred_interp[ind])-xv.squeeze()[ind])**2
    mse = np.mean(se)
    err = np.abs(np.mean(np.abs(y_pred_interp[ind]) - xv.squeeze()[ind]))

    return err

config_path = '/om/user/bashen/repositories/tof-inverse/experiments/2024-01-17_08:53:05_training_run_fmv/config_used.json' 
with open(config_path, "r") as jsonfile:
    param = json.load(jsonfile)
state_filename = '/om/user/bashen/repositories/tof-inverse/experiments/2024-01-17_08:53:05_training_run_fmv/model_state_250000_samples.pth'
model_state = torch.load(state_filename)

# load config dictionary with run information   
config_path = os.path.join(ROOT_DIR, "config", "config_data.json")
with open(config_path, "r") as jsonfile:
        param_data = json.load(jsonfile)
inputs = param_data['human_data']

trepi = 0.504
trpc = 1.00985

cm = 1/2.54/10  # inches to mm
width = 25 # mm
height = 25 # mm
figsize = (cm*width, cm*height)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HUMAN BREATH 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ERRlist_hb = []
for i in range(len(inputs)):
    dset = inputs[i]
    subject = dset['name']
    postype = dset['postype']
    pcruns = dset['pc']
    fmriruns = dset['fmri']
    mctimes = dset['mctimes']
    err = compute_error(subject, pcruns, fmriruns, param, model_state, mctimes=mctimes)
        
    ERRlist_hb.append(err)

fig, ax = splt.plotstrip(ERRlist_hb, figsize=figsize)
ax.set_title('ERR breath')
ax.set_xlim(-0.25, 0.25)
ax.set_yticks([0, 0.25, 0.5]) 
ax.set_ylabel('Error (cm/s)^2/Hz')
save_figure(f"TD_summary_err_hb", fig)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HUMAN REST 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ERRlist_hr = []
for i in range(len(inputs)):
    dset = inputs[i]
    subject = dset['name']
    postype = dset['postype']
    try:
        pcruns = dset['pc_rest']
        fmriruns = dset['fmri_rest']
        mctimes = dset['mctimes_rest']
        err = compute_error(subject, pcruns, fmriruns, param, model_state, mctimes=mctimes)
        ERRlist_hr.append(err)
    except KeyError:
        pass
    
fig, ax = splt.plotstrip(ERRlist_hr, figsize=figsize)
ax.set_title('ERR rest')
ax.set_xlim(-0.25, 0.25)
ax.set_yticks([0, 0.25, 0.5]) 
ax.set_ylabel('Error (cm/s)^2/Hz')
save_figure(f"TD_summary_err_hr", fig)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PHANTOM 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# load config dictionary with run information   
config_path = os.path.join(ROOT_DIR, "config", "config_data.json")
with open(config_path, "r") as jsonfile:
        param_data = json.load(jsonfile)
inputs = param_data['phantom_data']

ERRlist_ph = []
for i in range(len(inputs)):
    dset = inputs[i]
    subject = dset['name']
    pcruns = dset['pc']
    fmriruns = dset['fmri']
    freq = dset['freq']
    vel = dset['velocity']
    err = compute_error(subject, pcruns, fmriruns, param, model_state)

    ERRlist_ph.append(err)

fig, ax = splt.plotstrip(ERRlist_ph, figsize=figsize)
ax.set_title('ERR phantom')
ax.set_xlim(-0.25, 0.25)
ax.set_yticks([0, 0.25, 0.5]) 
ax.set_ylabel('Error (cm/s)^2/Hz')
save_figure(f"TD_summary_err_ph", fig)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print(f"mean phantom: {np.mean(ERRlist_ph)} +- {np.std(ERRlist_ph)}")
print(f"mean human rest: {np.mean(ERRlist_hr)} +- {np.std(ERRlist_hr)}")
print(f"mean human breath: {np.mean(ERRlist_hb)} +- {np.std(ERRlist_hb)}")


