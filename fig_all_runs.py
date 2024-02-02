# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import inflowan.utils as inf_utils
import inflowan.filters as filt
import inflowan.spectrum as spec
import inflowan.processing as proc

import tofinv.utils as utils
from tofinv.models import TOFinverse

from config.path import ROOT_DIR

from inflowan.plotting import set_default_rcParams
set_default_rcParams()
import matplotlib as mpl
mpl.rcParams['font.size'] = 10


def routine(subject, pcruns, fmriruns, param, model_state, mctimes=None):

    fstart = param['data_simulation']['frequency_start']
    fend = param['data_simulation']['frequency_end']
    fspacing = param['data_simulation']['frequency_spacing']
    frequencies = np.arange(fstart, fend, fspacing)
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
    s_raw = s_raw[40:, :]
    
    # preprocess fmri signal
    s_data = proc.scale_epi(s_raw)
    s_data_for_nn = s_data - np.mean(s_data, axis=0)
    
    velocity_NN = utils.input_batched_signal_into_NN(s_data_for_nn, model)
    # velocity_NN = utils.input_batched_signal_into_NN(s_raw, model)

    f_plot_IS1, x1_true = spec.compute_frequency_spectra(s_data_for_nn[:, 0], trepi, method='welch', NW=2)
    f_plot_IS2, x2_true = spec.compute_frequency_spectra(s_data_for_nn[:, 1], trepi, method='welch', NW=2)
    f_plot_IS3, x3_true = spec.compute_frequency_spectra(s_data_for_nn[:, 2], trepi, method='welch', NW=2)
    f_plot1, xvPC = spec.compute_frequency_spectra(v.flatten() - np.mean(v), trpc, method='welch', NW=4)
    f_plot2, xvNN = spec.compute_frequency_spectra(velocity_NN - np.mean(velocity_NN), trepi, method='welch', NW=2)

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    ax1.plot(f_plot_IS1, x1_true, linewidth=0.5)
    ax1.plot(f_plot_IS2, x2_true, linewidth=0.5)
    ax1.plot(f_plot_IS3, x3_true, linewidth=0.5)
    
    ax2.plot(f_plot1, xvPC, linewidth=0.5, color='black')
    ax2.plot(f_plot2, xvNN, linewidth=0.5, color='seagreen')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(s_data)
    return 

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


for i in range(len(inputs)):
    
    dset = inputs[i]
    subject = dset['name']
    try:
        pcruns = dset['pc_rest']
        fmriruns = dset['fmri_rest']
        mctimes = dset['mctimes_rest']
        # pcruns = dset['pc']
        # fmriruns = dset['fmri']
        # mctimes = dset['mctimes']
        print(i)
    except KeyError:
        continue
    routine(subject, pcruns, fmriruns, param, model_state, mctimes=mctimes)
    