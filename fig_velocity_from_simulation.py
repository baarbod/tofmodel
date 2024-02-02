# -*- coding: utf-8 -*-

import os
import json
import torch

import numpy as np
import matplotlib.pyplot as plt
from config.path import ROOT_DIR

from tof import posfunclib as pfl
from tof import tofmodel as tm

import tofinv.sampling as vd
import tofinv.utils as utils
from tofinv.models import TOFinverse

import inflowan.utils as inf_utils
import inflowan.spectrum as spec
import inflowan.processing as proc

from functools import partial

from inflowan.plotting import set_default_rcParams
set_default_rcParams()
import matplotlib as mpl
mpl.rcParams['font.size'] = 12


# function for defining the position-time function
def define_x_func(main_frequencies=None, an_vals=None, phase=None):
    an = np.array(an_vals)
    bn = np.zeros(np.size(an) - 1)
    w0 = 2 * np.pi * np.array(main_frequencies)
    x_func = partial(pfl.compute_position_fourier_phase, an=an, bn=bn, w0=w0, phase=phase)
    return x_func

    
# run tof model using the position-time function
def run_model(x_func, scan_param):
    
    # run model 
    signal = tm.run_tof_model(scan_param, x_func,
                              showplot=False,
                              uselookup=False,
                              progress=False)
    s_three_slices = signal[:, 0:3]
    
    return s_three_slices

config_path = '/om/user/bashen/repositories/tof-inverse/experiments/2024-01-17_08:53:05_training_run_fmv/config_used.json' 
with open(config_path, "r") as jsonfile:
    param = json.load(jsonfile)
state_filename = '/om/user/bashen/repositories/tof-inverse/experiments/2024-01-17_08:53:05_training_run_fmv/model_state_250000_samples.pth'
model_state = torch.load(state_filename)

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

# set number of pulses based on desired length of spectrum
scan_param = param['scan_param']
scan_param["num_pulse"] = 1000 + 20

v_offset = 0.05

# plot velocity timeseries
tr = scan_param['repetition_time']
tdummy = tr*np.arange(0, 200)
velocity = np.zeros(np.size(tdummy))
velocity += v_offset

frequencies = np.array([0.02, 0.05, 0.2, 0.8])
rand_numbers = np.array([0.1, 0.3, 0.1, 0.2])
rand_phase = np.array([20, 0, 0, 0])

for amp, phase, w in zip(rand_numbers, rand_phase, frequencies):
    T = 1/w
    vsine = amp*np.cos(2*np.pi*w*(tdummy - phase))
    velocity += vsine

# define position-time function based on parameter set
x_func = define_x_func(main_frequencies=frequencies, an_vals=[v_offset, *rand_numbers], phase=rand_phase)

# solve the tof forward model
s_raw = run_model(x_func, scan_param)
s_raw = s_raw[20:]

# preprocess raw simulated signal 
s = proc.scale_epi(s_raw)
s_proc = np.array(s)
s -= np.mean(s, axis=0)

velocity_NN = utils.input_batched_signal_into_NN(s, model)

trvect = tr * np.arange(np.size(velocity_NN))

fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(trvect, s_proc)
ax2.plot(trvect, velocity_NN)
