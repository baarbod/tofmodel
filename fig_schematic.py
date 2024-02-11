# -*- coding: utf-8 -*-

import os
import json
import torch

import numpy as np
import matplotlib.pyplot as plt
from config.path import ROOT_DIR

from tof import posfunclib as pfl
from tof import tofmodel as tm

import tofinv.simulations as sim
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


def save_figure(name, fig):
    results_dir = os.path.join(ROOT_DIR, "results", "schematic")
    isExist = os.path.exists(results_dir)
    if not isExist:
        os.makedirs(results_dir)
    figname = f"{name}.svg"
    figpath = os.path.join(results_dir, figname)
    fig.savefig(figpath, bbox_inches="tight", format='svg', dpi=300)
    
config_path = '/om/user/bashen/repositories/tof-inverse/experiments/2024-02-11_09:35:24_training_run_fmv/config_used.json' 
with open(config_path, "r") as jsonfile:
    param = json.load(jsonfile)
state_filename = '/om/user/bashen/repositories/tof-inverse/experiments/2024-02-11_09:35:24_training_run_fmv/model_state_100000_samples.pth'
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
scan_param["num_pulse"] = 200+20

v_offset = 0.05

# plot velocity timeseries
tr = scan_param['repetition_time']
tdummy = tr*np.arange(0, 200)
velocity = np.zeros(np.size(tdummy))
velocity += v_offset

frequencies = np.array([0.05, 0.2, 0.8])
rand_numbers = np.array([0.25, 0.1, 0.2])
rand_phase = np.array([0, 0, 0])

for amp, phase, w in zip(rand_numbers, rand_phase, frequencies):
    T = 1/w
    vsine = amp*np.cos(2*np.pi*w*(tdummy - phase))
    velocity += vsine

# define position-time function based on parameter set
x_func = sim.define_x_func(main_frequencies=frequencies, an_vals=[v_offset, *rand_numbers], phase=rand_phase)

xoffset = 0
width = 0.8
xarea = np.linspace(-3, 3, 200)
N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2)
area = 0.6*N(xarea, xoffset, width)

tdummy = np.linspace(0, 200, 1000)
xdummy = x_func(tdummy, 0)
vdummy = (np.diff(xdummy)/(tdummy[1]-tdummy[0])) 
x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=tdummy[:999], vts=vdummy, xarea=xarea, area=area)

# solve the tof forward model
s_raw = sim.run_model(x_func_area, scan_param)
s_raw = s_raw[20:]

# preprocess raw simulated signal 
s = proc.scale_epi(s_raw)
s_proc = np.array(s)
s -= np.mean(s, axis=0)

# put features in the input array
x = np.zeros((1, 5, 200))
x[0, 0, :] = s[:200, 0].squeeze()
x[0, 1, :] = s[:200, 1].squeeze()
x[0, 2, :] = s[:200, 2].squeeze()
x[0, 3, :] = xarea
x[0, 4, :] = area

# run the tof inverse model using the features as input
x = torch.tensor(x, dtype=torch.float32)
y_predicted = model(x).detach().numpy().squeeze()

trvect = tr * np.arange(np.size(y_predicted))

# define size of figures
cm = 1/2.54/10  # inches to mm
width = 30 # mm
height = 30 # mm
figsize = (cm*width, cm*height)

# plotting
fig_inf, ax = plt.subplots(figsize=figsize)
ax.plot(trvect, s_proc[:, 0], linewidth=0.5, color='slateblue')
ax.plot(trvect, s_proc[:, 1], linewidth=0.5, color='teal')  
ax.plot(trvect, s_proc[:, 2], linewidth=0.5, color='orange')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Inflow signal normalized')

# plotting
fig_vel, ax = plt.subplots(figsize=figsize)
ax.plot(trvect, y_predicted, color='seagreen', linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity (cm/s)')

fig_area, ax = plt.subplots(figsize=figsize)
ax.plot(xarea, area)
ax.set_xlabel('Distance to 1st fMRI slice (cm)')
ax.set_ylabel('Cross-sectional area (cm^2)')

# save results
save_figure(f"TD_inflow_signal", fig_inf)
save_figure(f"TD_predicted_velocity", fig_vel)
save_figure(f"TD_area_depth", fig_area)