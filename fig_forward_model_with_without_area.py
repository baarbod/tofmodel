# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from functools import partial

import inflowan.spectrum as spec
import inflowan.processing as proc
import inflowan.utils as inf_utils
import inflowan.loading as ld
import inflowan.peaklib as pk
import inflowan.plotting as splt

import tofinv.sampling as vd
import tofinv.simulations as sim

from tof import posfunclib as pfl
from tof import tofmodel as tm
from config.path import ROOT_DIR

config_path = os.path.join(ROOT_DIR, "config", "config.json")
with open(config_path, "r") as jsonfile:
        param = json.load(jsonfile)
 
from inflowan.plotting import set_default_rcParams
set_default_rcParams()
       
scan_param = {
    'slice_width': 0.25,
    'repetition_time': 0.25,
    'flip_angle': 45,
    't1_time': 4,
    'num_slice': 20,
    'num_pulse': 120,
    'MBF': 4,
    'alpha_list': [0, 0.05, 0.1, 0.15, 0.2, 
                   0, 0.05, 0.1, 0.15, 0.2, 
                   0, 0.05, 0.1, 0.15, 0.2, 
                   0, 0.05, 0.1, 0.15, 0.2]}


cm = 1/2.54/10  # inches to mm
width = 50 # mm
height = 50 # mm
figsize = (cm*width, cm*height)

# define velocity
v_offset = 0.05
cycledur = 6
frequencies = np.array([1/cycledur])
rand_numbers = np.array([0.6])
rand_phase = np.array([0])

# NO AREA
# define position-time function based on parameter set
x_func1 = sim.define_x_func(main_frequencies=frequencies, an_vals=[v_offset, *tuple(rand_numbers)], phase=rand_phase)

# solve the tof forward model
s_raw = sim.run_model(x_func1, scan_param)
s_raw = s_raw[20:, :]
sdata = proc.scale_epi(s_raw)

fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
# figav, [ax1av, ax2av, ax3av] = plt.subplots(nrows=1, ncols=3)

tdummy = np.linspace(0, 1.1*scan_param["num_pulse"]*scan_param["repetition_time"], 1000)
xdummy = x_func1(tdummy, 0)
vdummy = (np.diff(xdummy)/(tdummy[1]-tdummy[0]))
ax1.plot(sdata)

flipfact = 1
offset = 3
upsamplefact = 20
trepi = scan_param['repetition_time']
# cycle average model output
signal_model_mean, signal_model_std, _ = pk.run_peak_pipeline(sdata, trepi,
                                                                prominence=0.1,
                                                                norm=True,
                                                                flipfact=flipfact,
                                                                offset=offset,
                                                                showplot=False,
                                                                cycledur=cycledur,
                                                                peakdist=cycledur - 1,
                                                                upsamplefact=upsamplefact)
fig_wo_area, ax = splt.plot_shaded_block_ave(signal_model_mean, signal_model_std, trepi / upsamplefact, linewidth=1.5, figsize=figsize)
ax.set_xlabel('')
ax.set_xlim(0, cycledur)
ax.set_ylim(-0.02, 1.2)
ax.set_xticks([0, cycledur]) 

# WITH REAL NARROW AREA
subject = 'fmv08'
slc1 = 23
filepath = os.path.join(ROOT_DIR, 'data', 'measured', subject, 'area.txt')
area = np.loadtxt(filepath)
xarea = 1 * np.arange(np.size(area))
xarea_new = 0.1 * np.arange(10 * np.size(area))
area = np.interp(xarea_new, xarea, area)
xarea = xarea_new - slc1
xarea *= 0.1

fig, ax_a = plt.subplots()
ax_a.plot(xarea, area)

# define the position function based on area and velocity
x_func2 = partial(pfl.compute_position_numeric_spatial, tr_vect=tdummy[:999], vts=vdummy, xarea=xarea, area=area)

# solve the tof forward model
s_raw = sim.run_model(x_func2, scan_param)
s_raw = s_raw[20:, :]
sdata_narrow = proc.scale_epi(s_raw)
ax2.plot(sdata_narrow)

trepi = scan_param['repetition_time']
# cycle average model output
signal_model_mean, signal_model_std, _ = pk.run_peak_pipeline(sdata_narrow, trepi,
                                                                prominence=0.1,
                                                                norm=True,
                                                                flipfact=flipfact,
                                                                offset=offset,
                                                                showplot=False,
                                                                cycledur=cycledur,
                                                                peakdist=cycledur - 1,
                                                                upsamplefact=upsamplefact)
fig_area_narrow, ax = splt.plot_shaded_block_ave(signal_model_mean, signal_model_std, trepi / upsamplefact, linewidth=1.5, figsize=figsize)
ax.set_xlabel('')
ax.set_xlim(0, cycledur)
ax.set_ylim(-0.02, 1.2)
ax.set_xticks([0, cycledur]) 


# WITH REAL WIDE AREA
subject = 'fmv08'
slc1 = 31
filepath = os.path.join(ROOT_DIR, 'data', 'measured', subject, 'area.txt')
area = np.loadtxt(filepath)
xarea = 1 * np.arange(np.size(area))
xarea_new = 0.1 * np.arange(10 * np.size(area))
area = np.interp(xarea_new, xarea, area)
xarea = xarea_new - slc1
xarea *= 0.1

fig, ax_a = plt.subplots()
ax_a.plot(xarea, area)


# define the position function based on area and velocity
x_func3 = partial(pfl.compute_position_numeric_spatial, tr_vect=tdummy[:999], vts=vdummy, xarea=xarea, area=area)

# solve the tof forward model
s_raw = sim.run_model(x_func3, scan_param)
s_raw = s_raw[20:, :]
sdata_wide = proc.scale_epi(s_raw)
ax3.plot(sdata_wide)

trepi = scan_param['repetition_time']
# cycle average model output
signal_model_mean, signal_model_std, _ = pk.run_peak_pipeline(sdata_wide, trepi,
                                                                prominence=0.1,
                                                                norm=True,
                                                                flipfact=flipfact,
                                                                offset=offset,
                                                                showplot=False,
                                                                cycledur=cycledur,
                                                                peakdist=cycledur - 1,
                                                                upsamplefact=upsamplefact)
fig_area_wide, ax = splt.plot_shaded_block_ave(signal_model_mean, signal_model_std, trepi / upsamplefact, linewidth=1.5, figsize=figsize)
ax.set_xlabel('')
ax.set_xlim(0, cycledur)
ax.set_ylim(-0.02, 1.2)
ax.set_xticks([0, cycledur]) 

def save_figure(name, fig):
    results_dir = os.path.join(ROOT_DIR, "results", "area")
    isExist = os.path.exists(results_dir)
    if not isExist:
        os.makedirs(results_dir)
    figname = f"{name}.svg"
    figpath = os.path.join(results_dir, figname)
    fig.savefig(figpath, format='svg', dpi=300)
    
save_figure('without_area.svg', fig_wo_area)
save_figure('with_area_wide.svg', fig_area_wide)
save_figure('with_area_narrow.svg', fig_area_narrow)