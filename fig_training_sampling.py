# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import os

from config.path import ROOT_DIR
from tofinv.sampling import routine, load_velocity, get_sampling_bounds

from inflowan.plotting import set_default_rcParams
set_default_rcParams()

def save_figure(name, fig):
    results_dir = os.path.join(ROOT_DIR, "results", "sampling")
    isExist = os.path.exists(results_dir)
    if not isExist:
        os.makedirs(results_dir)
    figname = f"{name}.svg"
    figpath = os.path.join(results_dir, figname)
    fig.savefig(figpath, bbox_inches="tight", format='svg', dpi=300)

config_path = os.path.join(ROOT_DIR, "config", "config.json")
with open(config_path, "r") as jsonfile:
    param = json.load(jsonfile)

# load config dictionary with run information   
config_path = os.path.join(ROOT_DIR, "config", "config_data.json")
with open(config_path, "r") as jsonfile:
        param_data = json.load(jsonfile)

trepi = 0.504
trpc = 1.00985

# phantom all
vspectrum_array_phantom = np.zeros((20, 1000))
inputs = param_data['phantom_data']
for i in range(len(inputs)):
    subject = inputs[i]['name']
    pcruns = inputs[i]['pc']
    v = load_velocity(subject, pcruns)
    f_plot_up, v_spectrum_up = routine(v, trpc)
    vspectrum_array_phantom[i, :] = v_spectrum_up.squeeze()

# phantom 0.1Hz
vspectrum_array_phantom01 = np.zeros((20, 1000))
inputs = param_data['phantom_data']
for i in range(len(inputs)):
    subject = inputs[i]['name']
    pcruns = inputs[i]['pc']
    freq = inputs[i]['freq']
    if freq == 0.1:
        v = load_velocity(subject, pcruns)
        f_plot_up, v_spectrum_up = routine(v, trpc)
        vspectrum_array_phantom01[i, :] = v_spectrum_up.squeeze()
    else:
        continue
    
# phantom 0.15Hz
vspectrum_array_phantom015 = np.zeros((20, 1000))
inputs = param_data['phantom_data']
for i in range(len(inputs)):
    subject = inputs[i]['name']
    pcruns = inputs[i]['pc']
    freq = inputs[i]['freq']
    if freq == 0.15:
        v = load_velocity(subject, pcruns)
        f_plot_up, v_spectrum_up = routine(v, trpc)
        vspectrum_array_phantom015[i, :] = v_spectrum_up.squeeze()
    else:
        continue
    
# phantom 0.2Hz
vspectrum_array_phantom02 = np.zeros((20, 1000))
inputs = param_data['phantom_data']
for i in range(len(inputs)):
    subject = inputs[i]['name']
    pcruns = inputs[i]['pc']
    freq = inputs[i]['freq']
    if freq == 0.2:
        v = load_velocity(subject, pcruns)
        f_plot_up, v_spectrum_up = routine(v, trpc)
        vspectrum_array_phantom02[i, :] = v_spectrum_up.squeeze()
    else:
        continue

# human breath 
vspectrum_array_human_breath = np.zeros((20, 1000))
inputs = param_data['human_data']
for i in range(len(inputs)):
    subject = inputs[i]['name']
    pcruns = inputs[i]['pc']
    v = load_velocity(subject, pcruns)
    f_plot_up, v_spectrum_up = routine(v, trpc)
    vspectrum_array_human_breath[i, :] = v_spectrum_up.squeeze()

# human rest 
vspectrum_array_human_rest = np.zeros((20, 1000))
inputs = param_data['human_data']
for i in range(len(inputs)):
    subject = inputs[i]['name']
    try:
        pcruns = inputs[i]['pc_rest']
    except KeyError:
        continue
    v = load_velocity(subject, pcruns)
    f_plot_up, v_spectrum_up = routine(v, trpc)
    vspectrum_array_human_rest[i, :] = v_spectrum_up.squeeze()

vspectrum_array_phantom = vspectrum_array_phantom[~np.all(vspectrum_array_phantom == 0, axis=1)]
vspectrum_array_phantom01 = vspectrum_array_phantom01[~np.all(vspectrum_array_phantom01 == 0, axis=1)]
vspectrum_array_phantom015 = vspectrum_array_phantom015[~np.all(vspectrum_array_phantom015 == 0, axis=1)]
vspectrum_array_phantom02 = vspectrum_array_phantom02[~np.all(vspectrum_array_phantom02 == 0, axis=1)]
vspectrum_array_human_breath = vspectrum_array_human_breath[~np.all(vspectrum_array_human_breath == 0, axis=1)]
vspectrum_array_human_rest = vspectrum_array_human_rest[~np.all(vspectrum_array_human_rest == 0, axis=1)]
vspectrum_array_all = np.vstack((vspectrum_array_phantom, vspectrum_array_human_breath, vspectrum_array_human_rest))

cm = 1/2.54/10  # inches to mm
width = 50 # mm
height = 30 # mm
figsize = (cm*width, cm*height)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(f_plot_up, vspectrum_array_phantom01.T, linewidth=0.5)
ax.plot(f_plot_up, vspectrum_array_phantom015.T, linewidth=0.5)
ax.plot(f_plot_up, vspectrum_array_phantom02.T, linewidth=0.5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (cm/s)')
save_figure(f"example_phantom_pc_spectra", fig)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(f_plot_up, vspectrum_array_human_breath.T, linewidth=0.5)
ax.plot(f_plot_up, np.mean(vspectrum_array_human_breath.T, axis=1), color='black', linewidth=1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (cm/s)')
save_figure(f"example_human_breath_pc_spectra", fig)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(f_plot_up, vspectrum_array_human_rest.T, linewidth=0.5)
ax.plot(f_plot_up, np.mean(vspectrum_array_human_rest.T, axis=1), color='black', linewidth=1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (cm/s)')
save_figure(f"example_human_rest_pc_spectra", fig)


# demo sampling

cm = 1/2.54/10  # inches to mm
width = 150 # mm
height = 40 # mm
figsize = (cm*width, cm*height)

fstart, fend, fspacing = 0.01, 1.5, 0.01
frequencies = np.arange(fstart, fend, fspacing)
fig, ax = plt.subplots(figsize=figsize)
for isample in range(1):
    f1 = 0.12
    ff_card = 0.9
    bound_array = get_sampling_bounds(frequencies, f1, ff_card=ff_card)
    # Plot the two curves
    ax.fill_between(frequencies, bound_array[:, 0], bound_array[:, 1], color='gray', alpha=0.2)
    rand_numbers = np.zeros(np.size(frequencies))
    for idx, frequency in enumerate(frequencies):
        lower = bound_array[idx, 0]
        upper = bound_array[idx, 1] 
        rand_numbers[idx] = np.random.uniform(low=lower, high=upper)
    ax.plot(frequencies, rand_numbers, label=f"sample{isample}", color='black')

ax.plot(frequencies, bound_array[:, 0], color='red')
ax.plot(frequencies, bound_array[:, 1], color='blue')

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (cm/s)')
save_figure(f"example_samples", fig)