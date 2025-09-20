# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tofmodel.inverse.dataset import get_sampling_bounds
import tofmodel.inverse.utils as utils
from scipy.signal import welch
import pickle


def compute_frequency_spectra(s, tr, NW=1):
    if s.ndim == 1:
        s = np.expand_dims(s, axis=1)
    f_plot, s_mag_plot = welch(s.T, 1/tr, nperseg=int(s.shape[0]/NW))
    s_mag_plot = s_mag_plot.T
    return f_plot, s_mag_plot


def view_sampling(config_path):
    param = OmegaConf.load(config_path)
    outputdir = param.paths.outdir
    
    sampling_param = param.sampling
    simulation_param = param.data_simulation

    frequencies = np.arange(simulation_param.frequency_start,
                            simulation_param.frequency_end,
                            simulation_param.frequency_spacing)

    plt.figure(figsize=(10, 6))
    plt.title("Preview of Random Samples from Frequency Bounds", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Velocity power", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)

    if sampling_param.mode == 'data':
        with open(param.paths.path_to_velocity_kde, 'rb') as f:
            kdes = pickle.load(f)
    elif sampling_param.mode == 'uniform':
        kdes = None
    

    num_samples = 1000
    for _ in range(num_samples):
        bound_array = get_sampling_bounds(frequencies, sampling_param.bounding_gaussians, 
                                          sampling_param.lower_fact, sampling_param.upper_fact, sampling_param.global_offset, kdes=kdes)
        rand_values = np.random.uniform(low=bound_array[:, 0], high=bound_array[:, 1])
        rand_phase = np.random.uniform(low=0, high=1/frequencies)
        v_offset = np.random.uniform(low=sampling_param.voffset_lower,
                                     high=sampling_param.voffset_upper)

        dt = 0.1
        t = np.arange(0, 100, dt)
        v = utils.define_velocity_fourier(t, rand_values, frequencies, rand_phase, v_offset)

        f_plot_v, x_mag_plot_v = compute_frequency_spectra(v, dt)
        plt.plot(f_plot_v, x_mag_plot_v, color='steelblue', alpha=0.3, linewidth=1)

    plt.xlim(frequencies[0], frequencies[-1])
    plt.tight_layout()

    os.makedirs(outputdir, exist_ok=True)
    plt.savefig(os.path.join(outputdir,  "preview_sampling_plot.png"))
    plt.close()
    print(f"Saved plot to {outputdir}")
    

from functools import partial
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm
import tofmodel.inverse.dataset as dataset


def view_simulations(config_path):
    param = OmegaConf.load(config_path)
    outputdir = param.paths.outdir
    dt = 0.1
    tr = param.scan_param.repetition_time

    def run_simulation(input_data):
        p = input_data['param'].scan_param
        t = np.arange(0, p.num_pulse*p.repetition_time, 0.1)
        v = utils.define_velocity_fourier(t, input_data['velocity_input'], 
                                        input_data['frequencies'], input_data['rand_phase'], 
                                        input_data['v_offset'])
        t_with_baseline, v_with_baseline = utils.add_baseline_period(t, v, p.repetition_time*p.num_pulse_baseline_offset)
        x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t_with_baseline, 
                            vts=v_with_baseline, xarea=input_data['xarea_sample'], area=input_data['area_sample'])
        s_raw = tm.simulate_inflow(p.repetition_time, p.echo_time, p.num_pulse+p.num_pulse_baseline_offset, 
                                p.slice_width, p.flip_angle, p.t1_time, p.t2_time, p.num_slice, p.alpha_list, 
                                p.MBF, x_func_area, multithread=False)[:, 0:3]
        return v, s_raw[p.num_pulse_baseline_offset:, :]


    param = OmegaConf.load(config_path)
    num_sample = 1
    task_id = 1
    input_data = dataset.define_input_params(num_sample, param, task_id)[0]
    v, s = run_simulation(input_data)

    xarea = input_data['xarea_sample']
    area = input_data['area_sample']
    
    fig, axes = plt.subplots(nrows=3, ncols=1)
    time_inflow = np.arange(0, param.scan_param.num_pulse*tr, tr)
    time_velocity = np.arange(0, param.scan_param.num_pulse*tr, dt)
    axes[0].plot(time_inflow, s)                      
    axes[1].plot(time_velocity, v, color='black') 
    axes[1].axhline(y=0, color='grey', linestyle='--')
    axes[2].plot(xarea, area)     
    plt.tight_layout()
    
    os.makedirs(outputdir, exist_ok=True)
    plt.savefig(os.path.join(outputdir,  "example_output_simulation.png"))
    plt.close()
    print(f"Saved plot to {outputdir}")
    
