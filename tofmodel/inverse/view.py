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

    num_samples = 10 * simulation_param.num_samples
    for _ in range(num_samples):
        bound_array = get_sampling_bounds(frequencies, sampling_param.bounding_gaussians, 
                                          sampling_param.lower_fact, sampling_param.upper_fact, sampling_param.global_offset)
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
    
    
def view_simulations(config_path):
    
    param = OmegaConf.load(config_path)
    datasetdir = param.paths.datasetdir
    outputdir = param.paths.outdir
    
    # find the correct output file
    output_file = None
    for file in os.listdir(datasetdir):
        if f"output" in file:
            output_file = os.path.join(datasetdir, file)
            
    with open(output_file, 'rb') as f:
        output = pickle.load(f)
    
    inflow_array = output[0]
    velocity_array = output[1]
    random_sample = np.random.randint(inflow_array.shape[0])
    dt = 0.1
    tr = param.scan_param.repetition_time
    
    s = inflow_array[random_sample, :3, :].T
    v = velocity_array[random_sample, 0, :]
    xarea = inflow_array[random_sample, 3, :]
    area = inflow_array[random_sample, 4, :]
    fig, axes = plt.subplots(nrows=3, ncols=1)
    time_inflow = tr * np.arange(s.shape[0])
    time_velocity = dt * np.arange(v.size)
    axes[0].set_title(f"sample {random_sample}")
    axes[0].plot(time_inflow, s)                      
    axes[1].plot(time_velocity, v, color='black') 
    axes[1].axhline(y=0, color='grey', linestyle='--')
    axes[2].plot(xarea, area)     
    plt.tight_layout()
    
    os.makedirs(outputdir, exist_ok=True)
    plt.savefig(os.path.join(outputdir,  "example_output_simulation.png"))
    plt.close()
    print(f"Saved plot to {outputdir}")