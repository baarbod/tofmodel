# -*- coding: utf-8 -*-

import multiprocessing.shared_memory as msm
import numpy as np
import itertools
import inflowan.spectrum as spec
import inflowan.processing as proc
import inflowan.utils as inf_utils
from functools import partial
from tof import posfunclib as pfl
from tof import tofmodel as tm

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


def simulate_parameter_set(idx, input_data):
    
    # unpack tuple of inputs
    frequencies, v_offset, rand_phase, velocity_input, scan_param, Xshape, Xtype, Yshape, Ytype, task_id = input_data
    
    # get shared memory arrays 
    Xshm = msm.SharedMemory(name=f"Xarray{task_id}")
    Yshm = msm.SharedMemory(name=f"Yarray{task_id}")
    X = np.ndarray(Xshape, dtype=Xtype, buffer=Xshm.buf)
    Y = np.ndarray(Yshape, dtype=Ytype, buffer=Yshm.buf)
    
    # define position-time function based on parameter set
    x_func = define_x_func(main_frequencies=frequencies, an_vals=[v_offset, *velocity_input], phase=rand_phase)
    
    # solve the tof forward model
    s_raw = run_model(x_func, scan_param)
    
    # skip first 20 points
    s_raw = s_raw[20:, :]
    
    # preprocess raw simulated signal 
    s = proc.scale_epi(s_raw)
    s -= np.mean(s, axis=0)
    
    # # Add Gaussian noise
    # mean = 0
    # std_dev = 0.05  # Adjust the standard deviation to control the amount of noise
    # noise = np.random.normal(mean, std_dev, (200, 3))
    # s += noise
    
    X[idx, 0, :] = s[:, 0].squeeze()
    X[idx, 1, :] = s[:, 1].squeeze()
    X[idx, 2, :] = s[:, 2].squeeze()
    
    tr = scan_param['repetition_time']
    tdummy = tr * np.arange(0, s.shape[0]+20)
    velocity = np.zeros(np.size(tdummy))
    velocity += v_offset

    for amp, phase, w in zip(velocity_input, rand_phase, frequencies):
        vsine = amp*np.cos(2*np.pi*w*(tdummy - phase))
        velocity += vsine

    Y[idx, 0, :] = velocity[20:]
    

    

