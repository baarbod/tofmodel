# -*- coding: utf-8 -*-

import multiprocessing.shared_memory as msm
import numpy as np
import inflowan.processing as proc
import tofinv.utils as utils
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
def run_model(x_func, scan_param, progress=False, showplot=False):
    
    # run model 
    signal = tm.run_tof_model(scan_param, x_func,
                              showplot=showplot,
                              progress=progress)
    s_three_slices = signal[:, 0:3]
    
    return s_three_slices


def simulate_parameter_set(idx, input_data):
    
    # unpack tuple of inputs
    frequencies, v_offset, rand_phase, velocity_input, \
    scan_param, Xshape, Xtype, Yshape, Ytype, task_id, \
    xarea_sample, area_sample, gauss_noise_std = input_data
    
    tr = scan_param['repetition_time']
    npulse = scan_param["num_pulse"]
    npulse_offset = scan_param["num_pulse_baseline_offset"]

    # get shared memory arrays 
    Xshm = msm.SharedMemory(name=f"Xarray{task_id}")
    Yshm = msm.SharedMemory(name=f"Yarray{task_id}")
    X = np.ndarray(Xshape, dtype=Xtype, buffer=Xshm.buf)
    Y = np.ndarray(Yshape, dtype=Ytype, buffer=Yshm.buf)
    
    # define velocity and add initial zero-flow baseline period 
    tdummy = tr * np.linspace(0, npulse, Yshape[2])
    vdummy = utils.define_velocity_fourier(tdummy, velocity_input, frequencies, rand_phase, v_offset)
    baseline_duration = tr*npulse_offset
    tdummy_with_baseline, vdummy_with_baseline = utils.add_baseline_period(tdummy, vdummy, baseline_duration)
    
    # define position function
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=tdummy_with_baseline, 
                          vts=vdummy_with_baseline, xarea=xarea_sample, area=area_sample)
    
    # solve the tof forward model including the extra offset pulses
    scan_param['num_pulse'] = npulse + npulse_offset
    s_raw = run_model(x_func_area, scan_param)
    
    # remove inital baseline period from signal
    s_raw = s_raw[npulse_offset:, :]
    
    # preprocess raw simulated signal 
    s = proc.scale_epi(s_raw)
    s -= np.mean(s, axis=0)
    
    # Add zero-mean gaussian noise
    mean = 0
    noise = np.random.normal(mean, gauss_noise_std, (Xshape[2], 3))
    s += noise
    
    # define ground truth velocity, same length as inflow signal
    tdummy = tr * np.arange(0, s.shape[0])
    velocity = np.zeros(np.size(tdummy))
    velocity += v_offset
    for amp, phase, w in zip(velocity_input, rand_phase, frequencies):
        vsine = amp*np.cos(2*np.pi*w*(tdummy - phase))
        velocity += vsine

    # fill matricies
    X[idx, 0, :] = s[:, 0].squeeze()
    X[idx, 1, :] = s[:, 1].squeeze()
    X[idx, 2, :] = s[:, 2].squeeze()
    X[idx, 3, :] = xarea_sample
    X[idx, 4, :] = area_sample
    
    Y[idx, 0, :] = velocity
    

    

