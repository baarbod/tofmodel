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
def run_model(x_func, scan_param, progress=False, showplot=False):
    
    # run model 
    signal = tm.run_tof_model(scan_param, x_func,
                              showplot=showplot,
                              uselookup=False,
                              progress=progress)
    s_three_slices = signal[:, 0:3]
    
    return s_three_slices


def simulate_parameter_set(idx, input_data):
    
    # unpack tuple of inputs
    frequencies, v_offset, rand_phase, velocity_input, \
    scan_param, Xshape, Xtype, Yshape, Ytype, task_id, \
    slc_offset, area_gauss_width, area_curve_fact, gauss_noise_std = input_data
    
    tr = scan_param['repetition_time']
    npulse = scan_param["num_pulse"]
    npulse_offset = scan_param["num_pulse_baseline_offset"]

    # get shared memory arrays 
    Xshm = msm.SharedMemory(name=f"Xarray{task_id}")
    Yshm = msm.SharedMemory(name=f"Yarray{task_id}")
    X = np.ndarray(Xshape, dtype=Xtype, buffer=Xshm.buf)
    Y = np.ndarray(Yshape, dtype=Ytype, buffer=Yshm.buf)
    
    xoffset = slc_offset # cm
    width = area_gauss_width # cm
    xarea = np.linspace(-3, 3, Xshape[2]) # cm
    N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2)
    area = area_curve_fact*N(xarea, xoffset, width)
    
    # define velocity based fourier series parameters
    tdummy = tr * np.linspace(0, npulse, 2000)
    vdummy = np.zeros(np.size(tdummy))
    vdummy += v_offset
    for amp, phase, w in zip(velocity_input, rand_phase, frequencies):
        vsine = amp*np.cos(2*np.pi*w*(tdummy - phase))
        vdummy += vsine
    
    # add initial zero-flow period to establish baseline signal before starting flow
    baseline_dt = tr*npulse_offset
    dt = tdummy[1] - tdummy[0]
    npoints_baseline = int(np.ceil(baseline_dt/dt))
    vdummy_with_baseline = np.concatenate((np.zeros(npoints_baseline), vdummy))
    tdummy_with_baseline = tr * np.linspace(0, npulse + npulse_offset, np.size(vdummy_with_baseline))
    
    # define position function
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=tdummy_with_baseline, vts=vdummy_with_baseline, xarea=xarea, area=area)
    
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
    noise = np.random.normal(mean, gauss_noise_std, (200, 3))
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
    X[idx, 3, :] = xarea
    X[idx, 4, :] = area
    
    Y[idx, 0, :] = velocity
    

    

