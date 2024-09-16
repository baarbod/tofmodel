# -*- coding: utf-8 -*-

import multiprocessing.shared_memory as msm
import os
import numpy as np
import json
import datetime
import torch
import inflowan.utils as inf_utils
import tofinv.sampling as vd


def create_shared_memory(original_array, name=None):
    # close if exists already
    close_shared_memory(name=name)
    
    # create new shared memory
    shm = msm.SharedMemory(create=True, size=original_array.nbytes, name=name)
    shared_array = np.ndarray(shape=original_array.shape, dtype=original_array.dtype, buffer=shm.buf)
    shared_array[:] = original_array[:]
    return shm


def close_shared_memory(name=None):
    try:
        shm = msm.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except:
        pass


def get_shared_array(name: str, shape=None):
    mp_array = globals()[name]
    np_array = np.frombuffer(mp_array.get_obj(), dtype=np.dtype(mp_array.get_obj()._type_))
    if (shape is None) and (name + '_shape' in globals().keys()):
        shape = globals()[name + '_shape']
        shape = np.frombuffer(shape.get_obj(), dtype=int)
    if shape is not None:
        np_array = np_array.reshape(shape)
    return np_array
    

def get_formatted_day_time():
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d_%H:%M:%S")
    return formatted_datetime


def load_data(subject, fmriruns, pcruns, mctimes=None, subdir='data/measured'):
    # combine pc runs
    if subject == 'ff07':
        v = inf_utils.combine_pc_runs(subject, pcruns, dofilter=False, delimiter=None, subdir=subdir)
    else:
        v = inf_utils.combine_pc_runs(subject, pcruns, dofilter=False, delimiter=',', subdir=subdir)

    # combine epi runs
    if subject == 'fmv04' or subject == 'ff07':
        s_data = inf_utils.combine_epi_runs(subject, fmriruns, startind=40, delimiter=None, remove_offset=True, subdir=subdir)
    else:
        s_data = inf_utils.combine_epi_runs(subject, fmriruns, delimiter=',', startind=40, remove_offset=True, subdir=subdir)

    if mctimes:
        s_raw = inf_utils.remove_motion_events(s_data, mctimes[0], mctimes[1])
    else:
        s_raw = s_data
    return s_raw, v


def input_batched_signal_into_NN_area(s_data_for_nn, NN_model, xarea, area, feature_length=200):
    nwindows = int(s_data_for_nn.shape[0] / feature_length)
    velocity_NN = np.zeros(nwindows * feature_length)
    for window in range(nwindows):
        ind1 = window*feature_length
        ind2 = (window+1)*feature_length
        s_window = s_data_for_nn[ind1:ind2, :]
        
        # put features in the input array
        x = np.zeros((1, 5, feature_length))
        x[0, 0, :] = s_window[:, 0].squeeze()
        x[0, 1, :] = s_window[:, 1].squeeze()
        x[0, 2, :] = s_window[:, 2].squeeze()
        x[0, 3, :] = xarea
        x[0, 4, :] = area
        
        # run the tof inverse model using the features as input
        x = torch.tensor(x, dtype=torch.float32)
        y_predicted = NN_model(x).detach().numpy().squeeze()

        velocity_NN[ind1:ind2] = y_predicted
    return velocity_NN


# load area and corresponding slice positions for a subject
def load_area(subject, slc1, root_dir='/om/user/bashen/repositories/inflow-analysis'):
    filepath = os.path.join(root_dir, 'data', subject, 'area.txt')
    area = np.loadtxt(filepath) # mm^2
    xarea = 1 * np.arange(np.size(area)) # mm
    xarea_new = 0.1 * np.arange(10 * np.size(area)) # mm
    area = np.interp(xarea_new, xarea, area) # mm
    xarea = xarea_new - slc1 # mm
    xarea *= 0.1 # convert the depth to cm
    return xarea, area


def get_slice_from_subject(inputs, subject, slcpos):
    for sub in inputs:
        if sub['name'] == subject:
            if sub['postype'] == slcpos:
                slc = sub['slc']
    return slc


def define_velocity_fourier(t, vcoeff, vfrequencies, vphase, voffset):
    velocity = np.zeros(np.size(t))
    velocity += voffset
    for amp, w, phase in zip(vcoeff, vfrequencies, vphase):
        vsine = amp*np.cos(2*np.pi*w*(t - phase))
        velocity += vsine
    return velocity


def add_baseline_period(t, x, baseline_duration, baseline_value=0):
    dt = t[1] - t[0]
    npoints_baseline = int(np.ceil(baseline_duration/dt))
    x_with_baseline = np.concatenate((baseline_value*np.ones(npoints_baseline), x))
    t_with_baseline = np.linspace(0, t.max() + baseline_duration, np.size(x_with_baseline))
    return t_with_baseline, x_with_baseline


def load_subject_area_matrix(config_data_path, area_size):
    with open(config_data_path, "r") as jsonfile:
            param_data = json.load(jsonfile)
    inputs = param_data['human_data']
    subjects = []
    slice_mode = []
    for subject_info in inputs:
        subjects.append(subject_info['name'])
        slice_mode.append(subject_info['postype'])
    Ay = np.zeros((area_size, len(subjects) + 1)) 
    Ax = np.zeros((area_size, len(subjects) + 1)) 
    for idx, (subject, slc_mode)  in enumerate(zip(subjects, slice_mode)):
        slc = get_slice_from_subject(inputs, subject, slc_mode)
        xarea, area = load_area(subject, slc)
        xarea = vd.downsample(xarea, area_size)
        area = vd.downsample(area, area_size)
        Ax[:, idx] = xarea.flatten()
        Ay[:, idx] = area.flatten()
    # add straight tube case
    subjects.append('straight_tube')
    Ax[:, -1] = Ax[:, -2]
    Ay[:, -1] = np.ones(Ay.shape[0])
    
    return Ax, Ay, subjects
