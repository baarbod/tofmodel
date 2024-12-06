# -*- coding: utf-8 -*-

import multiprocessing.shared_memory as msm
import numpy as np
import datetime
import torch


def create_shared_memory(original_array, name=None):
    """ Create a shared memory array than can be used across multiple CPU

    Parameters
    ----------
    original_array : numpy.ndarray
        input array
        
    name : str, optional
        string to use as the shared arrays name, by default None

    Returns
    -------
    shm: multiprocessing.shared_memory.SharedMemory
        shared memory array
    """
    
    # close if exists already
    close_shared_memory(name=name)
    
    # create new shared memory
    shm = msm.SharedMemory(create=True, size=original_array.nbytes, name=name)
    shared_array = np.ndarray(shape=original_array.shape, dtype=original_array.dtype, buffer=shm.buf)
    shared_array[:] = original_array[:]
    return shm


def close_shared_memory(name=None):
    """ Close a shared memory variable

    Parameters
    ----------
    name : str, optional
        name of shared memory to close, by default None
    """
    
    try:
        shm = msm.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except:
        pass


def get_shared_array(name: str, shape=None):
    """ Get the numpy array from a shared memory array

    Parameters
    ----------
    name : str
        name of shared memory variable
    shape : tuple, optional
        shape of numpy array, by default None

    Returns
    -------
    np_array: numpy.ndarray
        numpy array taken from shared memory array
    """
    
    mp_array = globals()[name]
    np_array = np.frombuffer(mp_array.get_obj(), dtype=np.dtype(mp_array.get_obj()._type_))
    if (shape is None) and (name + '_shape' in globals().keys()):
        shape = globals()[name + '_shape']
        shape = np.frombuffer(shape.get_obj(), dtype=int)
    if shape is not None:
        np_array = np_array.reshape(shape)
    return np_array


def define_velocity_fourier(t, vcoeff, vfrequencies, vphase, voffset):
    """ Define a velocity vector based on fourier series parameters

    Parameters
    ----------
    t : numpy.ndarray
        time points to evaluate fourier series
        
    vcoeff : list
        fourier frequency coeffiecients (cm/s)
        
    vfrequencies : list
        frequencies (Hz)
        
    vphase : list
        time-shifts (s) 
        
    voffset : float
        offset velocity (cm/s)

    Returns
    -------
    velocity: numpy.ndarray
        evaluated velocity vector (cm/s)
    """
    
    velocity = np.zeros(np.size(t))
    velocity += voffset
    for amp, w, phase in zip(vcoeff, vfrequencies, vphase):
        vsine = amp*np.cos(2*np.pi*w*(t - phase))
        velocity += vsine
    return velocity


def add_baseline_period(t, x, baseline_duration, baseline_value=0.0):
    """ Add a period of baseline to the beginning of a timeseries

    Parameters
    ----------
    t : numpy.ndarray
        input time vector
        
    x : numpy.ndarray
        input signal vector
        
    baseline_duration : float
        number of seconds of the baseline period
        
    baseline_value : float, optional
        baseline value, by default 0

    Returns
    -------
    t_with_baseline: 
        time vector with added baseline
    
    x_with_baseline: 
        signal vector with added baseline
        
    """
    
    dt = t[1] - t[0]
    npoints_baseline = int(np.ceil(baseline_duration/dt))
    x_with_baseline = np.concatenate((baseline_value*np.ones(npoints_baseline), x))
    t_with_baseline = np.linspace(0, t.max() + baseline_duration, np.size(x_with_baseline))
    return t_with_baseline, x_with_baseline


def smooth(y, box_pts):
    """ Smooth signal

    Parameters
    ----------
    y : numpy.ndarry
        input signal 
        
    box_pts : int
        size of smoothing box kernal

    Returns
    -------
    y_smooth: numpy.ndarry
        smoothing version input signal with the same dimensions
    """
    
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def upsample(y_input, n, tr):
    """ Upsample a signal using interpolation

    Parameters
    ----------
    y_input : numpy.ndarray
        input signal
        
    n : int
        number of timepoints in upsampled signal

    Returns
    -------
    y_interp: numpy.ndarray
        signal upsampled to the new size
    """
    
    # increase sampling rate using linear interpolation
    if y_input.ndim == 1:
        y_input = np.expand_dims(y_input, 0).T
    npoints, ncols = np.shape(y_input)
    y_interp = np.zeros((n, ncols))
    x = tr * np.arange(npoints)
    for icol in range(ncols):
        y = y_input[:, icol]
        xvals = np.linspace(np.min(x), np.max(x), n)
        y_interp[:, icol] = np.interp(xvals, x, y)
    return y_interp


def downsample(data, new_size):
    """ Downsample a 1d signal using interpolation

    Parameters
    ----------
    data : numpy.ndarray
        input signal
        
    new_size : int
        number of elements in downsampled signal

    Returns
    -------
    downsampled_vector: numpy.ndarray
        signal downsampled to the new size   
    """
    
    old_size = np.size(data)
    new_indices = np.linspace(0, old_size - 1, new_size)
    downsampled_vector = np.interp(new_indices, np.arange(old_size), data)
    return downsampled_vector


def get_formatted_day_time():
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d_%H:%M:%S")
    return formatted_datetime


def input_batched_signal_into_NN_area(s_data_for_nn, NN_model, xarea, area, input_feature_length=200, output_feature_length=1000):
    nwindows = int(s_data_for_nn.shape[0] / input_feature_length)
    if nwindows == 0:
        s_data_for_nn = np.pad(s_data_for_nn, pad_width=((0, input_feature_length-s_data_for_nn.shape[0]), (0, 0)), mode='reflect')
        nwindows = int(s_data_for_nn.shape[0] / input_feature_length)
    velocity_NN = np.zeros(nwindows * output_feature_length)
    for window in range(nwindows):
        ind1 = window*input_feature_length
        ind2 = (window+1)*input_feature_length
        s_window = s_data_for_nn[ind1:ind2, :]
        
        # put features in the input array
        x = np.zeros((1, 5, input_feature_length))
        x[0, 0, :] = s_window[:, 0].squeeze()
        x[0, 1, :] = s_window[:, 1].squeeze()
        x[0, 2, :] = s_window[:, 2].squeeze()
        x[0, 3, :] = xarea
        x[0, 4, :] = area
        
        # run the tof inverse model using the features as input
        x = torch.tensor(x, dtype=torch.float32)
        y_predicted = NN_model(x).detach().numpy().squeeze()

        ind1_out = window*output_feature_length
        ind2_out = (window+1)*output_feature_length
        velocity_NN[ind1_out:ind2_out] = y_predicted
    return velocity_NN


def mean_pct_portion(x, pct, fromtop=False):
    # compute the mean top/bottom percent of signal

    x = np.sort(x, axis=0)
    if fromtop:
        x = np.flipud(x)
    num_rows, _ = x.shape
    n = round(num_rows * pct / 100)
    return np.mean(x[:n, :], axis=0)


def scale_epi(s, startind=0, remove_offset=True):
    # input csf after loading from file
    s = s[startind:, :]
    if remove_offset:
        s -= mean_pct_portion(s, 5)
    slice1_maximum = mean_pct_portion(s[:, [0]], 5, fromtop=True)
    s /= slice1_maximum
    return s