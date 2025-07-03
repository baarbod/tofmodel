# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch
from tofmodel.inverse.models import TOFinverse
import tofmodel.inverse.utils as utils
import tofmodel.inverse.utils as utils
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm
from functools import partial



def run_inference(data_tuple_list, param, state_path, runsim=True):
    eval_data = []
    model = load_network(state_path, param)
    for sdat, xarea, area in data_tuple_list:
        s_data_for_nn = scale_data(sdat)
        velocity_NN = infer_velocity(model, s_data_for_nn, xarea, area)
        if runsim:
            ssim = run_forward_model(velocity_NN, xarea, area, param)
            eval_data.append((sdat, velocity_NN, ssim))
        else:
            eval_data.append((sdat, velocity_NN))
    return eval_data


def compute_error(x1, x2):
    ntime1, _ = np.shape(x1)
    ntime2, _ = np.shape(x2)
    ntime = np.min([ntime1, ntime2])
    x1_trimmed = x1[:ntime, :]
    x2_trimmed = x2[:ntime, :]
    err = np.abs(x1_trimmed - x2_trimmed)
    return err


def evaluate_output(eval_data):
    errs = []
    return errs


def load_data(spath, area_path, param):
    s = np.loadtxt(spath)
    A = np.loadtxt(area_path, skiprows=1, delimiter=',')
    xarea, area = A[:, 0], A[:, 1]
    new_length = param.data_simulation.input_feature_size
    x_old = np.linspace(0, 1, xarea.size)
    x_new = np.linspace(0, 1, new_length)
    xarea_resampled = np.interp(x_new, x_old, xarea)
    area_resampled = np.interp(x_new, x_old, area)
    return s, xarea_resampled, area_resampled


def scale_data(s):
    mean_ref = np.mean(s[:, 0])
    std_ref = np.std(s[:, 0])
    s_data_for_nn = np.empty_like(s)
    s_data_for_nn[:, 0] = (s[:, 0] - mean_ref) / std_ref
    s_data_for_nn[:, 1:] = (s[:, 1:] - mean_ref) / std_ref
    return s_data_for_nn
    
    
def load_network(state_filename, param):
    model_state = torch.load(state_filename, map_location=torch.device('cpu'))
    num_input = param.data_simulation.num_input_features
    input_size = param.data_simulation.input_feature_size
    output_size = param.data_simulation.output_feature_size
    model = TOFinverse(nfeature_in=num_input, nfeature_out=1, 
                    input_size=input_size,
                    output_size=output_size)
    model.load_state_dict(model_state)
    return model


def infer_velocity(model, s_data_for_nn, xarea, area):
    input_feature_length = 200
    output_feature_length = 200
    velocity_NN = utils.input_batched_signal_into_NN_area(s_data_for_nn, model, 
                                                      xarea, area,
                                                      input_feature_length=input_feature_length, 
                                                      output_feature_length=output_feature_length)
    return velocity_NN


def run_forward_model(velocity_NN, xarea, area, param):
    tr = param.scan_param.repetition_time
    te = param.scan_param.echo_time
    w = param.scan_param.slice_width
    fa = param.scan_param.flip_angle
    t1 = param.scan_param.t1_time
    t2 = param.scan_param.t2_time
    nslice = param.scan_param.num_slice
    npulse = velocity_NN.size
    multi_factor = param.scan_param.MBF
    alpha = param.scan_param.alpha_list
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=tr*np.arange(velocity_NN.size), 
                        vts=velocity_NN, xarea=xarea, area=area)
    s_raw = tm.simulate_inflow(tr, te, npulse, w, fa, t1, t2, nslice, alpha, multi_factor, 
                                x_func_area, multithread=True, enable_logging=True)[:, 0:3]
    return s_raw

