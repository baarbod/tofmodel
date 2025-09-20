# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from tofmodel.inverse.models import TOFinverse
import tofmodel.inverse.utils as utils
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm
from functools import partial
from omegaconf import OmegaConf
import argparse
import matplotlib.pyplot as plt
from scipy.signal import welch


def main():
    
    parser = argparse.ArgumentParser(description='run velocity inference using TOF framework')
    parser.add_argument('--signal', type=str, help='path to signal')
    parser.add_argument('--area', type=str, help='path to area')
    parser.add_argument('--model', type=str, help='path to model state')
    parser.add_argument('--outdir', type=str, help='path to output folder')
    parser.add_argument('--config', type=str, help='path to configuration file')
    parser.add_argument('--simulate', action='store_true', help='run simulation using predicted velocity')
    args = parser.parse_args()
    
    print(args.outdir)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    print("Loading data...")
    param = OmegaConf.load(args.config)
    model = load_network(args.model, param)
    sraw, xarea, area = load_data(args.signal, args.area, param)
    s_data_for_nn = scale_data(sraw)
    print("Running velocity inference...")
    velocity_NN = infer_velocity(model, s_data_for_nn, xarea, area, 
                                 input_length=param.data_simulation.input_feature_size, 
                                 output_length=param.data_simulation.output_feature_size)
    print("Solving forward model using velocity...")
    ssim = run_forward_model(velocity_NN, xarea, area, param)
    
    print("Plotting results...")
    trvect = param.scan_param.repetition_time*np.arange(velocity_NN.size)
    
    # time domain plot
    fig_td, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
    ax1.plot(trvect, sraw[:velocity_NN.size])
    ax2.plot(trvect, velocity_NN, color='black')
    ax2.axhline(y=0, color='gray', linestyle='--', zorder=0)
    ax3.plot(trvect, ssim)
    plt.tight_layout()
    plt.show()
    
    tr = param.scan_param.repetition_time
    fs = 1 / tr
    f, psd_velocity = welch(velocity_NN, fs=fs)
    fig_psd, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))
    for i in range(3):
        f, psd_sraw = welch(sraw[:, i], fs=fs)
        ax1.plot(f, psd_sraw, label=f"sraw col {i+1}")
    ax1.set_title("sraw (first 3 columns)")
    ax1.legend()
    ax2.plot(f, psd_velocity, color='black')
    ax2.set_title("velocity_NN")
    for i in range(3):
        f, psd_sim = welch(ssim[:, i], fs=fs)
        ax3.plot(f, psd_sim, label=f"ssim col {i+1}")
    ax3.set_title("ssim (first 3 columns)")
    ax3.legend()
    plt.tight_layout()
    plt.show()
    

    print("Saving results...")
    fig_td.savefig(os.path.join(args.outdir, 'signal_TD_plot.png'), format='png')
    fig_psd.savefig(os.path.join(args.outdir, 'signal_PSD_plot.png'), format='png')
    np.savetxt(os.path.join(args.outdir, 'velocity_predicted.txt'), velocity_NN)
    np.savetxt(os.path.join(args.outdir, 'signal_data.txt'), sraw)
    np.savetxt(os.path.join(args.outdir, 'signal_simulation.txt'), ssim)

    sraw_scaled = scale_data(sraw)
    ssim_scaled = scale_data(ssim)
    signal_error = compute_error(sraw_scaled, ssim_scaled)
    np.savetxt(os.path.join(args.outdir, 'signal_error.txt'), signal_error)
    
    fig_err, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
    ax1.plot(trvect, sraw_scaled[:velocity_NN.size, :])
    ax2.plot(trvect, ssim_scaled)
    ax3.plot(trvect, signal_error)
    plt.tight_layout()
    plt.show()
    fig_err.savefig(os.path.join(args.outdir, 'signal_err_plot.png'), format='png')
    
    
def compute_error(x1, x2):
    ntime1, _ = np.shape(x1)
    ntime2, _ = np.shape(x2)
    ntime = np.min([ntime1, ntime2])
    x1_trimmed = x1[:ntime, :]
    x2_trimmed = x2[:ntime, :]
    err = np.abs(x1_trimmed - x2_trimmed)
    return err


def evaluate_output(eval_data_list):
    errs = []
    for eval_data in eval_data_list:
        sdat, _, ssim = eval_data
        sdat = scale_data(sdat)
        ssim = scale_data(ssim)
        err = compute_error(sdat, ssim)
        errs.append(err.mean(axis=0))
    return errs


def load_data(spath, area_path, param):
    s = np.loadtxt(spath)
    A = np.loadtxt(area_path)
    xarea, area = A[:, 0], A[:, 1]
    new_length = param.data_simulation.input_feature_size
    x_old = np.linspace(0, 1, xarea.size)
    x_new = np.linspace(0, 1, new_length)
    xarea_resampled = np.interp(x_new, x_old, xarea)
    area_resampled = np.interp(x_new, x_old, area)
    return s[20:, :3], xarea_resampled, area_resampled


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


def infer_velocity(model, s_data_for_nn, xarea, area, input_length=300, output_length=300):
    velocity_NN = utils.input_batched_signal_into_NN_area(s_data_for_nn, model, 
                                                      xarea, area,
                                                      input_feature_length=input_length, 
                                                      output_feature_length=output_length)
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
    num_pulse_baseline_offset = 20
    velocity_NN = utils.upsample(velocity_NN, velocity_NN.size*100+1, tr).flatten()
    t = np.arange(0, tr*npulse, tr/100)
    t_with_baseline, v_with_baseline = utils.add_baseline_period(t, velocity_NN, tr*num_pulse_baseline_offset)
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t_with_baseline, 
                        vts=v_with_baseline, xarea=xarea, area=area)
    s_raw = tm.simulate_inflow(tr, te, npulse+num_pulse_baseline_offset, w, fa, t1, t2, nslice, alpha, multi_factor, 
                                x_func_area, multithread=True, enable_logging=True)[:, 0:3]
    s = s_raw[num_pulse_baseline_offset:, :]
    return s


if __name__ == "__main__":
    main()