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
import json
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
    velocity_NN = infer_velocity(model, s_data_for_nn, xarea, area)
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
    for i in range(param.nslice_to_use):
        f, psd_sraw = welch(sraw[:, i], fs=fs)
        ax1.plot(f, psd_sraw, label=f"sraw col {i+1}")
    ax1.set_title(f"sraw (first {param.nslice_to_use} columns)")
    ax1.legend()
    ax2.plot(f, psd_velocity, color='black')
    ax2.set_title("velocity_NN")
    for i in range(param.nslice_to_use):
        f, psd_sim = welch(ssim[:, i], fs=fs)
        ax3.plot(f, psd_sim, label=f"ssim col {i+1}")
    ax3.set_title(f"ssim (first {param.nslice_to_use} columns)")
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

    # compute metrics (uses tr to convert lag to seconds)
    tr = param.scan_param.repetition_time
    metrics = compute_metrics(param, sraw_scaled, ssim_scaled, tr=tr, nperseg=256)

    # save metrics
    save_metrics_json(metrics, os.path.join(args.outdir, "metrics.json"))

    # also save the raw per-timepoint absolute error for compatibility
    signal_error = np.abs(sraw_scaled[:min(sraw_scaled.shape[0], ssim_scaled.shape[0]), :]
                         - ssim_scaled[:min(sraw_scaled.shape[0], ssim_scaled.shape[0]), :])
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
    num_pulse_baseline_offset = param.scan_param.num_pulse_baseline_offset
    s = np.loadtxt(spath)
    A = np.loadtxt(area_path)
    xarea, area = A[:, 0], A[:, 1]
    new_length = param.data_simulation.input_feature_size
    x_old = np.linspace(0, 1, xarea.size)
    x_new = np.linspace(0, 1, new_length)
    xarea_resampled = np.interp(x_new, x_old, xarea)
    area_resampled = np.interp(x_new, x_old, area)
    return s[num_pulse_baseline_offset:, :param.nslice_to_use], xarea_resampled, area_resampled


def scale_data(s, bottom_pct=2.5, top_pct=2.5, divide_by_top=True):
    def mean_pct_portion(x, pct, fromtop=False):
        x_sorted = np.sort(x)
        if fromtop:
            x_sorted = x_sorted[::-1]
        n = max(1, round(len(x_sorted) * pct / 100))
        return np.mean(x_sorted[:n])
    s_copy = s.copy()
    for ch in range(s_copy.shape[1]):
        baseline = mean_pct_portion(s_copy[:, ch], bottom_pct)
        s_copy[:, ch] -= baseline
    if divide_by_top:
        top_val = mean_pct_portion(s_copy[:, 0], top_pct, fromtop=True)
        s_copy /= top_val
    return s_copy
    
    
def load_network(state_filename, param):
    model_state = torch.load(state_filename, map_location=torch.device('cpu'))['model_state_dict']
    num_input = param.data_simulation.num_input_features
    input_size = param.data_simulation.input_feature_size
    output_size = param.data_simulation.output_feature_size
    model = TOFinverse(nfeature_in=num_input, nfeature_out=1, 
                    input_size=input_size,
                    output_size=output_size)
    model.load_state_dict(model_state)
    return model


def infer_velocity(model, s_data_for_nn, xarea, area):
    velocity_NN = utils.input_batched_signal_into_NN_area(s_data_for_nn, model, xarea, area)
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
    num_pulse_baseline_offset = param.scan_param.num_pulse_baseline_offset
    velocity_NN = utils.upsample(velocity_NN, velocity_NN.size*100+1, tr).flatten()
    t = np.arange(0, tr*npulse, tr/100)
    t_with_baseline, v_with_baseline = utils.add_baseline_period(t, velocity_NN, tr*num_pulse_baseline_offset)
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t_with_baseline, 
                        vts=v_with_baseline, xarea=xarea, area=area)
    s_raw = tm.simulate_inflow(tr, te, npulse+num_pulse_baseline_offset, w, fa, t1, t2, nslice, alpha, multi_factor, 
                                x_func_area, multithread=True, enable_logging=True)
    s = s_raw[num_pulse_baseline_offset:, :param.nslice_to_use]
    return s


def compute_metrics(param, x_ref, x_pred, tr=None, nperseg=256):

    ntime = min(x_ref.shape[0], x_pred.shape[0])
    x_ref = x_ref[:ntime, :]
    x_pred = x_pred[:ntime, :]

    nslice = x_ref.shape[1]
    fs = 1.0 / tr
    
    # ---- Amplitude weights (based on RMS per slice of the reference signal) ----
    rms = np.sqrt(np.mean(x_ref**2, axis=0))  # RMS per channel
    rms[rms == 0] = np.nan  # avoid div-by-zero
    weights = 1.0 / rms
    
    per_slice = {}
    summary = {}
    summary_acc = {
        "MSE": [], "MSE_weight": [], "Pearson": [], "R2": [], "PSD_MSE": [], "PSD_MSE_weight": [], "Lag_sec": []
    }
    
    band_names = []
    if param is not None and "sampling" in param and hasattr(param.sampling, "bounding_gaussians"):
        band_names = [g["name"] for g in param.sampling.bounding_gaussians]
        for band in band_names:
            summary_acc[f"PSD_MSE_{band}"] = []
            summary_acc[f"PSD_MSE_{band}_weight"] = []
    
    for islice in range(nslice):
        ref = x_ref[:, islice]
        pred = x_pred[:, islice]
        residual = ref - pred
        weight = weights[islice]

        # --- time-domain metrics ---
        mse = np.mean(residual**2)
        weighted_mse = weight * mse
        
        pearson = float(np.corrcoef(ref, pred)[0, 1])
        ss_res = np.sum((ref - pred) ** 2)
        ss_tot = np.sum((ref - np.mean(ref)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        f_ref, P_ref = welch(ref, fs=fs, nperseg=min(nperseg, len(ref)))
        f_pred, P_pred = welch(pred, fs=fs, nperseg=min(nperseg, len(pred)))
        psd_mse = float(np.mean((P_ref - P_pred) ** 2))
        weighted_psd_mse = weight * psd_mse
        
        # --- band-specific PSD errors ---
        band_psd_mse = {}
        if band_names:
            for g in param.sampling.bounding_gaussians:
                band = g["name"]
                fmin, fmax = g["freq_range"]
                mask = (f_ref >= fmin) & (f_ref <= fmax)
                if np.any(mask):
                    band_err = np.mean((P_ref[mask] - P_pred[mask]) ** 2)
                else:
                    band_err = np.nan
                band_psd_mse[band] = float(band_err)
                summary_acc[f"PSD_MSE_{band}"].append(band_err)
                summary_acc[f"PSD_MSE_{band}_weight"].append(weight * band_err)
                
        # zero-mean signals for cross-corr
        r = ref - ref.mean()
        p = pred - pred.mean()
        corr = np.correlate(r, p, mode='full')
        lag_idx = int(np.argmax(corr) - (len(ref) - 1))
        if tr is not None:
            lag_sec = float(lag_idx * tr)
        else:
            lag_sec = int(lag_idx)  # samples

        per_slice[f"slice{islice+1}"] = {
            "MSE": float(mse),
            "Weighted_MSE": float(weighted_mse),
            "Pearson": None if np.isnan(pearson) else float(pearson),
            "R2": float(r2),
            "PSD_MSE": float(psd_mse),
            "Weighted_PSD_MSE": float(weighted_psd_mse),
            "Lag_sec": float(lag_sec)
        }
        if band_names:
            for band, val in band_psd_mse.items():
                per_slice[f"slice{islice+1}"][f"PSD_MSE_{band}"] = val
                per_slice[f"slice{islice+1}"][f"Weighted_PSD_MSE_{band}"] = weight * val
                
        # append to summary accumulators
        summary_acc["MSE"].append(mse)
        summary_acc["MSE_weight"].append(weight * mse)
        summary_acc["Pearson"].append(np.nan if np.isnan(pearson) else pearson)
        summary_acc["R2"].append(r2)
        summary_acc["PSD_MSE"].append(psd_mse)
        summary_acc["PSD_MSE_weight"].append(weight * psd_mse)
        summary_acc["Lag_sec"].append(lag_sec)

    # compute summary (mean across channels)
    for k, vals in summary_acc.items():
        vals = np.array(vals, dtype=float)
        summary[k + "_mean"] = float(np.nanmean(vals))
        summary[k + "_std"] = float(np.nanstd(vals))

    # Also compute normalized weighted means for interpretability
    for metric, weighted_key in [("MSE", "MSE_weight"), ("PSD_MSE", "PSD_MSE_weight")]:
        wmean = np.nansum(summary_acc[weighted_key]) / np.nansum(weights)
        summary[metric + "_weighted_mean"] = float(wmean)

    return {"per_slice": per_slice, "summary": summary}


def save_metrics_json(metrics_dict, outpath):
    with open(outpath, "w") as f:
        json.dump(metrics_dict, f, indent=2)


if __name__ == "__main__":
    main()