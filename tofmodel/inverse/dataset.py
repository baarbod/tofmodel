# -*- coding: utf-8 -*-

from multiprocessing import Pool
import multiprocessing.shared_memory as msm
from functools import partial
from scipy.interpolate import interp1d
import numpy as np
import pickle
import os
import csv
import shutil
import logging
import multiprocessing
import sys
import tofmodel.inverse.utils as utils
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm


# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def setup_worker_logger(log_level=logging.INFO):
    """Initialize logger for worker processes to log to stdout."""
    
    logger = multiprocessing.get_logger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(processName)s] %(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(log_level)
    
    
def prepare_inputs(param, dirs, task_id):
    """Generate and save simulation input samples with initial proton positions."""
    
    ds_param = param.data_simulation
    batch_size = ds_param.num_samples // ds_param.num_batches
    input_data_batch = define_input_params(batch_size, param, dirs, task_id)
    num_usable_cpu = len(os.sched_getaffinity(0))
    num_workers = min(batch_size, num_usable_cpu) 
    with Pool(processes=num_workers, initializer=setup_worker_logger) as pool:
        x0_list = pool.starmap(compute_sample_init_positions, enumerate(input_data_batch))    
    inputs = [{'input_data': input_data_batch[i], 'x0_array': x0_list[i]} for i in range(len(input_data_batch))]
    inputs_path = os.path.join(dirs['batched'], f"inputs_list_{len(inputs)}_samples_task{task_id:03}.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs, f)
    

def sort_inputs(dir_unsorted, dir_sorted, batch_size):
    """Load all input batches, sort them by proton count, and redistribute into sorted batches."""
    
    logger = logging.getLogger(__name__)
    inputs_all_samples = []
    nproton_list = []
    for batch_name in os.listdir(dir_unsorted):
        path = os.path.join(dir_unsorted, batch_name)
        with open(path, "rb") as f:
            batch_inputs = pickle.load(f)
        for sample_input in batch_inputs:
            inputs_all_samples.append(sample_input)
            nproton_list.append(sample_input['x0_array'].shape[0])
    sort_indices = np.argsort(nproton_list)[::-1]
    inputs_all_samples_sorted = [inputs_all_samples[i] for i in sort_indices]
    batches = [inputs_all_samples_sorted[x:x+batch_size] for x in range(0, len(inputs_all_samples_sorted), batch_size)]
    for i, batch in enumerate(batches):
        new_task_id = i+1
        for sample in batch:
            sample['input_data']['task_id'] = new_task_id         
        inputs_path = os.path.join(dir_sorted, f"inputs_list_{len(batch)}_samples_task{new_task_id:03}.pkl")
        with open(inputs_path, "wb") as f:
            pickle.dump(batch, f)  
        logger.info(f"saved to {inputs_path}")
 
 
def get_sampling_bounds(frequencies, bounding_gaussians, lower_fact, upper_fact, global_offset, kdes=None):
    """Compute lower and upper bounds for velocity amplitude sampling using Gaussians."""
    
    N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2) # gaussian distribution
    Gtotal = np.zeros_like(frequencies)
    for g in bounding_gaussians:
        if kdes:
            amp = kdes[f"{g['name']}_amp"].resample(1).flatten()
            fsd = kdes[f"{g['name']}_width"].resample(1).flatten()
        else:
            amp = np.random.uniform(low=0, high=g['scale'])
            fsd = np.random.uniform(*g['fsd_range'])
        freq = np.random.uniform(*g['freq_range'])
        G = amp * N(frequencies, freq, fsd)
        for harm in g.get('harmonics', []):
            G += amp/harm[1] * N(frequencies, freq*harm[0], fsd)
        Gtotal += G  
    upper = upper_fact * Gtotal + global_offset
    lower = lower_fact * Gtotal
    return np.column_stack((lower, upper))


def define_sample_area(param, dirs=None):
    """Generate cross-sectional area arrays."""
    
    logger = logging.getLogger(__name__)
    sampling_param = param.sampling
    simulation_param = param.data_simulation
    area_param = sampling_param.cross_sectional_area
    if area_param.mode == 'straight_tube':
        x = np.linspace(-3, 3, simulation_param.output_feature_size)
        a = np.ones(simulation_param.output_feature_size)
    elif area_param.mode == 'collection':
        path_to_collection = os.path.join(dirs['data'], 'area_collection')
        files = sorted(os.listdir(path_to_collection))
        if not files:
            logger.error(f"No area files found in {path_to_collection}")
            raise FileNotFoundError(f"No valid area files in {path_to_collection}")
        selected_area_file = np.random.choice(files)
        selected_area_path = os.path.join(path_to_collection, selected_area_file)
        area_data = np.loadtxt(selected_area_path)
        x, a = area_data[:, 0], area_data[:, 1]
        scale = np.random.uniform(low=area_param.area_scale_lower, high=area_param.area_scale_upper)
        offset = np.random.uniform(low=area_param.slc1_offset_lower, high=area_param.slc1_offset_upper)
        widest_position = x[np.argmax(a)]
        x, a = x-widest_position-offset, scale * a  
    def resample(x, y, n):
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        x_new = np.linspace(x.min(), x.max(), n)
        return x_new, f(x_new)
    return resample(x, a, simulation_param.input_feature_size)
    
    
def define_input_params(num_sample, param, dirs, task_id):
    """Generate randomized input parameter dictionaries for simulation."""
    
    input_data = []

    sp = param.sampling
    if sp.mode == 'data':
        path_to_velocity_kde = os.path.join(dirs['data'], 'velocity_kde.pkl')
        with open(path_to_velocity_kde, 'rb') as f:
            kdes = pickle.load(f)
    elif sp.mode == 'uniform':
        kdes = None
    
    ds = param.data_simulation
    frequencies = np.arange(ds.frequency_start, ds.frequency_end, ds.frequency_spacing)
    for _ in range(num_sample):
        bounds = get_sampling_bounds(frequencies, sp.bounding_gaussians, sp.lower_fact, sp.upper_fact, sp.global_offset, kdes=kdes)
        amplitude = np.random.uniform(bounds[:, 0], bounds[:, 1])
        phase = np.random.uniform(low=0, high=1/frequencies)
        voff = np.random.uniform(low=sp.voffset_lower, high=sp.voffset_upper)
        xarea, area = define_sample_area(param, dirs=dirs)
        
        batch_size = ds.num_samples // ds.num_batches
        Xshape = (batch_size, ds.num_input_features, ds.input_feature_size)
        Yshape = (batch_size, ds.num_output_features, ds.output_feature_size)
        Xtype = ds.Xtype
        Ytype = ds.Ytype
        input_data.append({
            'frequencies': tuple(frequencies),
            'v_offset': voff,
            'rand_phase': phase,
            'velocity_input': tuple(amplitude),
            'param': param,
            'Xshape': Xshape,
            'Xtype': Xtype,
            'Yshape': Yshape,
            'Ytype': Ytype,
            'task_id': task_id,
            'xarea_sample': xarea,
            'area_sample': area,
        })
    return input_data


def compute_sample_init_positions(isample, input_data):
    """Compute initial spatial positions for protons based on velocity and area profile."""

    p = input_data['param'].scan_param
    t = np.arange(0, p.num_pulse*p.repetition_time, p.repetition_time/100)
    v = utils.define_velocity_fourier(t, input_data['velocity_input'], input_data['frequencies'], input_data['rand_phase'], input_data['v_offset'])
    t_with_baseline, v_with_baseline = utils.add_baseline_period(t, v, p.repetition_time*p.num_pulse_baseline_offset)
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t_with_baseline, 
                        vts=v_with_baseline, xarea=input_data['xarea_sample'], area=input_data['area_sample'])
    timings, _ = tm.get_pulse_targets(p.repetition_time, p.num_slice, p.num_pulse + p.num_pulse_baseline_offset, 
                                      np.array(p.alpha_list, ndmin=2).T)
    logger = multiprocessing.get_logger()
    logger.info(f"running initial position loop...")
    lb, ub = tm.get_init_position_bounds(x_func_area, np.unique(timings), p.slice_width, p.num_slice)
    logger.info(f"initial position bounds: ({lb:.3f}, {ub:.3f}) cm")
    return np.arange(lb, ub + 0.01, 0.01)


def simulate_parameter_set(isample, inputs, batch_dir):
    """Simulate inflow signal for one sample using its parameter set and write to shared memory array."""
    logger = multiprocessing.get_logger()
    input_data = inputs['input_data']
    
    # Create batch subdirectory
    batch_subdir = os.path.join(batch_dir, f"batch_{input_data['task_id']:03}")
    os.makedirs(batch_subdir, exist_ok=True)
    progress_file = os.path.join(batch_subdir, "progress.csv")
    
    # File for this sample
    sample_file = os.path.join(batch_subdir, f"sample_{isample:03}.pkl")
    
    try:
        logger.info(f"Starting simulation {isample} in batch {input_data['task_id']}")
        sys.stdout.flush()
        p = input_data['param'].scan_param
        t = np.arange(0, p.num_pulse*p.repetition_time, p.repetition_time/100)
        v = utils.define_velocity_fourier(t, input_data['velocity_input'], 
                                        input_data['frequencies'], input_data['rand_phase'], 
                                        input_data['v_offset'])
        t_with_baseline, v_with_baseline = utils.add_baseline_period(t, v, p.repetition_time*p.num_pulse_baseline_offset)
        x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t_with_baseline, 
                            vts=v_with_baseline, xarea=input_data['xarea_sample'], area=input_data['area_sample'])
        s_raw = tm.simulate_inflow(p.repetition_time, p.echo_time, p.num_pulse+p.num_pulse_baseline_offset, 
                                p.slice_width, p.flip_angle, p.t1_time, p.t2_time, p.num_slice, p.alpha_list, 
                                p.MBF, x_func_area, multithread=False)[:, :input_data['param'].nslice_to_use]
        s = s_raw[p.num_pulse_baseline_offset:, :]
        
        v_downsampled = utils.downsample(v, input_data['param'].data_simulation.output_feature_size)
        
        # ---- save sample ----
        with open(sample_file, "wb") as f:
            pickle.dump({
                'X': s,
                'v': v_downsampled,
                'xarea': input_data['xarea_sample'],
                'area': input_data['area_sample'],
                'input': input_data
            }, f)
        
        # ---- update progress CSV ----
        with open(progress_file, "a", newline='') as pf:
            writer = csv.writer(pf)
            writer.writerow([isample, s.shape[0], "done"])

        logger.info(f"Finished simulation {isample} in batch {input_data['task_id']}, saved to {sample_file}")
        sys.stdout.flush()
        
    except Exception as e:
        logger.error(f"Simulation {isample} in batch {input_data['task_id']} failed: {e}")
        sys.stdout.flush()
        with open(progress_file, "a", newline='') as pf:
            writer = csv.writer(pf)
            writer.writerow([isample, 0, "error"])


def run_simulations(param, dirs, task_id):
    """Run simulations for a task batch and saved the resulting filled shared memory arrays."""
    
    logger = logging.getLogger(__name__)
    input_file = None
    for file in os.listdir(dirs['sorted']):
        if file.startswith(f"inputs_list_") and file.endswith(f"task{task_id:03}.pkl"):
            input_file = os.path.join(dirs['sorted'], file)
    if input_file is None:
        logger.error(f"No file found for task ID {task_id}")
        return
    with open(input_file, 'rb') as f:
        inputs_batch = pickle.load(f)
    logger.info(f"loaded {input_file}")
    batch_size = len(inputs_batch)

    num_usable_cpu = len(os.sched_getaffinity(0))
    num_workers = min(batch_size, num_usable_cpu)
    logger.info(f"Running {batch_size} samples on {num_workers} workers")
    with Pool(processes=num_workers, initializer=setup_worker_logger) as pool:
        pool.starmap(
            partial(simulate_parameter_set, batch_dir=dirs['sim_batched']),
            list(enumerate(inputs_batch))
        )
    logger.info(f"Batch {task_id} completed. Results in {os.path.join(dirs['sim_batched'], f'batch_{task_id:03}')}")


def combine_simulations(param, dirs):
    """Combine all simulated batches into one dataset and save to output directory."""
    
    logger = logging.getLogger(__name__)
    X_final, y_final, inputs_final = [], [], []

    # Find all batch directories
    batch_dirs = sorted(
        [d for d in os.listdir(dirs['sim_batched']) if d.startswith("batch_")]
    )

    if not batch_dirs:
        logger.warning(f"No batch directories found in {dirs['sim_batched']}. Nothing to combine.")
        return

    # Load all per-sample .pkl files
    for batch_name in batch_dirs:
        batch_subdir = os.path.join(dirs['sim_batched'], batch_name)
        sample_files = sorted(f for f in os.listdir(batch_subdir) if f.endswith(".pkl"))
        for sf in sample_files:
            sample_path = os.path.join(batch_subdir, sf)
            try:
                with open(sample_path, "rb") as f:
                    data = pickle.load(f)
                xx = np.concatenate((data['X'], 
                                     np.expand_dims(data['xarea'], axis=1), 
                                     np.expand_dims(data['area'], axis=1)), axis=1)
                
                xx = np.swapaxes(xx, -1, -2)  
                yy = np.expand_dims(data['v'], axis=0)  

                # Filter NaNs and Infs
                if np.isnan(xx).any() or np.isinf(xx).any():
                    continue  # skip invalid sample
                
                # Append to final lists
                X_final.append(xx)
                y_final.append(yy)
                inputs_final.append(data['input'])
                
            except Exception as e:
                logger.warning(f"Failed to load {sample_path}: {e}")

    if not X_final:
        logger.warning("No valid samples found. Exiting without saving.")
        return

    # Stack data
    X_final = np.stack(X_final, axis=0)  
    y_final = np.stack(y_final, axis=0) 
    total_samples = X_final.shape[0]
    
    if getattr(param, "n_dataset_split", 0) > 0:
        N = param.n_dataset_split
        div = total_samples // N

        for k in range(1, N + 1):
            n_keep = div * k if k < N else total_samples
            rand_ind = np.random.choice(total_samples, n_keep, replace=False)
            X_sub = X_final[rand_ind]
            y_sub = y_final[rand_ind]
            inputs_sub = [inputs_final[i] for i in rand_ind]

            sub_file = os.path.join(dirs['dataset'], f"dataset_{n_keep}_samples.pkl")
            with open(sub_file, "wb") as f:
                pickle.dump([X_sub, y_sub, inputs_sub], f)

            logger.info(f"Saved split {k}/{N}: {sub_file} "
                        f"(X: {X_sub.shape}, y: {y_sub.shape}, inputs: {len(inputs_sub)})")
    else:
        # Save combined dataset
        output_file = os.path.join(dirs['dataset'], f"dataset_{total_samples}_samples.pkl")
        with open(output_file, "wb") as f:
            pickle.dump([X_final, y_final, inputs_final], f)

        logger.info(f"Combined dataset saved to {output_file}")
        logger.info(f"X shape: {X_final.shape}, y shape: {y_final.shape}, inputs: {len(inputs_final)}")

def cleanup_directories(dirs):
    """Remove temporary directories used during simulation."""
    logger = logging.getLogger(__name__)
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        logger.info(f"Successfully removed directory: {d}")