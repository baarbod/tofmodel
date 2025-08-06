# -*- coding: utf-8 -*-

from multiprocessing import Pool
import multiprocessing.shared_memory as msm
from functools import partial
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
import time
import numpy as np
import pickle
import os
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
    """Initialize logger for worker processes to log to stdout.

    Parameters
    ----------
    log_level : int
        Logging verbosity level (e.g., logging.INFO).
    """
    
    logger = multiprocessing.get_logger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(processName)s] %(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(log_level)
    
    
def prepare_inputs(param, dirs, task_id):
    """Generate and save simulation input samples with initial proton positions.

    Parameters
    ----------
    param : OmegaConf
        Configuration with simulation and sampling parameters.
    
    dirs : dict
        Dictionary containing paths for storing intermediate files.

    task_id : int
        ID number for this task/batch.
    """
    
    ds_param = param.data_simulation
    batch_size = ds_param.num_samples // ds_param.num_batches
    input_data_batch = define_input_params(batch_size, param, task_id)
    num_usable_cpu = len(os.sched_getaffinity(0))
    num_workers = min(batch_size, num_usable_cpu) 
    with Pool(processes=num_workers, initializer=setup_worker_logger) as pool:
        x0_list = pool.starmap(compute_sample_init_positions, enumerate(input_data_batch))    
    inputs = [{'input_data': input_data_batch[i], 'x0_array': x0_list[i]} for i in range(len(input_data_batch))]
    inputs_path = os.path.join(dirs['batched'], f"inputs_list_{len(inputs)}_samples_task{task_id:03}.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs, f)
    

def sort_inputs(dir_unsorted, dir_sorted, batch_size):
    """Load all input batches, sort them by proton count, and redistribute into sorted batches.

    Parameters
    ----------
    dir_unsorted : str
        Path to directory with unsorted batches

    dir_sorted : str
        Path to directory to put sorted batches
        
    batch_size : int
        Number of samples to have in sorted batches
        
    dirs : dict
        Directory paths containing batched and sorted inputs.
    """
    
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
 
 
def get_sampling_bounds(frequencies, bounding_gaussians, lower_fact, upper_fact, global_offset):
    """Compute lower and upper bounds for velocity amplitude sampling using Gaussians.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Array of frequencies to use.
    
    bounding_gaussians: list
        List of dicts containing parameters for each Gaussian.
        
    lower_fact: float
        Factor to multiply lower boun.

    upper_fact: float
        Factor to multiply upper bound.
        
    global_offset: float
        Factor to add to upper bound.
        
    Returns
    -------
    bounds : numpy.ndarray
        Array of shape (len(frequencies), 2) with lower and upper bounds.
    """
    
    N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2) # gaussian distribution
    Gtotal = np.zeros_like(frequencies)
    for g in bounding_gaussians:
        amp = np.random.uniform(low=0, high=g['scale'])
        freq = np.random.uniform(*g['freq_range'])
        fsd = np.random.uniform(*g['fsd_range'])
        G = amp * N(frequencies, freq, fsd)
        for harm in g.get('harmonics', []):
            G += amp/harm[1] * N(frequencies, freq*harm[0], fsd)
        Gtotal += G  
    upper = upper_fact * Gtotal + global_offset
    lower = lower_fact * Gtotal
    return np.column_stack((lower, upper))


def define_sample_area(param):
    """Generate cross-sectional area arrays.

    Parameters
    ----------
    param : OmegaConf
        Configuration parameters for sampling and simulation.

    Returns
    -------
    input_data : tuple
        (xarea, area). areas cm^2 at each x position
    """
    
    logger = logging.getLogger(__name__)
    sampling_param = param.sampling
    simulation_param = param.data_simulation
    area_param = sampling_param.cross_sectional_area
    if area_param.mode == 'straight_tube':
        x = np.linspace(-3, 3, simulation_param.output_feature_size)
        a = np.ones(simulation_param.output_feature_size)
    elif area_param.mode == 'custom':
        area_data = np.loadtxt(area_param.path_to_custom)
        x, a = area_data[:, 0], area_data[:, 1]
    elif area_param.mode == 'collection':
        files = sorted(os.listdir(area_param.path_to_collection))
        if not files:
            logger.error(f"No area files found in {area_param.path_to_collection}")
            raise FileNotFoundError(f"No valid area files in {area_param.path_to_collection}")
        selected_area_file = np.random.choice(files)
        selected_area_path = os.path.join(area_param.path_to_collection, selected_area_file)
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
    
    
def define_input_params(num_sample, param, task_id):
    """Generate randomized input parameter dictionaries for simulation.

    Parameters
    ----------
    num_sample : int
        Number of samples to generate.
    
    param : OmegaConf
        Configuration parameters for sampling and simulation.
    
    task_id : int
        Task identifier for this batch.

    Returns
    -------
    input_data : list of dict
        List of dictionaries with simulation parameters per sample.
    """
    
    input_data = []

    sp = param.sampling
    ds = param.data_simulation
    frequencies = np.arange(ds.frequency_start, ds.frequency_end, ds.frequency_spacing)
    for _ in range(num_sample):
        bounds = get_sampling_bounds(frequencies, sp.bounding_gaussians, sp.lower_fact, sp.upper_fact, sp.global_offset)
        amplitude = np.random.uniform(bounds[:, 0], bounds[:, 1])
        phase = np.random.uniform(low=0, high=1/frequencies)
        voff = np.random.uniform(low=sp.voffset_lower, high=sp.voffset_upper)
        xarea, area = define_sample_area(param)
        
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
    """Compute initial spatial positions for protons based on velocity and area profile.

    Parameters
    ----------
    isample : int
        Dummy argument needed for enumerate to work when passing to Pool call.

    input_data : dict
        Dictionary of parameters for the sample.

    Returns
    -------
    x0_array : numpy.ndarray
        Array of initial proton positions (in cm).
    """

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


def simulate_parameter_set(isample, inputs):
    """Simulate inflow signal for one sample using its parameter set and write to shared memory array.

    Parameters
    ----------
    isample : int
        Sample index in the batch used for indexing into shared memory arrays.

    inputs : dict
        Contains 'input_data' and other metadata.

    Returns
    -------
    t_start, t_end : float
        Start and end times of the simulation used for reporting.
    """
    
    t_start = time.time()
    input_data = inputs['input_data']
    p = input_data['param'].scan_param
    Xshm = msm.SharedMemory(name=f"Xarray{input_data['task_id']}")
    Yshm = msm.SharedMemory(name=f"Yarray{input_data['task_id']}")
    X = np.ndarray(input_data['Xshape'], dtype=input_data['Xtype'], buffer=Xshm.buf)
    Y = np.ndarray(input_data['Yshape'], dtype=input_data['Ytype'], buffer=Yshm.buf)
    t = np.arange(0, p.num_pulse*p.repetition_time, p.repetition_time/100)
    v = utils.define_velocity_fourier(t, input_data['velocity_input'], 
                                      input_data['frequencies'], input_data['rand_phase'], 
                                      input_data['v_offset'])
    t_with_baseline, v_with_baseline = utils.add_baseline_period(t, v, p.repetition_time*p.num_pulse_baseline_offset)
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t_with_baseline, 
                          vts=v_with_baseline, xarea=input_data['xarea_sample'], area=input_data['area_sample'])
    s_raw = tm.simulate_inflow(p.repetition_time, p.echo_time, p.num_pulse+p.num_pulse_baseline_offset, 
                               p.slice_width, p.flip_angle, p.t1_time, p.t2_time, p.num_slice, p.alpha_list, 
                               p.MBF, x_func_area, multithread=False)[:, 0:3]
    s = s_raw[p.num_pulse_baseline_offset:, :]
    X[isample,0:3,:] = s.T
    X[isample,3,:], X[isample,4,:] = input_data['xarea_sample'], input_data['area_sample']
    Y[isample, 0, :] = utils.downsample(v, input_data['Yshape'][2])
    t_end = time.time()
    return (t_start, t_end)


def run_simulations(param, dirs, task_id):
    """Run simulations for a task batch and saved the resulting filled shared memory arrays.

    Parameters
    ----------
    param : OmegaConf
        Configuration parameters for simulation.
    
    dirs : dict
        Dictionary of directory paths for input/output.

    task_id : int
        Task number to identify which input batch to run.
    """
    
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
    Xshape = (batch_size, param.data_simulation.num_input_features, param.data_simulation.input_feature_size)
    Yshape = (batch_size, param.data_simulation.num_output_features, param.data_simulation.output_feature_size)
    X = np.zeros(Xshape)
    Y = np.zeros(Yshape)
    X_shared = utils.create_shared_memory(X, name=f"Xarray{task_id}")
    Y_shared = utils.create_shared_memory(Y, name=f"Yarray{task_id}")
    num_usable_cpu = len(os.sched_getaffinity(0))
    num_workers = min(batch_size, num_usable_cpu) 
    with Pool(processes=num_workers, initializer=setup_worker_logger) as pool:
        times = pool.starmap(simulate_parameter_set, enumerate(inputs_batch))
    start_times, end_times = zip(*times)
    start_times, end_times = np.array(start_times), np.array(end_times)
    tref = np.min(start_times)
    start_times -= tref
    end_times -= tref
    logger.info(f"summary of timing (relative to start of first simulation) across {len(times)} simulations")
    for idx, (tstart, tend) in enumerate(zip(start_times, end_times)):
        logger.info(f"  simulation: {idx}, start time :{tstart:.3f} seconds, total simulation time: {(tend - tstart):.3f} seconds")
    total_times = end_times - start_times
    logger.info(f"Mean simulation time: {np.mean(total_times):.3f} +- {np.std(total_times):.3f}")
    config_path = os.path.join(dirs['sim_batched'], 'config_used.yml')
    OmegaConf.save(config=OmegaConf.create(param), f=config_path)
    output_path = os.path.join(dirs['sim_batched'], f"output_{Xshape[0]}_samples_task{task_id:03}" '.pkl')
    logger.info('Saving updated training_data set...')
    with open(output_path, "wb") as f:
        pickle.dump([np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf), 
                     np.ndarray(Y.shape, dtype=Y.dtype, buffer=Y_shared.buf), inputs_batch], f)
    logger.info('Finished saving.')
    utils.close_shared_memory(name=X_shared.name)
    utils.close_shared_memory(name=Y_shared.name)


def combine_simulations(param, dirs):
    """Combine all simulated batches into one dataset and save to output directory.

    Parameters
    ----------
    param : OmegaConf
        Configuration with path to final output directory.
    
    dirs : dict
        Directory paths containing simulation results.
    """
    
    logger = logging.getLogger(__name__)
    datasetdir = param.paths.datasetdir
    xs, ys, ins = [],[],[]
    config_flag = 0
    sim_files = os.listdir(dirs['sim_batched'])
    if not sim_files:
        logger.warning(f"No files found in {dirs['sim_batched']}. Exiting.")
    else:
        for file in sim_files:
            if file.endswith('.pkl'):
                filepath = os.path.join(dirs['sim_batched'], file)
                if os.path.isfile(filepath):
                    with open(filepath, "rb") as f:
                        x, y, b = pickle.load(f)
                    xs.append(x); ys.append(y); ins.extend(b)
                    logger.info(f"combined {file}")
            if file.endswith('_used.yml') and not config_flag:
                filepath = os.path.join(dirs['sim_batched'], file)
                shutil.move(filepath, datasetdir)
                config_flag = 1
    if xs and ys:
        logger.info(f'Combined {len(xs)} batches')   
        logger.info('Saving updated training_data set...')   
        filename = f"output_{len(xs)*y.shape[0]}_samples.pkl"
        filepath = os.path.join(datasetdir, filename)
        with open(filepath, "wb") as f:
            pickle.dump([np.concatenate(xs, axis=0) , np.concatenate(ys, axis=0), ins], f)
        logger.info('Finished saving.')   
    else:
        logger.warning("No valid .pkl files processed. Nothing to save.")


def cleanup_directories(dirs):
    """Remove temporary directories used during simulation.

    Parameters
    ----------
    dirs : dict
        Dictionary of directory paths to remove.
    """
    
    logger = logging.getLogger(__name__)
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        logger.info(f"Successfully removed directory: {d}")