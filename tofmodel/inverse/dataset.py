# -*- coding: utf-8 -*-

from multiprocessing import Pool
import multiprocessing.shared_memory as msm
from functools import partial
from omegaconf import OmegaConf
import time
import numpy as np
import pickle
import os
import shutil
import logging
import multiprocessing
import sys

import tofmodel.inverse.utils as utils
import tofmodel.inverse.io as io
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm


# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)


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
    
    # generate inputs for all samples in the dataset
    batch_size = param.data_simulation.num_samples // param.data_simulation.num_batches
    input_data_batch = define_input_params(batch_size, param, task_id)

    # determine number of workers based on batch_size and available CPU cores
    num_usable_cpu = len(os.sched_getaffinity(0))
    num_workers = min(batch_size, num_usable_cpu) 

    # for compute number of protons for every sample
    with Pool(processes=num_workers, initializer=setup_worker_logger) as pool:
        x0_list = pool.starmap(compute_sample_init_positions, enumerate(input_data_batch))    
    
    inputs = [{'input_data': input_data_batch[i], 'x0_array': x0_list[i]} for i in range(len(input_data_batch))]
    
    inputs_path = os.path.join(dirs['batched'], f"inputs_list_{len(inputs)}_samples_task{task_id:03}.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs, f)
    

def sort_inputs(param, dirs):
    """Load all input batches, sort them by proton count, and redistribute into sorted batches.

    Parameters
    ----------
    param : OmegaConf
        Simulation configuration.

    dirs : dict
        Directory paths containing batched and sorted inputs.
    """
    
    logger = logging.getLogger(__name__)
    
    inputs_all_samples = []
    nproton_list = []

    # combine all samples into lists
    for batch_name in os.listdir(dirs['batched']):
        path = os.path.join(dirs['batched'], batch_name)
        with open(path, "rb") as f:
            batch_inputs = pickle.load(f)
        for sample_input in batch_inputs:
            inputs_all_samples.append(sample_input)
            nproton_list.append(sample_input['x0_array'].shape[0])
        
    inputs_path = os.path.join(dirs['full'], f"inputs_list_{len(inputs_all_samples)}_samples.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs_all_samples, f)

    # sort dataset by nproton
    sort_indices = np.argsort(nproton_list)
    inputs_all_samples_sorted = [inputs_all_samples[i] for i in sort_indices]
        
    # distribute sorted full dataset into nbatch batches
    batch_size = param.data_simulation.num_samples // param.data_simulation.num_batches
    chunks = [inputs_all_samples_sorted[x:x+batch_size] for x in range(0, len(inputs_all_samples_sorted), batch_size)]
    
    for idx, chunk in enumerate(chunks):
        override_task_id = idx + 1
        inputs_path = os.path.join(dirs['sorted'], f"inputs_list_{len(chunk)}_samples_task{override_task_id:03}.pkl")
        logger.info(f"saved to {inputs_path}")
        with open(inputs_path, "wb") as f:
            pickle.dump(chunk, f)  
 
 
def get_sampling_bounds(param, frequencies):
    """Compute lower and upper bounds for velocity amplitude sampling using Gaussians.

    Parameters
    ----------
    param : OmegaConf
        Sampling configuration parameters.
    
    frequencies : numpy.ndarray
        Array of frequency values.

    Returns
    -------
    bounds : numpy.ndarray
        Array of shape (len(frequencies), 2) with lower and upper bounds.
    """
    
    sampling_param = param.sampling
    bounding_gaussians = sampling_param.bounding_gaussians
    N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2) # gaussian
    
    Gtotal = np.zeros_like(frequencies)
    
    for gtype in bounding_gaussians:
        amp = np.random.uniform(low=0, high=gtype['scale'])
        freq = np.random.uniform(low=gtype['range'][0], high=gtype['range'][1])
        fsd = gtype['fsd']
        
        G = amp*N(frequencies, freq, fsd)
        
        if 'harmonics' in gtype:
            for harmonic in gtype['harmonics']:
                amp_harmonic = amp/harmonic[1]
                freq_harmonic = freq*harmonic[0]
                G += amp_harmonic*N(frequencies, freq_harmonic, fsd)

        Gtotal += G
        
    upper = sampling_param.upper_fact * Gtotal + sampling_param.global_offset
    lower = sampling_param.lower_fact * Gtotal

    return np.column_stack((lower, upper))


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

    sampling_param = param.sampling
    simulation_param = param.data_simulation
    scan_param = param.scan_param
    frequencies = np.arange(simulation_param.frequency_start, simulation_param.frequency_end, simulation_param.frequency_spacing)
    
    # load subject area information
    config_data_path = '/om/user/bashen/repositories/tofmodel/config/config_data.json'
    Ax, Ay, subjects = io.load_subject_area_matrix(config_data_path, simulation_param.input_feature_size)
    
    # random sampling constrained by lower and upper bounds
    for _ in range(num_sample):

        bound_array = get_sampling_bounds(param, frequencies)
        
        # define velocity amplitudes and timeshifts
        rand_numbers = np.random.uniform(low=bound_array[:, 0], high=bound_array[:, 1])
        rand_phase = np.random.uniform(low=0, high=1/frequencies)
        
        v_offset = np.random.uniform(low=sampling_param.voffset_lower, 
                                        high=sampling_param.voffset_upper)

        # define noise injection
        gauss_noise_std = np.random.uniform(low=sampling_param.gauss_noise_lower, 
                                            high=sampling_param.gauss_noise_upper)
        
        # define cross-sectional area
        area_subject_ind = np.random.randint(len(subjects) - 1) # subtract one to not include straight tube
        area_scale_factor = np.random.uniform(low=sampling_param.area_scale_lower, 
                                                high=sampling_param.area_scale_upper)
        slc1_offset = np.random.uniform(low=sampling_param.slc1_offset_lower, 
                                        high=sampling_param.slc1_offset_upper)
        xarea = Ax[:, area_subject_ind]
        area = Ay[:, area_subject_ind]
        widest_position = xarea[np.argmax(area)]
        xarea_sample = xarea - widest_position - slc1_offset
        area_sample = area_scale_factor * area
            
        batch_size = simulation_param.num_samples // simulation_param.num_batches
        Xshape = (batch_size, simulation_param.num_input_features, simulation_param.input_feature_size)
        Yshape = (batch_size, simulation_param.num_output_features, simulation_param.output_feature_size)
        Xtype = simulation_param.Xtype
        Ytype = simulation_param.Ytype
        
        # store all variables
        input_data.append({
            'frequencies': tuple(frequencies),
            'v_offset': v_offset,
            'rand_phase': rand_phase,
            'velocity_input': tuple(rand_numbers),
            'scan_param': scan_param,
            'Xshape': Xshape,
            'Xtype': Xtype,
            'Yshape': Yshape,
            'Ytype': Ytype,
            'task_id': task_id,
            'xarea_sample': xarea_sample,
            'area_sample': area_sample,
            'gauss_noise_std': gauss_noise_std
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
    
    # unpack dict of inputs
    frequencies = input_data['frequencies']
    v_offset = input_data['v_offset']
    rand_phase = input_data['rand_phase']
    velocity_input = input_data['velocity_input']
    scan_param = input_data['scan_param']
    xarea_sample = input_data['xarea_sample']
    area_sample = input_data['area_sample']
    
    tr = scan_param.repetition_time
    w = scan_param.slice_width
    nslice = scan_param.num_slice
    npulse = scan_param.num_pulse
    npulse_offset = scan_param.num_pulse_baseline_offset
    alpha = np.array(scan_param.alpha_list, ndmin=2).T
    
    # define velocity
    dt = 0.1
    t = np.arange(0, npulse*tr, dt)
    v = utils.define_velocity_fourier(t, velocity_input, frequencies, rand_phase, v_offset)
    
    # add initial zero-flow baseline period 
    baseline_duration = tr*npulse_offset
    t_with_baseline, v_with_baseline = utils.add_baseline_period(t, v, baseline_duration)
    
    # define position function
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t_with_baseline, 
                        vts=v_with_baseline, xarea=xarea_sample, area=area_sample)
    
    npulse += npulse_offset
    dx = 0.01
    timings, _ = tm.get_pulse_targets(tr, nslice, npulse, alpha)
    logger = multiprocessing.get_logger()
    logger.info(f"running initial position loop...")
    lower_bound, upper_bound = tm.get_init_position_bounds(x_func_area, np.unique(timings), w, nslice)
    logger.info(f"initial position bounds: ({lower_bound:.3f}, {upper_bound:.3f}) cm")
    x0_array = np.arange(lower_bound, upper_bound + dx, dx)

    return x0_array


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
    t0, t1 : float
        Start and end times of the simulation used for reporting.
    """
    
    t0 = time.time()
    
    # unpack dict of inputs
    frequencies = inputs['input_data']['frequencies']
    v_offset = inputs['input_data']['v_offset']
    rand_phase = inputs['input_data']['rand_phase']
    velocity_input = inputs['input_data']['velocity_input']
    scan_param = inputs['input_data']['scan_param']
    Xshape = inputs['input_data']['Xshape']
    Xtype = inputs['input_data']['Xtype']
    Yshape = inputs['input_data']['Yshape']
    Ytype = inputs['input_data']['Ytype']
    task_id = inputs['input_data']['task_id']
    xarea_sample = inputs['input_data']['xarea_sample']
    area_sample = inputs['input_data']['area_sample']
    gauss_noise_std = inputs['input_data']['gauss_noise_std']
        
    # scan parameters
    tr = scan_param.repetition_time
    w = scan_param.slice_width
    fa = scan_param.flip_angle
    t1 = scan_param.t1_time
    nslice = scan_param.num_slice
    npulse = scan_param.num_pulse
    npulse_offset = scan_param.num_pulse_baseline_offset
    multi_factor = scan_param.MBF
    alpha = scan_param.alpha_list
    
    # get shared memory arrays 
    Xshm = msm.SharedMemory(name=f"Xarray{task_id}")
    Yshm = msm.SharedMemory(name=f"Yarray{task_id}")
    X = np.ndarray(Xshape, dtype=Xtype, buffer=Xshm.buf)
    Y = np.ndarray(Yshape, dtype=Ytype, buffer=Yshm.buf)
    
    # define velocity
    dt = 0.1
    t = np.arange(0, npulse*tr, dt)
    v = utils.define_velocity_fourier(t, velocity_input, frequencies, rand_phase, v_offset)
    
    # add initial zero-flow baseline period 
    baseline_duration = tr*npulse_offset
    t_with_baseline, v_with_baseline = utils.add_baseline_period(t, v, baseline_duration)
    
    # define position function
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t_with_baseline, 
                          vts=v_with_baseline, xarea=xarea_sample, area=area_sample)
    
    # add offset to number of simulation pulses
    npulse += npulse_offset
    
    # solve the tof forward model including the extra offset pulses
    # turn off multitreading because we are already distributing each simulation across cores 
    s_raw = tm.simulate_inflow(tr, npulse, w, fa, t1, nslice, alpha, multi_factor, 
                               x_func_area, multithread=False)[:, 0:3]
    
    # remove inital baseline period from signal
    s_raw = s_raw[npulse_offset:, :]
    
    # preprocess raw simulated signal 
    s = utils.scale_epi(s_raw)
    s -= np.mean(s, axis=0)
    
    # Add zero-mean gaussian noise
    mean = 0
    noise = np.random.normal(mean, gauss_noise_std, (Xshape[2], 3))
    s += noise

    # fill matricies
    X[isample, 0, :] = s[:, 0].squeeze()
    X[isample, 1, :] = s[:, 1].squeeze()
    X[isample, 2, :] = s[:, 2].squeeze()
    X[isample, 3, :] = xarea_sample
    X[isample, 4, :] = area_sample
    
    v_downsample = utils.downsample(v, Yshape[2])
    
    Y[isample, 0, :] = v_downsample
    
    t1 = time.time()
    return (t0, t1)


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
    
    # find the correct input data batch based on the task_id
    input_file = None
    for file in os.listdir(dirs['sorted']):
        if f"task{task_id:03}.pkl" in file:
            input_file = os.path.join(dirs['sorted'], file)
    
    if input_file is None:
        logger.error(f"No file found for task ID {task_id}")
        return

    # Load the input data batch
    with open(input_file, 'rb') as f:
        inputs_batch = pickle.load(f)
    logger.info(f"loaded {input_file}")
    
    batch_size = len(inputs_batch)
    Xshape = (batch_size, param.data_simulation.num_input_features, param.data_simulation.input_feature_size)
    Yshape = (batch_size, param.data_simulation.num_output_features, param.data_simulation.output_feature_size)
    for i in range(len(inputs_batch)):
        # renumber task_i, after having sorted the batches
        inputs_batch[i]['input_data']['task_id'] = task_id 
        # ensure dimensions are correct as well
        inputs_batch[i]['input_data']['Xshape'] = Xshape
        inputs_batch[i]['input_data']['Yshape'] = Yshape

    X = np.zeros(Xshape)
    Y = np.zeros(Yshape)
    X_shared = utils.create_shared_memory(X, name=f"Xarray{task_id}")
    Y_shared = utils.create_shared_memory(Y, name=f"Yarray{task_id}")

    # determine number of workers based on batch_size and available CPU cores
    num_usable_cpu = len(os.sched_getaffinity(0))
    num_workers = min(batch_size, num_usable_cpu) 

    # call pool pointing to simulation routine
    with Pool(processes=num_workers, initializer=setup_worker_logger) as pool:
        times = pool.starmap(simulate_parameter_set, enumerate(inputs_batch))

    # define the numpy arrays to save
    x = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf)
    y = np.ndarray(Y.shape, dtype=Y.dtype, buffer=Y_shared.buf)

    start_times, end_times = zip(*times)
    start_times = np.array(start_times)
    end_times = np.array(end_times)
    tref = np.min(start_times)
    start_times -= tref
    end_times -= tref
    logger.info(f"summary of timing (relative to start of first simulation) across {len(times)} simulations")
    for idx, (tstart, tend) in enumerate(zip(start_times, end_times)):
        logger.info(f"  simulation: {idx}, start time :{tstart:.3f} seconds, total simulation time: {(tend - tstart):.3f} seconds")
    total_times = end_times - start_times
    mn, sd = np.mean(total_times), np.std(total_times)
    logger.info(f"Mean simulation time: {mn:.3f} +- {sd:.3f}")
    
    # log simulation experiment information
    config_path = os.path.join(dirs['sim_batched'], 'config_used.yml')
    OmegaConf.save(config=OmegaConf.create(param), f=config_path)
    
    # save training data simulation information
    output_path = os.path.join(dirs['sim_batched'], f"output_{Xshape[0]}_samples_task{task_id:03}" '.pkl')
    logger.info('Saving updated training_data set...')
    with open(output_path, "wb") as f:
        pickle.dump([x, y, inputs_batch], f)
    logger.info('Finished saving.')

    # close shared memory
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
    
    outdir = param.paths.outdir
    x_list = []
    y_list = []
    inputs_list = []
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
                        x, y, input_data_batch = pickle.load(f)
                    x_list.append(x)
                    y_list.append(y)
                    inputs_list += input_data_batch
                    logger.info(f"combined and removed {file}")
                    os.remove(filepath)
            if file.endswith('_used.yml') and not config_flag:
                filepath = os.path.join(dirs['sim_batched'], file)
                shutil.move(filepath, outdir)
                config_flag = 1

    if x_list and y_list:
        x = np.concatenate(x_list, axis=0)    
        y = np.concatenate(y_list, axis=0)

        # save training data simulation information
        filename = f"output_{y.shape[0]}_samples.pkl"
        filepath = os.path.join(outdir, filename)
        logger.info(f'Combined {len(x_list)} samples')   
        logger.info('Saving updated training_data set...')   
        with open(filepath, "wb") as f:
            pickle.dump([x, y, inputs_list], f)
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
    
    shutil.rmtree(dirs['batched'])
    logger.info(f"Successfully removed directory: {dirs['batched']}")
    
    shutil.rmtree(dirs['full'])
    logger.info(f"Successfully removed directory: {dirs['full']}")

    shutil.rmtree(dirs['sorted'])
    logger.info(f"Successfully removed directory: {dirs['full']}")
    
    shutil.rmtree(dirs['sim_batched'])
    logger.info(f"Successfully removed directory: {dirs['sim_batched']}")