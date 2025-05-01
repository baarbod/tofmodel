# -*- coding: utf-8 -*-

from multiprocessing import Pool
import multiprocessing.shared_memory as msm
from functools import partial
import time
import numpy as np
import pickle
import os
import shutil
import argparse
from omegaconf import OmegaConf

import tofmodel.inverse.utils as utils
import tofmodel.inverse.io as io
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm
from tofmodel.path import ROOT_DIR


def prepare_simulation_inputs(param, dirs, task_id):
    
    batch_size = param.data_simulation.num_samples // param.data_simulation.num_batches
    print(f"running for batch size {batch_size}")
    
    # generate inputs for all samples in the dataset
    input_data = define_input_params(batch_size, param, task_id)

    # for compute number of protons for every sample
    with Pool() as pool:
        X0_list = pool.starmap(compute_sample_init_positions, enumerate(input_data))    
    
    inputs = [{'input_data': input_data[i], 'x0_array': X0_list[i]} for i in range(len(input_data))]
    
    inputs_path = os.path.join(dirs['batched'], f"inputs_list_{len(inputs)}_samples_task{task_id:03}.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs, f)
    

def sort_simulation_inputs(param, dirs):

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
        print(f"saved to {inputs_path}")
        with open(inputs_path, "wb") as f:
            pickle.dump(chunk, f)  
 
 
def get_sampling_bounds(param, frequencies):

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
        input_data.append([tuple(frequencies), v_offset, rand_phase, tuple(rand_numbers), 
                                scan_param, Xshape, Xtype, Yshape, Ytype, task_id, 
                                xarea_sample, area_sample, gauss_noise_std])

    return input_data


def compute_sample_init_positions(isample, sample_input_data):
    
    # sample_input_data = input_data[isample]
    
    # unpack tuple of inputs
    frequencies, v_offset, rand_phase, velocity_input, \
    scan_param, Xshape, Xtype, Yshape, Ytype, task_id, \
    xarea_sample, area_sample, gauss_noise_std = sample_input_data
    
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
    print(f"running initial position loop...")
    lower_bound, upper_bound = tm.get_init_position_bounds(x_func_area, np.unique(timings), w, nslice)
    x0_array = np.arange(lower_bound, upper_bound + dx, dx)

    return x0_array


def simulate_parameter_set(idx, input_data):
    
    # unpack tuple of inputs
    frequencies, v_offset, rand_phase, velocity_input, \
    scan_param, Xshape, Xtype, Yshape, Ytype, task_id, \
    xarea_sample, area_sample, gauss_noise_std = input_data['input_data']
        
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
    X[idx, 0, :] = s[:, 0].squeeze()
    X[idx, 1, :] = s[:, 1].squeeze()
    X[idx, 2, :] = s[:, 2].squeeze()
    X[idx, 3, :] = xarea_sample
    X[idx, 4, :] = area_sample
    
    v_downsample = utils.downsample(v, Yshape[2])
    
    Y[idx, 0, :] = v_downsample
    
    
def run_simulation(param, dirs, task_id):
    
    # find the correct input data batch based on the task_id
    for file in os.listdir(dirs['sorted']):
        if f"task{task_id:03}.pkl" in file:
            inputs_path = os.path.join(dirs['sorted'], file)
            with open(inputs_path, 'rb') as f:
                input_data_batch = pickle.load(f)
    
    print(f"loaded {inputs_path}")
    
    # mark start time
    tstart = time.time()

    # override array shapes based on actual dimensions
    batch_size = len(input_data_batch)
    Xshape = (batch_size, param.data_simulation.num_input_features, param.data_simulation.input_feature_size)
    Yshape = (batch_size, param.data_simulation.num_output_features, param.data_simulation.output_feature_size)
    for i in range(len(input_data_batch)):
        input_data_batch[i]['input_data'][9] = task_id
        input_data_batch[i]['input_data'][5] = Xshape
        input_data_batch[i]['input_data'][7] = Yshape

    X = np.zeros(Xshape)
    Y = np.zeros(Yshape)
    X_shared = utils.create_shared_memory(X, name=f"Xarray{task_id}")
    Y_shared = utils.create_shared_memory(Y, name=f"Yarray{task_id}")

    # call pool pointing to simulation routine
    with Pool(processes=len(os.sched_getaffinity(0))) as pool:
        print(f"running process with {os.cpu_count()} cpu cores ({len(os.sched_getaffinity(0))} usable)")
        pool.starmap(simulate_parameter_set, enumerate(input_data_batch))

    # define the numpy arrays to save
    x = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf)
    y = np.ndarray(Y.shape, dtype=Y.dtype, buffer=Y_shared.buf)

    # mark end time
    tfinal = time.time()
    tstr = '{:3.2f}'.format(tfinal - tstart)
    print(f"total simulation time = {tstr}")
    print(f"total number of samples = {x.shape[0]}")

    # log simulation experiment information
    config_path = os.path.join(dirs['sim_batched'], 'config_used.yml')
    OmegaConf.save(config=OmegaConf.create(param), f=config_path)
    
    # save training data simulation information
    output_path = os.path.join(dirs['sim_batched'], f"output_{Xshape[0]}_samples_task{task_id:03}" '.pkl')
    print('Saving updated training_data set...')   
    with open(output_path, "wb") as f:
        pickle.dump([x, y, input_data_batch], f)
    print('Finished saving.')   

    # close shared memory
    utils.close_shared_memory(name=X_shared.name)
    utils.close_shared_memory(name=Y_shared.name)


def combine_simulated_batches(param, dirs):

    dataset_name = param.info.name
    dataset_root = os.path.join(ROOT_DIR, 'data', 'simulated')
    folder = os.path.join(dataset_root, dataset_name)
    x_list = []
    y_list = []
    inputs_list = []
    config_flag = 0
    for file in os.listdir(dirs['sim_batched']):
        if file.endswith('.pkl'):
            filepath = os.path.join(dirs['sim_batched'], file)
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    x, y, input_data_batch = pickle.load(f)
                x_list.append(x)
                y_list.append(y)
                inputs_list +=  input_data_batch
                print(f"combined and NOT removed {file}")
                # os.remove(filepath)
        if file.endswith('_used.yml') and not config_flag:
            filepath = os.path.join(dirs['sim_batched'], file)
            shutil.move(filepath, folder)
            config_flag = 1

    x = np.concatenate(x_list, axis=0)    
    y = np.concatenate(y_list, axis=0)

    # save training data simulation information
    filename = f"output_{y.shape[0]}_samples.pkl"
    filepath = os.path.join(folder, filename)
    print('Saving updated training_data set...')   
    with open(filepath, "wb") as f:
        pickle.dump([x, y, inputs_list], f)
    print('Finished saving.')   


def cleanup_directories(dirs):
    
    shutil.rmtree(dirs['batched'])
    print(f"temporary inputs batched directory {dirs['batched']} removed succesfully")

    shutil.rmtree(dirs['full'])
    print(f"temporary all inputs directory {dirs['full']} removed succesfully")
    
    shutil.rmtree(dirs['sim_batched'])
    print(f"temporary simulated batched directory {dirs['sim_batched']} removed succesfully")
    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dataset simulations and preprocessing batches"
    )
    parser.add_argument("task_id", type=int, help="Batch/task ID (1-based)")
    parser.add_argument("config_file", type=str, help="Config JSON filename under ROOT_DIR/config")
    parser.add_argument("action", type=str,
                        choices=[
                            "prepare_simulation_inputs", "sort_simulation_inputs",
                            "run_simulation", "combine_simulated_batches"
                        ], help="Operation to perform")
    return parser.parse_args()


def load_config(path):
    return OmegaConf.load(path)


def setup_directories(dataset_name):
    dataset_root = os.path.join(ROOT_DIR, 'data', 'simulated')
    base = os.path.join(dataset_root, dataset_name)
    dirs = {
        'batched': os.path.join(base, 'inputs_batched'),
        'full': os.path.join(base, 'inputs_all'),
        'sorted': os.path.join(base, 'inputs_all_sorted'),
        'sim_batched': os.path.join(base, 'simulated_batched')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def main():
    args = parse_args()
    param = load_config(args.config_file)
    
    # override some parameters
    param.scan_param.num_pulse = param.data_simulation.input_feature_size
    param.scan_param.num_pulse_baseline_offset = 20
    
    dirs = setup_directories(param.info.name)
    action = args.action
    
    if action == 'prepare_simulation_inputs':
        nbatch = param.data_simulation.num_batches
        print(f"on task {args.task_id} of {nbatch}")
        prepare_simulation_inputs(param, dirs, args.task_id)
        
    elif action == 'sort_simulation_inputs':
        sort_simulation_inputs(param, dirs)
        
    elif action == 'run_simulation':
        run_simulation(param, dirs, args.task_id)
        
    elif action == 'combine_simulated_batches':
        combine_simulated_batches(param, dirs)
        
    elif action == 'cleanup_directories':
        cleanup_directories(dirs)
        
    else:
        raise ValueError(f"Unknown action: {action}")

if __name__ == '__main__':
    main()
