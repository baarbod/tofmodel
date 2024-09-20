# -*- coding: utf-8 -*-

from multiprocessing import Pool
import multiprocessing.shared_memory as msm
from functools import partial
import time
import numpy as np
import json
import sys
import pickle
import os
import shutil

import tofmodel.inverse.utils as utils
import tofmodel.inverse.io as io
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm
from tofmodel.path import ROOT_DIR
import inflowan.processing as proc

debug_mode = 0

if debug_mode:
    config_path = os.path.join(ROOT_DIR, "config", "config.json")
    with open(config_path, "r") as jsonfile:
            param = json.load(jsonfile)
    task_id = 1   
    nbatch = param['data_simulation']['num_batches']
else:
    # read config json file
    if len(sys.argv) > 1:
        config_path = os.path.join(ROOT_DIR, "config", str(sys.argv[2]))
        print(config_path)
        with open(config_path, "r") as jsonfile:
            param = json.load(jsonfile)
    else:
        print('Missing config file argument!')

    # define batch number for job array
    print(sys.argv[1])
    task_id = int(sys.argv[1])
    nbatch = param['data_simulation']['num_batches']
    print(f"on task {task_id} of {nbatch}")


# frequency parameters
fstart = param['data_simulation']['frequency_start']
fend = param['data_simulation']['frequency_end']
fspacing = param['data_simulation']['frequency_spacing']
frequencies = np.arange(fstart, fend, fspacing)

# sampling parameters
sampling_param = param['sampling']
fresp_lower = sampling_param['fresp_lower']
fresp_upper = sampling_param['fresp_upper']
fcard_lower = sampling_param['fcard_lower']
fcard_upper = sampling_param['fcard_upper']
voffset_lower = sampling_param['voffset_lower']
voffset_upper = sampling_param['voffset_upper']
fslow_fact = sampling_param['fslow_fact']
fslow_exp = sampling_param['fslow_exp']
breath_fsd = sampling_param['breath_fsd']
cardiac_fsd = sampling_param['cardiac_fsd']
global_lower_bound = sampling_param['global_lower_bound']
global_offset = sampling_param['global_offset']
gauss_noise_lower = sampling_param['gauss_noise_lower']
gauss_noise_upper = sampling_param['gauss_noise_upper']
area_scale_lower = sampling_param['area_scale_lower']
area_scale_upper = sampling_param['area_scale_upper']
slc1_offset_lower = sampling_param['slc1_offset_lower']
slc1_offset_upper= sampling_param['slc1_offset_upper']

# input/output parameters
nsample = param['data_simulation']['num_samples']
nbatch = param['data_simulation']['num_batches']
batch_size = nsample // nbatch
num_input = param['data_simulation']['num_input_features']
input_size = param['data_simulation']['input_feature_size']
num_output = param['data_simulation']['num_output_features']
output_size = param['data_simulation']['output_feature_size']
Xshape = (batch_size, num_input, input_size)
Yshape = (batch_size, num_output, output_size)
Xtype = param['data_simulation']['Xtype']
Ytype = param['data_simulation']['Ytype']

# scan parameters
scan_param = param['scan_param']
scan_param["num_pulse"] = input_size
scan_param["num_pulse_baseline_offset"] = 20

# load subject area information
config_data_path = '/om/user/bashen/repositories/inflow-analysis/config/config_data.json'
Ax, Ay, subjects = io.load_subject_area_matrix(config_data_path, input_size)

dataset_root = '/om/user/bashen/repositories/tofmodel/data/simulated'
dataset_name = param['info']['name']

dataset_batched_dir = os.path.join(dataset_root, dataset_name, 'inputs_batched')
os.makedirs(dataset_batched_dir, exist_ok=True)

dataset_full_dir = os.path.join(dataset_root, dataset_name, 'inputs_all')
os.makedirs(dataset_full_dir, exist_ok=True)

dataset_sorted_dir = os.path.join(dataset_root, dataset_name, 'inputs_all_sorted')
os.makedirs(dataset_sorted_dir, exist_ok=True)

dataset_simulated_batched_dir = os.path.join(dataset_root, dataset_name, 'simulated_batched')
os.makedirs(dataset_simulated_batched_dir, exist_ok=True)


def prepare_simulation_inputs():
    
    print(f"running for batch size {batch_size}")
    
    # generate for all samples in dataset
    input_data = define_input_params(batch_size)

    # for compute number of protons for every sample
    with Pool() as pool:
        x0_array_list = pool.starmap(compute_sample_init_position, enumerate(input_data))    
    
    inputs = [{'input_data': input_data[i], 'x0_array': x0_array_list[i]} for i in range(len(input_data))]
    
    inputs_path = os.path.join(dataset_batched_dir, f"inputs_list_{len(inputs)}_samples_task{task_id}.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs, f)
    
    
def sort_simulation_inputs():

    inputs_all_samples = []
    nproton_list = []
    
    # check and fill in missed batches before sorting
    if len(os.listdir(dataset_batched_dir)) != nbatch:
        num_missing_batch = np.abs(len(os.listdir(dataset_batched_dir)) - nbatch)
        print(f"rerunning {batch_size*num_missing_batch} samples for {num_missing_batch} missing batches")
        input_data = define_input_params(batch_size*num_missing_batch)
        with Pool() as pool:
            x0_array_list = pool.starmap(compute_sample_init_position, enumerate(input_data))    
            inputs = [{'input_data': input_data[i], 'x0_array': x0_array_list[i]} for i in range(len(input_data))]
            inputs_path_missing = os.path.join(dataset_batched_dir, f"inputs_list_{len(inputs)}_samples_missing_batch_REDO.pkl")
            with open(inputs_path_missing, "wb") as f:
                pickle.dump(inputs, f)
    
    # combine all samples into lists
    for batch_name in os.listdir(dataset_batched_dir):
        path = os.path.join(dataset_batched_dir, batch_name)
        with open(path, "rb") as f:
            batch_inputs = pickle.load(f)
        for sample_input in batch_inputs:
            inputs_all_samples.append(sample_input)
            nproton_list.append(sample_input['x0_array'].size)
        
    inputs_path = os.path.join(dataset_full_dir, f"inputs_list_{len(inputs_all_samples)}_samples.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs_all_samples, f)
    
    # save sorted input data list 
    sort_indices = np.argsort(nproton_list)
    inputs_all_samples_sorted = [inputs_all_samples[i] for i in sort_indices]
    inputs_path = os.path.join(dataset_sorted_dir, f"inputs_list_sorted_{len(inputs_all_samples_sorted)}_samples.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs_all_samples_sorted, f)    

    if len(inputs_all_samples_sorted) != nsample:
        print(f"WARNING: total number of inputs do not match defined number of samples")
 
def get_sampling_bounds(ff, f1, ff_card=1):
    
    N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2)
    slow = np.exp(fslow_exp * -ff)/fslow_fact
    resp = (N(ff, f1, breath_fsd) + N(ff, 2*f1, breath_fsd)/3 + N(ff, 3*f1, breath_fsd)/6) / 3.33
    card = N(ff, ff_card, cardiac_fsd) / 3.33
    
    upper = slow + resp + card + global_offset
    lower = np.ones(np.shape(ff))*global_lower_bound

    return np.column_stack((lower, upper))


def define_input_params(num_sample):
    input_data = []

    # random sampling constrained by lower and upper bounds
    for _ in range(num_sample):
        
        # define frequency bounds
        mu = np.random.uniform(low=fresp_lower, high=fresp_upper)
        ff_card = np.random.uniform(low=fcard_lower, high=fcard_upper) 
        bound_array = get_sampling_bounds(frequencies, mu, ff_card=ff_card)
        
        # define velocity amplitudes and timeshifts
        rand_numbers = np.random.uniform(low=bound_array[:, 0], high=bound_array[:, 1])
        rand_phase = np.random.uniform(low=0, high=1/frequencies)
        v_offset = np.random.uniform(low=voffset_lower, high=voffset_upper)

        # define noise injection
        gauss_noise_std = np.random.uniform(low=gauss_noise_lower, high=gauss_noise_upper)
        
        # define cross-sectional area
        area_subject_ind = np.random.randint(len(subjects))
        area_scale_factor = np.random.uniform(low=area_scale_lower, high=area_scale_upper)
        slc1_offset = np.random.uniform(low=slc1_offset_lower, high=slc1_offset_upper)
        xarea = Ax[:, area_subject_ind]
        area = Ay[:, area_subject_ind]
        widest_position = xarea[np.argmax(area)]
        xarea_sample = xarea - widest_position - slc1_offset
        area_sample = area_scale_factor * area
        
        # store all variables
        input_data.append([tuple(frequencies), v_offset, rand_phase, tuple(rand_numbers), 
                                scan_param, Xshape, Xtype, Yshape, Ytype, task_id, 
                                xarea_sample, area_sample, gauss_noise_std])

    return input_data


def compute_sample_init_position(isample, sample_input_data):
    
    # sample_input_data = input_data[isample]
    
    # unpack tuple of inputs
    frequencies, v_offset, rand_phase, velocity_input, \
    scan_param, Xshape, Xtype, Yshape, Ytype, task_id, \
    xarea_sample, area_sample, gauss_noise_std = sample_input_data
    
    tr = scan_param['repetition_time']
    w = scan_param['slice_width']
    nslice = scan_param['num_slice']
    npulse = scan_param["num_pulse"]
    npulse_offset = scan_param["num_pulse_baseline_offset"]

    # # FOR DEBUGGING
    # np.random.seed(isample + task_id)
    # x0_array = np.random.rand(np.random.randint(6000))
    
    # define velocity
    t = tr * np.linspace(0, npulse, Yshape[2])
    v = utils.define_velocity_fourier(t, velocity_input, frequencies, rand_phase, v_offset)
    
    # add initial zero-flow baseline period 
    baseline_duration = tr*npulse_offset
    t_with_baseline, v_with_baseline = utils.add_baseline_period(t, v, baseline_duration)
    
    # define position function
    x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t_with_baseline, 
                        vts=v_with_baseline, xarea=xarea_sample, area=area_sample)
            
    dx = 0.01
    x0_array = tm.set_init_positions(x_func_area, tr, w, npulse + npulse_offset, nslice, dx, progress=False)
        
    return x0_array


def simulate_parameter_set(idx, input_data):
    
    # unpack tuple of inputs
    frequencies, v_offset, rand_phase, velocity_input, \
    scan_param, Xshape, Xtype, Yshape, Ytype, task_id, \
    xarea_sample, area_sample, gauss_noise_std = input_data['input_data']
    
    x0_array = input_data['x0_array']
    
    # scan parameters
    tr = scan_param['repetition_time']
    w = scan_param['slice_width']
    fa = scan_param['flip_angle']
    t1 = scan_param['t1_time']
    nslice = scan_param['num_slice']
    npulse = scan_param['num_pulse']
    npulse_offset = scan_param["num_pulse_baseline_offset"]
    multi_factor = scan_param['MBF']
    alpha = scan_param['alpha_list']
    
    # get shared memory arrays 
    Xshm = msm.SharedMemory(name=f"Xarray{task_id}")
    Yshm = msm.SharedMemory(name=f"Yarray{task_id}")
    X = np.ndarray(Xshape, dtype=Xtype, buffer=Xshm.buf)
    Y = np.ndarray(Yshape, dtype=Ytype, buffer=Yshm.buf)
    
    # define velocity
    t = tr * np.linspace(0, npulse, Yshape[2])
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
                               x_func_area, x0_array_given=x0_array, multithread=False)[:, 0:3]
    
    # # FOR DEBUGGING
    # s_raw = np.random.rand(npulse, 3)
    
    # remove inital baseline period from signal
    s_raw = s_raw[npulse_offset:, :]
    
    # preprocess raw simulated signal 
    s = proc.scale_epi(s_raw)
    s -= np.mean(s, axis=0)
    
    # # Add zero-mean gaussian noise
    # mean = 0
    # noise = np.random.normal(mean, gauss_noise_std, (Xshape[2], 3))
    # s += noise

    # fill matricies
    X[idx, 0, :] = s[:, 0].squeeze()
    X[idx, 1, :] = s[:, 1].squeeze()
    X[idx, 2, :] = s[:, 2].squeeze()
    X[idx, 3, :] = xarea_sample
    X[idx, 4, :] = area_sample
    
    Y[idx, 0, :] = v
    
    
def run_simulation():

    if len(os.listdir(dataset_sorted_dir)) > 1:
        print(f"found {len(os.listdir(dataset_sorted_dir))} but expected 1 file in sorted dataset inputs folder")
        return 0
    
    file = os.listdir(dataset_sorted_dir)[0]
    inputs_path = os.path.join(dataset_sorted_dir, file)
    with open(inputs_path, 'rb') as f:
        input_data = pickle.load(f)
    input_data_batch = input_data[task_id*batch_size:(task_id+1)*batch_size]
    
    for i in range(len(input_data_batch)):
        input_data_batch[i]['input_data'][9] = task_id

    # mark start time
    tstart = time.time()

    X = np.zeros(Xshape)
    Y = np.zeros(Yshape)
    X_shared = utils.create_shared_memory(X, name=f"Xarray{task_id}")
    Y_shared = utils.create_shared_memory(Y, name=f"Yarray{task_id}")

    # if __name__ == '__main__':
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

    # create folder associated with this simulation
    os.makedirs(dataset_simulated_batched_dir, exist_ok=True)

    # log simulation experiment information
    config_path = os.path.join(dataset_simulated_batched_dir, 'config_used.json')
    with open(config_path, 'w') as fp:
        json.dump(param, fp, indent=4)

    # save training data simulation information
    output_path = os.path.join(dataset_simulated_batched_dir, f"output_{Xshape[0]}_samples_task{task_id}" '.pkl')
    print('Saving updated training_data set...')   
    with open(output_path, "wb") as f:
        pickle.dump([x, y], f)
    print('Finished saving.')   

    # close shared memory
    utils.close_shared_memory(name=X_shared.name)
    utils.close_shared_memory(name=Y_shared.name)


def combine_simulated_batches():
    
    folder = os.path.join(dataset_root, dataset_name)
    os.makedirs(folder, exist_ok=True)

    x_list = []
    y_list = []
    config_flag = 0
    for file in os.listdir(dataset_simulated_batched_dir):
        if file.endswith('.pkl'):
            filepath = os.path.join(dataset_simulated_batched_dir, file)
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    x, y = pickle.load(f)
                x_list.append(x)
                y_list.append(y)
                print(f"combined and removed {file}")
                os.remove(filepath)
        if file.endswith('_used.json') and not config_flag:
            filepath = os.path.join(dataset_simulated_batched_dir, file)
            shutil.move(filepath, folder)
            config_flag = 1

    x = np.concatenate(x_list, axis=0)    
    y = np.concatenate(y_list, axis=0)

    # save training data simulation information
    filename = f"output_{y.shape[0]}_samples.pkl"
    filepath = os.path.join(folder, filename)
    print('Saving updated training_data set...')   
    with open(filepath, "wb") as f:
        pickle.dump([x, y], f)
    print('Finished saving.')   


def cleanup_directories():
    
    shutil.rmtree(dataset_batched_dir)
    print(f"temporary inputs batched directory {dataset_batched_dir} removed succesfully")

    shutil.rmtree(dataset_full_dir)
    print(f"temporary all inputs directory {dataset_full_dir} removed succesfully")
    
    shutil.rmtree(dataset_simulated_batched_dir)
    print(f"temporary simulated batched directory {dataset_simulated_batched_dir} removed succesfully")
    

if __name__ == '__main__':
    
    # run given function name
    globals()[sys.argv[3]]()
