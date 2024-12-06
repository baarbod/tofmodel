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
fslow_lower = sampling_param['fslow_lower']
fslow_upper = sampling_param['fslow_upper']
fresp_lower = sampling_param['fresp_lower']
fresp_upper = sampling_param['fresp_upper']
fcard_lower = sampling_param['fcard_lower']
fcard_upper = sampling_param['fcard_upper']
voffset_lower = sampling_param['voffset_lower']
voffset_upper = sampling_param['voffset_upper']
fslow_scale = sampling_param['fslow_scale']
fresp_scale = sampling_param['fresp_scale']
fcard_scale = sampling_param['fcard_scale']
slow_fsd = sampling_param['slow_fsd']
breath_fsd = sampling_param['breath_fsd']
cardiac_fsd = sampling_param['cardiac_fsd']
global_lower_bound = sampling_param['global_lower_bound']
global_offset = sampling_param['global_offset']
gauss_noise_lower = sampling_param['gauss_noise_lower']
gauss_noise_upper = sampling_param['gauss_noise_upper']
area_scale_lower = sampling_param['area_scale_lower']
area_scale_upper = sampling_param['area_scale_upper']
slc1_offset_lower = sampling_param['slc1_offset_lower']
slc1_offset_upper = sampling_param['slc1_offset_upper']
prob_phantom_mode = sampling_param['prob_phantom_mode']
fphantom_scale = sampling_param['fphantom_scale']
phantom_noise_fact = sampling_param['phantom_noise_fact']

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
        X0_list = pool.starmap(compute_sample_init_positions, enumerate(input_data))    
    
    inputs = [{'input_data': input_data[i], 'x0_array': X0_list[i]} for i in range(len(input_data))]
    
    inputs_path = os.path.join(dataset_batched_dir, f"inputs_list_{len(inputs)}_samples_task{task_id:03}.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs, f)
    

def sort_simulation_inputs():

    inputs_all_samples = []
    nproton_list = []

    # combine all samples into lists
    for batch_name in os.listdir(dataset_batched_dir):
        path = os.path.join(dataset_batched_dir, batch_name)
        with open(path, "rb") as f:
            batch_inputs = pickle.load(f)
        for sample_input in batch_inputs:
            inputs_all_samples.append(sample_input)
            nproton_list.append(sample_input['x0_array'].shape[0])
        
    inputs_path = os.path.join(dataset_full_dir, f"inputs_list_{len(inputs_all_samples)}_samples.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(inputs_all_samples, f)

    # sort dataset by nproton
    sort_indices = np.argsort(nproton_list)
    inputs_all_samples_sorted = [inputs_all_samples[i] for i in sort_indices]
        
    # distribute sorted full dataset into nbatch batches
    nsample = len(inputs_all_samples_sorted)
    batch_size = nsample // nbatch
    chunks = [inputs_all_samples_sorted[x:x+batch_size] for x in range(0, len(inputs_all_samples_sorted), batch_size)]
    
    for idx, chunk in enumerate(chunks):
        override_task_id = idx + 1
        inputs_path = os.path.join(dataset_sorted_dir, f"inputs_list_{len(chunk)}_samples_task{override_task_id:03}.pkl")
        print(f"saved to {inputs_path}")
        with open(inputs_path, "wb") as f:
            pickle.dump(chunk, f)  
 
 
def get_sampling_bounds(ff, fslow, fresp, fcard):
    
    N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2)
    slow = fslow_scale * N(ff, fslow, slow_fsd)
    resp = fresp_scale * (N(ff, fresp, breath_fsd) + N(ff, 2*fresp, breath_fsd)/3 + N(ff, 3*fresp, breath_fsd)/6)
    card = fcard_scale * N(ff, fcard, cardiac_fsd)
    
    upper = slow + resp + card + global_offset
    lower = np.ones(np.shape(ff))*global_lower_bound

    return np.column_stack((lower, upper))


def define_input_params(num_sample):
    input_data = []

    # random sampling constrained by lower and upper bounds
    for _ in range(num_sample):
        
        p = np.random.uniform() # probabilty used to choose sampling mode (human or phantom)
        
        prob_phantom_mode = 0.2
        if p < prob_phantom_mode:
            
            # randomly choose one of the phantom frequencies
            phantom_frequencies = [0.05, 0.1, 0.2]
            freq_ind = np.random.randint(len(phantom_frequencies))
            osc_freq = phantom_frequencies[freq_ind]
            
            # randomly assign amplitude at the frequency, with two harmonics
            rand_numbers = np.zeros(frequencies.size)
            osc_mag = np.random.uniform(low=0, high=fphantom_scale)            
            rand_numbers[int(osc_freq / fspacing) - 1] = osc_mag
            rand_numbers[int(2*osc_freq / fspacing) - 1] = osc_mag / 3
            rand_numbers[int(3*osc_freq / fspacing) - 1] = osc_mag / 6
            rand_phase = np.random.uniform(low=0, high=1/frequencies)

            v_offset = 0

            # define reduced noise injection for phantom
            gauss_noise_std = np.random.uniform(low=gauss_noise_lower, high=gauss_noise_upper*phantom_noise_fact)
            
            # define cross-sectional area
            area_subject_ind = -1
            xarea_sample = Ax[:, area_subject_ind]
            area_sample =Ay[:, area_subject_ind]
        else:
            # define frequency bounds
            fslow = np.random.uniform(low=fslow_lower, high=fslow_upper)
            fresp = np.random.uniform(low=fresp_lower, high=fresp_upper)
            fcard = np.random.uniform(low=fcard_lower, high=fcard_upper) 
            bound_array = get_sampling_bounds(frequencies, fslow, fresp, fcard)
            
            # define velocity amplitudes and timeshifts
            rand_numbers = np.random.uniform(low=bound_array[:, 0], high=bound_array[:, 1])
            rand_phase = np.random.uniform(low=0, high=1/frequencies)
            
            v_offset = np.random.uniform(low=voffset_lower, high=voffset_upper)

            # define noise injection
            gauss_noise_std = np.random.uniform(low=gauss_noise_lower, high=gauss_noise_upper)
            
            # define cross-sectional area
            area_subject_ind = np.random.randint(len(subjects) - 1) # subtract one to not include straight tube
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


def compute_sample_init_positions(isample, sample_input_data):
    
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
    alpha = np.array(scan_param['alpha_list'], ndmin=2).T
    
    # # FOR DEBUGGING
    # np.random.seed(isample + task_id)
    # x0_array = np.random.rand(np.random.randint(6000))
    
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
    
    # Xproton = input_data['Xproton']
    
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
    
    # # FOR DEBUGGING
    # s_raw = np.random.rand(npulse, 3)
    
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
    
    
def run_simulation():
    
    # find the correct input data batch based on the task_id
    for file in os.listdir(dataset_sorted_dir):
        if f"task{task_id:03}.pkl" in file:
            inputs_path = os.path.join(dataset_sorted_dir, file)
            with open(inputs_path, 'rb') as f:
                input_data_batch = pickle.load(f)
    
    print(f"loaded {inputs_path}")
    
    # mark start time
    tstart = time.time()

    # override array shapes based on actual dimensions
    batch_size = len(input_data_batch)
    Xshape = (batch_size, num_input, input_size)
    Yshape = (batch_size, num_output, output_size)
    for i in range(len(input_data_batch)):
        input_data_batch[i]['input_data'][9] = task_id
        input_data_batch[i]['input_data'][5] = Xshape
        input_data_batch[i]['input_data'][7] = Yshape

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

    # log simulation experiment information
    config_path = os.path.join(dataset_simulated_batched_dir, 'config_used.json')
    with open(config_path, 'w') as fp:
        json.dump(param, fp, indent=4)

    # save training data simulation information
    output_path = os.path.join(dataset_simulated_batched_dir, f"output_{Xshape[0]}_samples_task{task_id:03}" '.pkl')
    print('Saving updated training_data set...')   
    with open(output_path, "wb") as f:
        pickle.dump([x, y, input_data_batch], f)
    print('Finished saving.')   

    # close shared memory
    utils.close_shared_memory(name=X_shared.name)
    utils.close_shared_memory(name=Y_shared.name)


def combine_simulated_batches():

    folder = os.path.join(dataset_root, dataset_name)
    x_list = []
    y_list = []
    inputs_list = []
    config_flag = 0
    for file in os.listdir(dataset_simulated_batched_dir):
        if file.endswith('.pkl'):
            filepath = os.path.join(dataset_simulated_batched_dir, file)
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    x, y, input_data_batch = pickle.load(f)
                x_list.append(x)
                y_list.append(y)
                inputs_list +=  input_data_batch
                print(f"combined and NOT removed {file}")
                # os.remove(filepath)
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
        pickle.dump([x, y, inputs_list], f)
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
