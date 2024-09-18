# -*- coding: utf-8 -*-

from multiprocessing import Pool
import multiprocessing.shared_memory as msm
from functools import partial
import numpy as np
import json
import sys
import time
import pickle
import os

import inflowan.processing as proc
import tofmodel.inverse.utils as utils
import tofmodel.inverse.io as io
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm
# from config.path import ROOT_DIR


ROOT_DIR = '/om/user/bashen/repositories/tofmodel'

debug_mode = 1
trim_input_length = 1 

if debug_mode:
    config_path = os.path.join(ROOT_DIR, "config", "config.json")
    with open(config_path, "r") as jsonfile:
            param = json.load(jsonfile)
    task_id = 1   
    nbatch = param['data_simulation']['num_batches']
    param['data_simulation']['num_cores'] = 1
else:
    # read config json file
    if len(sys.argv) > 1:
        config_path = os.path.join(ROOT_DIR, "config", str(sys.argv[-1]))
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


def get_sampling_bounds(ff, f1, ff_card=1):
    
    N = lambda x, u, s: np.exp((-0.5) * ((x-u)/s)**2)
    slow = np.exp(fslow_exp * -ff)/fslow_fact
    resp = (N(ff, f1, breath_fsd) + N(ff, 2*f1, breath_fsd)/3 + N(ff, 3*f1, breath_fsd)/6) / 3.33
    card = N(ff, ff_card, cardiac_fsd) / 3.33
    
    upper = slow + resp + card + global_offset
    lower = np.ones(np.shape(ff))*global_lower_bound

    return np.column_stack((lower, upper))


def simulate_parameter_set(idx, input_data):
    
    # unpack tuple of inputs
    frequencies, v_offset, rand_phase, velocity_input, \
    scan_param, Xshape, Xtype, Yshape, Ytype, task_id, \
    xarea_sample, area_sample, gauss_noise_std = input_data
    
    tr = scan_param['repetition_time']
    npulse = scan_param["num_pulse"]
    npulse_offset = scan_param["num_pulse_baseline_offset"]

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
    
    # solve the tof forward model including the extra offset pulses
    scan_param['num_pulse'] = npulse + npulse_offset
    s_raw = tm.simulate_inflow(scan_param, x_func_area, multithread=False)[:, 0:3]
    
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
    

# mark start time
tstart = time.time()
input_data = []

# random sampling constrained by lower and upper bounds
for _ in range(batch_size):
    
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
    input_data.append(tuple([tuple(frequencies), v_offset, rand_phase, tuple(rand_numbers), 
                             scan_param, Xshape, Xtype, Yshape, Ytype, task_id, 
                             xarea_sample, area_sample, gauss_noise_std]))

X = np.zeros(Xshape)
Y = np.zeros(Yshape)
X_shared = utils.create_shared_memory(X, name=f"Xarray{task_id}")
Y_shared = utils.create_shared_memory(Y, name=f"Yarray{task_id}")

if trim_input_length:
    input_data = input_data[:4]

# call pool pointing to simulation routine
if __name__ == '__main__':
    with Pool() as pool:
        pool.starmap(simulate_parameter_set, enumerate(input_data))

# define the numpy arrays to save
x = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf)
y = np.ndarray(Y.shape, dtype=Y.dtype, buffer=Y_shared.buf)

# mark end time
tfinal = time.time()
tstr = '{:3.2f}'.format(tfinal - tstart)
print(f"total simulation time = {tstr}")

# create folder associated with this simulation
folder = os.path.join(ROOT_DIR, "data", "simulated", "ongoing")
if not os.path.exists(folder):
   os.makedirs(folder)

# log simulation experiment information
config_path = os.path.join(folder, 'config_used.json')
with open(config_path, 'w') as fp:
    json.dump(param, fp, indent=4)

# save training data simulation information
output_path = os.path.join(folder, f"output_task{task_id}" '.pkl')
print('Saving updated training_data set...')   
with open(output_path, "wb") as f:
    pickle.dump([x, y], f)
print('Finished saving.')   

# close shared memory
utils.close_shared_memory(name=X_shared.name)
utils.close_shared_memory(name=Y_shared.name)
