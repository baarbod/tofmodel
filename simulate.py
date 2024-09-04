# -*- coding: utf-8 -*-

from multiprocessing import Pool
import numpy as np
import json
import sys
import time
import pickle
import os

import tofinv.simulations as sim
import tofinv.utils as utils
import tofinv.sampling as vd

from config.path import ROOT_DIR

debug_mode = 0
trim_input_length = 0

if debug_mode:
    config_path = os.path.join(ROOT_DIR, "config", "config.json")
    with open(config_path, "r") as jsonfile:
            param = json.load(jsonfile)
    task_id = 1   
    nbatch = param['data_simulation']['num_batches']
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

# initialize parameters
input_data = []
Xtype = 'float64'
Ytype = 'float64'
nsample = param['data_simulation']['num_samples']
batch_size = int(nsample / nbatch)

# frequency parameters
fstart = param['data_simulation']['frequency_start']
fend = param['data_simulation']['frequency_end']
fspacing = param['data_simulation']['frequency_spacing']
frequencies = np.arange(fstart, fend, fspacing)
num_frequencies = np.size(frequencies)

# shape of input and output features
num_input = param['data_simulation']['num_input_features']
input_size = param['data_simulation']['input_feature_size']
num_output = param['data_simulation']['num_output_features']
output_size = param['data_simulation']['output_feature_size']
Xshape = (batch_size, num_input, input_size)
Yshape = (batch_size, num_output, output_size)

# set number of pulses based on desired length of spectrum
scan_param = param['scan_param']
scan_param["num_pulse"] = input_size
scan_param["num_pulse_baseline_offset"] = 20

sampling_param = param['sampling']
fresp_lower = sampling_param['fresp_lower']
fresp_upper = sampling_param['fresp_upper']
fcard_lower = sampling_param['fcard_lower']
fcard_upper = sampling_param['fcard_upper']
voffset_lower = sampling_param['voffset_lower']
voffset_upper = sampling_param['voffset_upper']

# random sampling constrained by lower and upper bounds
for isample in range(batch_size):
    mu = np.random.uniform(low=fresp_lower, high=fresp_upper)
    ff_card = np.random.uniform(low=fcard_lower, high=fcard_upper) 
    bound_array = vd.get_sampling_bounds(frequencies, mu, ff_card=ff_card)
    rand_numbers = np.zeros(np.size(frequencies))
    rand_phase = np.zeros(np.size(frequencies))
    for idx, frequency in enumerate(frequencies):
        lower = bound_array[idx, 0]
        upper = bound_array[idx, 1] 
        rand_numbers[idx] = np.random.uniform(low=lower, high=upper)
        rand_phase[idx] = np.random.uniform(low=0, high=1/frequency)
        v_offset = np.random.uniform(low=voffset_lower, high=voffset_upper)    
    slc_offset = np.random.uniform(low=-0.8, high=0.8)
    area_gauss_width = np.random.uniform(low=0.4, high=0.8)
    area_curve_fact = np.random.uniform(low=0.4, high=1)
    gauss_noise_std = np.random.uniform(low=0.01, high=0.1)
    input_data.append(tuple([tuple(frequencies), v_offset, rand_phase, tuple(rand_numbers), 
                             scan_param, Xshape, Xtype, Yshape, Ytype, task_id, 
                             slc_offset, area_gauss_width, area_curve_fact, gauss_noise_std]))

X = np.zeros(Xshape)
Y = np.zeros(Yshape)
X_shared = utils.create_shared_memory(X, name=f"Xarray{task_id}")
Y_shared = utils.create_shared_memory(Y, name=f"Yarray{task_id}")

if trim_input_length:
    input_data = input_data[:5]

# mark start time
t = time.time()

# call pool pointing to simulation routine
if __name__ == '__main__':
    with Pool(processes=param['data_simulation']['num_cores']) as pool:
        pool.starmap(sim.simulate_parameter_set, enumerate(input_data))

# define the numpy arrays to save
x = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf)
y = np.ndarray(Y.shape, dtype=Y.dtype, buffer=Y_shared.buf)

# mark end time
tfinal = time.time()
tstr = '{:3.2f}'.format(tfinal - t)
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
