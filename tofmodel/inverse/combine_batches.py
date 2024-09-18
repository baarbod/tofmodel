# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import tofmodel.inverse.utils as utils
import shutil

from config.path import ROOT_DIR

# create folder associated with this simulation
folder_root = os.path.join(ROOT_DIR, "data", "simulated")
formatted_datetime =  utils.get_formatted_day_time()
folder_name = f"{formatted_datetime}_dataset"
folder = os.path.join(folder_root, folder_name)
folder_log = os.path.join(folder, 'logs')
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder_log)
    
training_data_folder = os.path.abspath(os.path.join(__file__, "..", 'data', 'simulated', 'ongoing'))
x_list = []
y_list = []
config_flag = 0
for file in os.listdir(training_data_folder):
    if file.endswith('.pkl'):
        filepath = os.path.join(training_data_folder, file)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                x, y = pickle.load(f)
            x_list.append(x)
            y_list.append(y)
            print(f"combined and removed {file}")
            os.remove(filepath)
            # print(f"added, but NOT removed {file}")
    if file.endswith('_used.json') and not config_flag:
        filepath = os.path.join(training_data_folder, file)
        shutil.move(filepath, folder)
        # shutil.copy(filepath, folder)
        config_flag = 1
    if file.endswith('.txt'):
        filepath = os.path.join(training_data_folder, file)
        shutil.move(filepath, folder_log)
        # shutil.copy(filepath, folder)

x = np.concatenate(x_list, axis=0)    
y = np.concatenate(y_list, axis=0)

# save training data simulation information
filename = f"output_{y.shape[0]}_samples.pkl"
filepath = os.path.join(folder, filename)
print('Saving updated training_data set...')   
with open(filepath, "wb") as f:
    pickle.dump([x, y], f)
print('Finished saving.')   

