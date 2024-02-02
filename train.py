# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pickle
import json
import argparse

from tofinv.models import TOFinverse
import tofinv.utils as utils

from config.path import ROOT_DIR

parser = argparse.ArgumentParser(description='Script for training neural network on simulated dataset')
parser.add_argument('--datafolder', help='name of folder containing simulated dataset and config file')
args = parser.parse_args()

sim_dataset_path = os.path.join(ROOT_DIR, "data", "simulated", args.datafolder)
config_path = os.path.join(sim_dataset_path, "config_used.json")    
with open(config_path, "r") as jsonfile:
    param = json.load(jsonfile)

# find simulated dataset pickle file
for file in os.listdir(sim_dataset_path):
    if file.endswith(".pkl"):
            training_data_filename = file
            
# load training data set 
training_data = os.path.join(sim_dataset_path, training_data_filename)    
with open(training_data, "rb") as f:
    X, y = pickle.load(f)
    
X = X.astype(float)
y = y.astype(float)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=param['training']['test_size'])
param['training_data'] = training_data

# hyperparameters
batch_size = param['training']['batch_size']
learning_rate = param['training']['learning_rate']
num_epochs = param['training']['num_epochs']

# create an instance of the TOFinverse model
nfeatures = param['data_simulation']['num_input_features']
feature_size = X.shape[2]
output_size = y.shape[2]  # length of output velocity spectrum
model = TOFinverse(nfeatures, feature_size, output_size)
print(model)

# convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# create data loader objects
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# initialize the model and optimizer
model = TOFinverse(nfeatures=nfeatures, feature_size=X_train.shape[2],
                   output_size=output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# training loop
loss_epoch = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    loss_epoch.append(running_loss)

fig, ax= plt.subplots()
ax.plot(loss_epoch)
ax.set_xlabel('Epoch')
ax.set_ylabel('Running MSE loss')
print('finished Training')

# evaluate model accuracy using test data
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    MSE = ((torch.pow((outputs - y_test[:, 0, :]), 2)).sum()) / outputs.numel()
    print(f'Mean Squared Error: {MSE:.4f}')

param['MSE'] = float(MSE)
# create folder associated with this simulation
folder_root = os.path.join(ROOT_DIR, "experiments")
formatted_datetime =  utils.get_formatted_day_time()
project_name = param['info']['project']
folder_name = f"{formatted_datetime}_training_run_{project_name}"
folder = os.path.join(folder_root, folder_name)
if not os.path.exists(folder):
   os.makedirs(folder)

# save plot of MSE loss vs. epochs
figpath = os.path.join(folder, 'MSE_loss_vs_epochs.png')
fig.savefig(figpath, format='png')

# log config file used for training
config_path = os.path.join(folder, 'config_used.json')
with open(config_path, 'w') as fp:
    json.dump(param, fp, indent=4)

# save training information
num_samples = X.shape[0]
output_path = os.path.join(folder, f"model_state_{num_samples}_samples.pth")
print('Saving  training experiment...')   
torch.save(model.state_dict(), output_path)
print('Finished saving.')   
