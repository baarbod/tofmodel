# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pickle
import json
import argparse
import time 
from tofmodel.inverse.models import TOFinverse
import tofmodel.inverse.utils as utils
from tofmodel.path import ROOT_DIR


parser = argparse.ArgumentParser(description='Script for training neural network on simulated dataset')
parser.add_argument('--datafolder', help='path to folder containing simulated dataset and config file')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run (default: 10)')
parser.add_argument('--batch', default=16, type=int, help='batch size (default: 16)')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.1)')
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
    X, y, _ = pickle.load(f)
    
X = X.astype(float)
y = y.astype(float)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
param['training_data'] = training_data

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

# initialize the model and optimizer
model = TOFinverse(nfeatures=nfeatures, feature_size=X_train.shape[2],
                   output_size=output_size)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

def train(loader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    avg_loss = epoch_loss / len(loader.dataset)
    return avg_loss


def test(loader, model, criterion):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            epoch_loss += loss.item() * inputs.size(0)
    avg_loss = epoch_loss / len(loader.dataset)
    return avg_loss


# training loop
train_losses = []
test_losses = []
time_start = time.time()
for epoch in range(args.epochs):

    train_loss = train(train_loader, model, criterion, optimizer)
    test_loss = test(test_loader, model, criterion)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

time_end = time.time()
training_time = time_end - time_start
print(f"Total time for training loop: {training_time:.4f} seconds")

# plot loss vs. epoch for training and evaluation data
fig_loss, ax= plt.subplots()
ax.plot(train_losses, label='train data')
with torch.no_grad():
    ax.plot(np.array(test_losses), label='test data')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid()
ax.xaxis.set_ticks(np.arange(0, args.epochs, 4))

# evaluate model accuracy using test data
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    MSE = ((torch.pow((outputs - y_test[:, 0, :]), 2)).sum()) / outputs.numel()
    print(f'Mean Squared Error: {MSE:.4f}')

param['MSE'] = float(MSE)
# create folder associated with this simulation
folder_root = os.path.join(sim_dataset_path, "experiments")
formatted_datetime =  utils.get_formatted_day_time()
project_name = param['info']['name']
folder_name = f"{formatted_datetime}_training_run_{project_name}"
folder = os.path.join(folder_root, folder_name)
if not os.path.exists(folder):
   os.makedirs(folder)

# save plot of MSE loss vs. epochs
figpath = os.path.join(folder, 'MSE_loss_vs_epochs.png')
fig_loss.savefig(figpath, format='png')

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
