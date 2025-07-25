# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pickle
import time 
from tofmodel.inverse.models import TOFinverse
import tofmodel.inverse.utils as utils
import tofmodel.inverse.noise as noise


def train_net(param, epochs=40, batch=16, lr=0.00001, noise_method=None, 
              gauss_low=0.01, gauss_high=0.1, noise_scale=1.5, exp_name=''):
    
    X_train, X_test, y_train, y_test = load_dataset(param, noise_method=noise_method, 
                                                    gauss_low=gauss_low, gauss_high=gauss_high, scalemax=noise_scale)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # initialize the model and optimizer
    input_size = X_train.shape[2]
    output_size = y_train.shape[2]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running training using cuda GPU")
    else:
        device = torch.device("cpu")
        print(f"Running training using CPU")
    model = TOFinverse(param.data_simulation.num_input_features, param.data_simulation.num_output_features, 
                       input_size, output_size)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    print(model)

    # training loop
    train_losses = []
    test_losses = []
    time_start = time.time()
    
    for epoch in range(epochs):
        train_loss = train(train_loader, model, criterion, optimizer)
        test_loss = test(test_loader, model, criterion)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

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
    ax.xaxis.set_ticks(np.arange(0, epochs, 4))

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        MSE = torch.mean((outputs - y_test) ** 2)
        print(f'Mean Squared Error: {MSE:.4f}')

    # create folder associated with this simulation
    folder_root = os.path.join(param.paths.datasetdir, "experiments")
    formatted_datetime =  utils.get_formatted_day_time()
    exp_name = '_' + exp_name
    folder_name = f"{formatted_datetime}_training_run{exp_name}"
    folder = os.path.join(folder_root, folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save plot of MSE loss vs. epochs
    figpath = os.path.join(folder, 'MSE_loss_vs_epochs.png')
    fig_loss.savefig(figpath, format='png')

    # save training information
    num_samples = X_train.shape[0]
    output_path = os.path.join(folder, f"model_state_{num_samples}_samples.pth")
    print('Saving  training experiment...')   
    torch.save(model.state_dict(), output_path)
    print('Finished saving.')   


def load_dataset(param, noise_method='none', gauss_low=0.01, gauss_high=0.1, scalemax=1.5):
    pkl_file = next(f for f in os.listdir(param.paths.datasetdir) if f.endswith('.pkl'))
    with open(os.path.join(param.paths.datasetdir, pkl_file), "rb") as f:
        X, y, _ = pickle.load(f)
    
    nan_ind = ~np.isnan(X).any(axis=-1).any(axis=-1)
    X = X[nan_ind]
    y = y[nan_ind]
    
    inf_ind = ~np.isinf(X).any(axis=-1).any(axis=-1)
    X = X[inf_ind]
    y = y[inf_ind]
    
    if noise_method == 'gaussian':
        print(f"Adding gaussian noise")
        X = noise.add_gaussian_noise(X, gauss_low=gauss_low, gauss_high=gauss_high)
        print('noise injection complete')
    elif noise_method == 'pca':
        print(f"Adding pca method noise")
        if os.path.exists(param.paths.path_to_pca_model):
            print(f"Using saved PCA model: {param.paths.path_to_pca_model}")
            model = noise.load_pca_model(param.paths.path_to_pca_model)
        else:
            print(f"PCA model not found")
            print(f"Generating new model using noise data: {param.paths.path_to_noise_data}")
            noise_data = noise.load_noise_data(param.paths.path_to_noise_data)
            model = noise.define_pca_model(noise_data)
            print(f"Saving PCA model to: {param.paths.path_to_pca_model}")
            noise.save_pca_model(model, param.paths.path_to_pca_model)
        X = noise.add_pca_noise(X, model, scalemax=scalemax)
        print('noise injection complete')
    else:
        print(f"no noise being added")
    
    # zscore slice signals relative to first slice
    ref = X[:, 0, :]  
    mean_ref = np.mean(ref, axis=1, keepdims=True)  
    std_ref = np.std(ref, axis=1, keepdims=True)    
    X[:, 0:3, :] = (X[:, 0:3, :] - mean_ref[:, None, :]) / std_ref[:, None, :]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    if torch.cuda.is_available():
        X_train = torch.tensor(X_train, dtype=torch.float32).cuda()
        y_train = torch.tensor(y_train, dtype=torch.float32).cuda()
        X_test = torch.tensor(X_test, dtype=torch.float32).cuda()
        y_test = torch.tensor(y_test, dtype=torch.float32).cuda()
    else:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test


def train(loader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.squeeze())
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
            loss = criterion(outputs.squeeze(), labels.squeeze())
            epoch_loss += loss.item() * inputs.size(0)
    avg_loss = epoch_loss / len(loader.dataset)
    return avg_loss

