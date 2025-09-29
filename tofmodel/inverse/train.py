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
import tofmodel.inverse.evaluation as eval
import json


def train_net(param, epochs=40, batch=16, lr=0.00001, noise_method=None, 
              gauss_low=0.01, gauss_high=0.1, noise_scale=0.5, exp_name='',
              patience=20, min_delta=1e-5, warmup_epochs=10):
    
    output_dir = param.output_dir
    dataset_name = param.dataset_name
    datasetdir = os.path.join(output_dir, dataset_name)

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
    
    # Prepare experiment folder
    folder_root = os.path.join(datasetdir, "experiments")
    formatted_datetime = utils.get_formatted_day_time()
    folder_name = f"{formatted_datetime}_training_run_{exp_name}"
    folder = os.path.join(folder_root, folder_name)
    os.makedirs(folder, exist_ok=True)
    checkpoint_path = os.path.join(folder, 'best_model.pth')

    # Save hyperparameters and config
    config = {
        'epochs': epochs,
        'batch': batch,
        'lr': lr,
        'noise_method': noise_method,
        'gauss_low': gauss_low,
        'gauss_high': gauss_high,
        'noise_scale': noise_scale,
        'patience': patience,
        'min_delta': min_delta,
        'warmup_epochs': warmup_epochs,
        'param_datasetdir': datasetdir
    }
    with open(os.path.join(folder, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    # training loop
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    patience_counter = 0
    time_start = time.time()
    
    for epoch in range(epochs):
        train_loss = train(train_loader, model, criterion, optimizer)
        test_loss = test(test_loader, model, criterion)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Early stopping check after warmup
        if epoch >= warmup_epochs:
            if test_loss < best_loss - min_delta:
                best_loss = test_loss
                patience_counter = 0
                # Save checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': test_loss
                }, checkpoint_path)
                print(f"Saved best model checkpoint at epoch {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        
    time_end = time.time()
    print(f"Total time for training loop: {time_end - time_start:.2f} seconds")

    history = {'train_losses': train_losses, 'test_losses': test_losses}
    np.savez(os.path.join(folder, 'training_history.npz'), **history)
    
    # plot loss curve
    fig, ax= plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(test_losses, label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid()
    ax.xaxis.set_ticks(np.arange(0, len(train_losses), max(1, len(train_losses)//10)))
    
    plot_path = os.path.join(folder, 'loss_vs_epochs.png')
    fig.savefig(plot_path)
    print(f"Saved loss plot to {plot_path}")
    
    print(f"Best validation loss: {best_loss}")


def load_dataset(param, noise_method='none', gauss_low=0.01, gauss_high=0.1, scalemax=0.5):
    
    output_dir = param.output_dir
    dataset_name = param.dataset_name
    datasetdir = os.path.join(output_dir, dataset_name)
    path_to_noise_data = os.path.join(output_dir, 'data', 'noise_data.pkl')
    path_to_pca_model = os.path.join(output_dir, 'data', 'pca_model.pkl')
    
    pkl_file = next(f for f in os.listdir(datasetdir) if f.endswith('.pkl'))
    with open(os.path.join(datasetdir, pkl_file), "rb") as f:
        X, y, _ = pickle.load(f)
    
    if noise_method == 'gaussian':
        print(f"Adding gaussian noise")
        X = noise.add_gaussian_noise(X, gauss_low=gauss_low, gauss_high=gauss_high)
        print('noise injection complete')
    elif noise_method == 'pca':
        print(f"Adding pca method noise")
        if os.path.exists(path_to_pca_model):
            print(f"Using saved PCA model: {path_to_pca_model}")
            model = noise.load_pca_model(path_to_pca_model)
        else:
            print(f"PCA model not found")
            print(f"Generating new model using noise data: {path_to_noise_data}")
            noise_data = noise.load_noise_data(path_to_noise_data)
            model = noise.define_pca_model(noise_data)
            print(f"Saving PCA model to: {path_to_pca_model}")
            noise.save_pca_model(model, path_to_pca_model)
        X = noise.add_pca_noise(X, model, scalemax=scalemax)
        print('noise injection complete')
    else:
        print(f"no noise being added")
    
    # scale signals 
    for i in range(X.shape[0]):
        to_scale = X[i, 0:3, :].T  # transpose to (time, channels)
        scaled = eval.scale_data(to_scale).T  # scale and transpose back
        X[i, 0:3, :] = scaled  # overwrite in-place
    
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

