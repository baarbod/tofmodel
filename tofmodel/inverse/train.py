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


def train_net(dataset, noisedir=None, noise_method=None, epochs=40, batch=16, lr=0.00001, 
              gauss_low=0.01, gauss_high=0.1, noise_scale=0.5, exp_name='',
              patience=30, min_delta=1e-4, warmup_epochs=10):
    
    datasetdir = os.path.dirname(dataset)

    X_train, X_test, y_train, y_test = load_dataset(dataset, noisedir=noisedir, noise_method=noise_method, 
                                                    gauss_low=gauss_low, gauss_high=gauss_high, scalemax=noise_scale)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)
    
    # initialize the model and optimizer
    num_in_features = X_train.shape[1]
    input_size = X_train.shape[2]
    num_out_features = y_train.shape[1]
    output_size = y_train.shape[2]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running training using cuda GPU")
    else:
        device = torch.device("cpu")
        print(f"Running training using CPU")
    model = TOFinverse(num_in_features, num_out_features, 
                       input_size, output_size)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    print(model)
    
    # Prepare experiment folder
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    folder_root = os.path.join(datasetdir, "experiments", dataset_name)
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
        'dataset': dataset,
        'noisedir': noisedir
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


def load_dataset(dataset, noisedir=None, noise_method='none', gauss_low=0.01, gauss_high=0.1, scalemax=0.1):
    
    with open(dataset, "rb") as f:
        X, y, inputs = pickle.load(f)
        
    if noise_method == 'gaussian':
        print("Adding Gaussian noise...")
        X = noise.add_gaussian_noise(X, gauss_low=gauss_low, gauss_high=gauss_high)
        print("Gaussian noise injection complete.")

    elif noise_method == 'pca':
        if noisedir is None:
            raise ValueError("noisedir must be provided when using PCA-based noise.")

        path_to_noise_data = os.path.join(noisedir, 'noise_data.pkl')
        path_to_pca_model = os.path.join(noisedir, 'pca_model.pkl')

        print("Adding PCA-based noise...")

        if os.path.exists(path_to_pca_model):
            print(f"Loading existing PCA model from {path_to_pca_model}")
            model = noise.load_pca_model(path_to_pca_model)
        else:
            if not os.path.exists(path_to_noise_data):
                raise FileNotFoundError(
                    f"Noise data not found at {path_to_noise_data}. Cannot train PCA model."
                )
            print(f"No PCA model found — generating new model from {path_to_noise_data}")
            noise_data = noise.load_noise_data(path_to_noise_data)
            model = noise.define_pca_model(noise_data)
            noise.save_pca_model(model, path_to_pca_model)
            print(f"PCA model saved to {path_to_pca_model}")

        X = noise.add_pca_noise(X, model, scalemax=scalemax)
        print("PCA noise injection complete.")

    elif noise_method in [None, '', 'none']:
        print("No noise method specified — skipping noise injection.")

    else:
        raise ValueError(f"Unknown noise method: '{noise_method}'")
        
    nslice_to_use = inputs[0]['param'].nslice_to_use
    
    # scale signals 
    for i in range(X.shape[0]):
        to_scale = X[i, :nslice_to_use, :].T  # transpose to (time, channels)
        scaled = eval.scale_data(to_scale).T  # scale and transpose back
        X[i, :nslice_to_use, :] = scaled  # overwrite in-place
    
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

