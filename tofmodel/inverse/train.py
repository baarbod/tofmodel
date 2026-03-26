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
from tofinv.utils import scale_data, scale_area

class SurrogateConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding='same')
        )

    def forward(self, x):
        return self.network(x)

def load_dataset(dataset, noisedir=None, noise_method='none', gauss_low=0.01, gauss_high=0.1, scalemax=0.1, nslice_to_use=3):
    with open(dataset, "rb") as f:
        X, y = pickle.load(f)
    
    nslice_to_use = X.shape[1] - 2
    pos_idx = nslice_to_use
    area_idx = nslice_to_use + 1
    
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
            model = noise.load_pca_model(path_to_pca_model)
        else:
            if not os.path.exists(path_to_noise_data):
                raise FileNotFoundError("Noise data not found.")
            noise_data = noise.load_noise_data(path_to_noise_data)
            model = noise.define_pca_model(noise_data)
            noise.save_pca_model(model, path_to_pca_model)
        X = noise.add_pca_noise(X, model, scalemax=scalemax)

    # Scale signals and area
    for i in range(X.shape[0]):
        to_scale = X[i, :nslice_to_use, :].T 
        scaled = scale_data(to_scale).T
        X[i, :nslice_to_use, :] = scaled 
        
        xarea_single = X[i, pos_idx, :]
        area_single = X[i, area_idx, :]
        X[i, area_idx, :] = scale_area(xarea_single, area_single)
        
    # Split the inputs
    flow_x = X[:, :nslice_to_use, :]
    area_x = X[:, area_idx : area_idx + 1, :]
    
    indices = np.arange(X.shape[0])
    idx_train, idx_test = train_test_split(indices, test_size=0.1)

    flow_train, flow_test = flow_x[idx_train], flow_x[idx_test]
    area_train, area_test = area_x[idx_train], area_x[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    if torch.cuda.is_available():
        flow_train = torch.tensor(flow_train, dtype=torch.float32).cuda()
        area_train = torch.tensor(area_train, dtype=torch.float32).cuda()
        y_train = torch.tensor(y_train, dtype=torch.float32).cuda()
        flow_test = torch.tensor(flow_test, dtype=torch.float32).cuda()
        area_test = torch.tensor(area_test, dtype=torch.float32).cuda()
        y_test = torch.tensor(y_test, dtype=torch.float32).cuda()
    else:
        flow_train = torch.tensor(flow_train, dtype=torch.float32)
        area_train = torch.tensor(area_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        flow_test = torch.tensor(flow_test, dtype=torch.float32)
        area_test = torch.tensor(area_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    return flow_train, area_train, flow_test, area_test, y_train, y_test

def train(loader, model, surrogate_model, criterion, optimizer, lambda_phys):
    model.train()
    epoch_loss, epoch_v_loss, epoch_phys_loss = 0.0, 0.0, 0.0
    
    for flow_inputs, area_inputs, labels in loader:
        optimizer.zero_grad()
        
        # Predict Velocity
        pred_v = model(flow_inputs, area_inputs) 
        loss_v = criterion(pred_v, labels) 
        
        # Physics Loss
        if lambda_phys > 0:
            surrogate_in = torch.cat([pred_v, area_inputs], dim=1)
            pred_signal = surrogate_model(surrogate_in)
            loss_physics = criterion(pred_signal, flow_inputs)
            total_loss = loss_v + (lambda_phys * loss_physics)
        else:
            loss_physics = torch.tensor(0.0)
            total_loss = loss_v

        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item() * flow_inputs.size(0)
        epoch_v_loss += loss_v.item() * flow_inputs.size(0)
        epoch_phys_loss += loss_physics.item() * flow_inputs.size(0)
        
    n = len(loader.dataset)
    return epoch_loss / n, epoch_v_loss / n, epoch_phys_loss / n

def test(loader, model, surrogate_model, criterion, lambda_phys):
    model.eval()
    epoch_loss, epoch_v_loss, epoch_phys_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for flow_inputs, area_inputs, labels in loader:
            pred_v = model(flow_inputs, area_inputs)
            loss_v = criterion(pred_v, labels)
            
            if lambda_phys > 0:
                surrogate_in = torch.cat([pred_v, area_inputs], dim=1)
                pred_signal = surrogate_model(surrogate_in)
                loss_physics = criterion(pred_signal, flow_inputs)
                total_loss = loss_v + (lambda_phys * loss_physics)
            else:
                loss_physics = torch.tensor(0.0)
                total_loss = loss_v
                
            epoch_loss += total_loss.item() * flow_inputs.size(0)
            epoch_v_loss += loss_v.item() * flow_inputs.size(0)
            epoch_phys_loss += loss_physics.item() * flow_inputs.size(0)
            
    n = len(loader.dataset)
    return epoch_loss / n, epoch_v_loss / n, epoch_phys_loss / n

def train_net(dataset, noisedir=None, noise_method=None, epochs=40, batch=128, lr=0.001, 
              gauss_low=0.01, gauss_high=0.1, noise_scale=0.5, exp_name='',
              patience=30, min_delta=1e-4, warmup_epochs=10, 
              lambda_phys=1.0, surrogate_path='surrogate_model_weights.pth'):
    
    # #override
    # batch = 128
    # lr = 0.001
    
    
    datasetdir = os.path.dirname(dataset)

    flow_train, area_train, flow_test, area_test, y_train, y_test = load_dataset(
        dataset, noisedir=noisedir, noise_method=noise_method, 
        gauss_low=gauss_low, gauss_high=gauss_high, scalemax=noise_scale
    )
    
    # Update to 3-tensor datasets
    train_dataset = torch.utils.data.TensorDataset(flow_train, area_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(flow_test, area_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)
    
    nflow_in = flow_train.shape[1]
    num_out_features = y_train.shape[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training using {device}")
    
    # Initialize split model
    model = TOFinverse(nflow_in=nflow_in, nfeature_out=num_out_features)
    model.to(device)
    
    # --- ADDED SCHEDULER INITIALIZATION ---
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # factor=0.5: cut LR in half when plateauing
    # patience=8: wait 8 epochs of no improvement before dropping LR
    # min_lr: never drop below this value
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )
    # --------------------------------------

    criterion = nn.MSELoss()
    
    # Load Surrogate (1 V + 1 Area = 2 in_channels)
    print(f"Loading surrogate model from {surrogate_path}...")
    surrogate_model = SurrogateConv1D(in_channels=2, out_channels=nflow_in)
    surrogate_model.load_state_dict(torch.load(surrogate_path, map_location=device, weights_only=True))
    surrogate_model.to(device)
    surrogate_model.eval()
    for param in surrogate_model.parameters():
        param.requires_grad = False
    print("Surrogate model loaded and frozen.")
    
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    # folder_root = os.path.join(datasetdir, "experiments", dataset_name)
    folder_root = '/orcd/data/ldlewis/001/om/bashen/repositories/tofinv/output/4_experiments'
    folder = os.path.join(folder_root, exp_name)
    os.makedirs(folder, exist_ok=True)
    checkpoint_path = os.path.join(folder, 'best_model.pth')

    config = {
        'epochs': epochs, 'batch': batch, 'lr': lr, 'noise_method': noise_method,
        'lambda_phys': lambda_phys, 'dataset': dataset
    }
    with open(os.path.join(folder, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    train_losses, test_losses = [], []
    best_loss = float('inf')
    patience_counter = 0
    time_start = time.time()
    
    for epoch in range(epochs):
        train_loss, train_v, train_phys = train(train_loader, model, surrogate_model, criterion, optimizer, lambda_phys)
        test_loss, test_v, test_phys = test(test_loader, model, surrogate_model, criterion, lambda_phys)
        
        # --- ADDED SCHEDULER STEP ---
        # Step the scheduler based on validation (test) loss
        scheduler.step(test_loss)
        # ----------------------------

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Fetch current learning rate for logging
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} (V: {train_v:.4f}, Phys: {train_phys:.4f}) | Test Loss: {test_loss:.4f} (V: {test_v:.4f}, Phys: {test_phys:.4f})')
        
        if epoch >= warmup_epochs:
            if test_loss < best_loss - min_delta:
                best_loss = test_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': test_loss
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
    time_end = time.time()
    print(f"Total training time: {time_end - time_start:.2f} seconds")

    history = {'train_losses': train_losses, 'test_losses': test_losses}
    np.savez(os.path.join(folder, 'training_history.npz'), **history)
    
    fig, ax= plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(test_losses, label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    
    plot_path = os.path.join(folder, 'loss_vs_epochs.png')
    fig.savefig(plot_path)
    print(f"Saved loss plot to {plot_path}")
    
    

# # -*- coding: utf-8 -*-

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import os
# import pickle
# import time 
# from tofmodel.inverse.models import TOFinverse
# import tofmodel.inverse.utils as utils
# import tofmodel.inverse.noise as noise
# import tofmodel.inverse.evaluation as eval
# import json

# # ### NEW: Step 1 - Define the Surrogate Architecture so we can load it ###
# class SurrogateConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_dim=64):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding='same'),
#             nn.BatchNorm1d(hidden_dim),
#             nn.GELU(),
#             nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding='same'),
#             nn.BatchNorm1d(hidden_dim * 2),
#             nn.GELU(),
#             nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding='same'),
#             nn.BatchNorm1d(hidden_dim),
#             nn.GELU(),
#             nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding='same')
#         )

#     def forward(self, x):
#         return self.network(x)

# def train_net(dataset, noisedir=None, noise_method=None, epochs=40, batch=16, lr=0.00001, 
#               gauss_low=0.01, gauss_high=0.1, noise_scale=0.5, exp_name='',
#               patience=30, min_delta=1e-4, warmup_epochs=10, 
#               lambda_phys=1.0, surrogate_path='surrogate_model_weights.pth'):
    
#     datasetdir = os.path.dirname(dataset)

#     X_train, X_test, y_train, y_test = load_dataset(dataset, noisedir=noisedir, noise_method=noise_method, 
#                                                     gauss_low=gauss_low, gauss_high=gauss_high, scalemax=noise_scale)
#     train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
#     test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)
    
#     # initialize the model and optimizer
#     num_in_features = X_train.shape[1]
#     input_size = X_train.shape[2]
#     num_out_features = y_train.shape[1]
#     output_size = y_train.shape[2]
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print(f"Running training using cuda GPU")
#     else:
#         device = torch.device("cpu")
#         print(f"Running training using CPU")
#     model = TOFinverse(num_in_features, num_out_features, 
#                        input_size, output_size)
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#     print(model)
    
#     # ### NEW: Step 2 - Load and Freeze the Surrogate Model ###
#     print(f"Loading surrogate model from {surrogate_path}...")
#     # Number of signal channels (to dynamically configure surrogate outputs)
#     nslice_to_use = num_in_features - 2 
#     # Surrogate inputs: V (1 channel) + A (2 channels)
#     surrogate_model = SurrogateConv1D(in_channels=1+2, out_channels=nslice_to_use)
#     surrogate_model.load_state_dict(torch.load(surrogate_path, map_location=device, weights_only=True))
#     surrogate_model.to(device)
#     surrogate_model.eval() # MUST be in eval mode
#     for param in surrogate_model.parameters():
#         param.requires_grad = False # Freeze weights completely
#     print("Surrogate model loaded and frozen.")
#     # #########################################################
    
    
#     # Prepare experiment folder
#     dataset_name = os.path.splitext(os.path.basename(dataset))[0]
#     folder_root = os.path.join(datasetdir, "experiments", dataset_name)
#     # formatted_datetime = utils.get_formatted_day_time()
#     # folder_name = f"{formatted_datetime}_training_run_{exp_name}"
#     # folder = os.path.join(folder_root, folder_name)
#     folder = os.path.join(folder_root, exp_name)
#     os.makedirs(folder, exist_ok=True)
#     checkpoint_path = os.path.join(folder, 'best_model.pth')

#     # Save hyperparameters and config
#     config = {
#         'epochs': epochs,
#         'batch': batch,
#         'lr': lr,
#         'noise_method': noise_method,
#         'gauss_low': gauss_low,
#         'gauss_high': gauss_high,
#         'noise_scale': noise_scale,
#         'patience': patience,
#         'min_delta': min_delta,
#         'warmup_epochs': warmup_epochs,
#         'dataset': dataset,
#         'noisedir': noisedir
#     }
#     with open(os.path.join(folder, 'training_config.json'), 'w') as f:
#         json.dump(config, f, indent=4)
        
#     # training loop
#     train_losses = []
#     test_losses = []
#     best_loss = float('inf')
#     patience_counter = 0
#     time_start = time.time()
    
#     for epoch in range(epochs):
#         train_loss = train(train_loader, model, surrogate_model, criterion, optimizer, lambda_phys)
#         test_loss = test(test_loader, model, surrogate_model, criterion, lambda_phys)
        
#         train_losses.append(train_loss)
#         test_losses.append(test_loss)
#         print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
#         # Early stopping check after warmup
#         if epoch >= warmup_epochs:
#             if test_loss < best_loss - min_delta:
#                 best_loss = test_loss
#                 patience_counter = 0
#                 # Save checkpoint
#                 torch.save({
#                     'epoch': epoch + 1,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'train_loss': train_loss,
#                     'val_loss': test_loss
#                 }, checkpoint_path)
#                 print(f"Saved best model checkpoint at epoch {epoch+1}")
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     print(f"Early stopping at epoch {epoch+1}")
#                     break
        
        
#     time_end = time.time()
#     print(f"Total time for training loop: {time_end - time_start:.2f} seconds")

#     history = {'train_losses': train_losses, 'test_losses': test_losses}
#     np.savez(os.path.join(folder, 'training_history.npz'), **history)
    
#     # plot loss curve
#     fig, ax= plt.subplots()
#     ax.plot(train_losses, label='Train Loss')
#     ax.plot(test_losses, label='Test Loss')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     ax.set_yscale('log')
#     ax.legend()
#     ax.grid()
#     ax.xaxis.set_ticks(np.arange(0, len(train_losses), max(1, len(train_losses)//10)))
    
#     plot_path = os.path.join(folder, 'loss_vs_epochs.png')
#     fig.savefig(plot_path)
#     print(f"Saved loss plot to {plot_path}")
    
#     print(f"Best validation loss: {best_loss}")


# def load_dataset(dataset, noisedir=None, noise_method='none', gauss_low=0.01, gauss_high=0.1, scalemax=0.1, nslice_to_use=3):
    
#     with open(dataset, "rb") as f:
#         X, y = pickle.load(f)
    
#     nslice_to_use = X.shape[1] - 2
    
#     # dataset_dir = os.path.dirname(dataset)
    
#     # # 1. Load the metadata to get shapes
#     # meta_path = os.path.join(dataset_dir, "dataset_info.pkl")
#     # with open(meta_path, "rb") as f:
#     #     meta = pickle.load(f)

#     # # 2. Map the binary files
#     # X_mmap = np.memmap(
#     #     os.path.join(dataset_dir, "X.bin"), 
#     #     dtype='float32', 
#     #     mode='r', 
#     #     shape=meta['x_shape']
#     # )

#     # y_mmap = np.memmap(
#     #     os.path.join(dataset_dir, "y.bin"), 
#     #     dtype='float32', 
#     #     mode='r', 
#     #     shape=meta['y_shape']
#     # )
    
#     # # 2. Force load into actual RAM so we can modify it
#     # print("Loading dataset into RAM...")
#     # X = np.array(X_mmap)
#     # y = np.array(y_mmap)
    
#     if noise_method == 'gaussian':
#         print("Adding Gaussian noise...")
#         X = noise.add_gaussian_noise(X, gauss_low=gauss_low, gauss_high=gauss_high)
#         print("Gaussian noise injection complete.")

#     elif noise_method == 'pca':
#         if noisedir is None:
#             raise ValueError("noisedir must be provided when using PCA-based noise.")

#         path_to_noise_data = os.path.join(noisedir, 'noise_data.pkl')
#         path_to_pca_model = os.path.join(noisedir, 'pca_model.pkl')

#         print("Adding PCA-based noise...")

#         if os.path.exists(path_to_pca_model):
#             print(f"Loading existing PCA model from {path_to_pca_model}")
#             model = noise.load_pca_model(path_to_pca_model)
#         else:
#             if not os.path.exists(path_to_noise_data):
#                 raise FileNotFoundError(
#                     f"Noise data not found at {path_to_noise_data}. Cannot train PCA model."
#                 )
#             print(f"No PCA model found — generating new model from {path_to_noise_data}")
#             noise_data = noise.load_noise_data(path_to_noise_data)
#             model = noise.define_pca_model(noise_data)
#             noise.save_pca_model(model, path_to_pca_model)
#             print(f"PCA model saved to {path_to_pca_model}")

#         X = noise.add_pca_noise(X, model, scalemax=scalemax)
#         print("PCA noise injection complete.")

#     elif noise_method in [None, '', 'none']:
#         print("No noise method specified — skipping noise injection.")

#     else:
#         raise ValueError(f"Unknown noise method: '{noise_method}'")
    

#     def scale_sim_baseline(s, baseline):
#         return s/baseline
    
#     from omegaconf import OmegaConf
#     param = OmegaConf.load('/orcd/data/ldlewis/001/om/bashen/repositories/tofinv/config/config.yml')
#     sin_fa = np.sin(param.scan_param.flip_angle * np.pi / 180)
#     cos_fa = np.cos(param.scan_param.flip_angle * np.pi / 180)
#     exp_tr_t1 = np.exp(-param.scan_param.repetition_time / param.scan_param.t1_time)
#     exp_te_t2 = np.exp(-param.scan_param.echo_time / param.scan_param.t2_time)
#     mT_ss =  sin_fa * exp_te_t2 * (1 - exp_tr_t1) / (1 - exp_tr_t1 * cos_fa)

#     # mT_ss = 0.127#0.3

#     # scale signals 
    
#     from tofinv.utils import scale_data
#     for i in range(X.shape[0]):
#         to_scale = X[i, :nslice_to_use, :].T  # transpose to (time, channels)
#         # scaled = eval.scale_data(to_scale).T  # scale and transpose back
    
#         scaled = scale_data(to_scale).T
    
#         # scaled = scale_sim_baseline(to_scale, mT_ss).T
#         # scaled = to_scale.copy()
#         # scaled[:, 0] = scaled[:, 0] / 0.3
#         # scaled[:, 1:] = scaled[:, 1:] / 0.4
        
#         X[i, :nslice_to_use, :] = scaled  # overwrite in-place
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#     if torch.cuda.is_available():
#         X_train = torch.tensor(X_train, dtype=torch.float32).cuda()
#         y_train = torch.tensor(y_train, dtype=torch.float32).cuda()
#         X_test = torch.tensor(X_test, dtype=torch.float32).cuda()
#         y_test = torch.tensor(y_test, dtype=torch.float32).cuda()
#     else:
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         y_train = torch.tensor(y_train, dtype=torch.float32)
#         X_test = torch.tensor(X_test, dtype=torch.float32)
#         y_test = torch.tensor(y_test, dtype=torch.float32)

#     return X_train, X_test, y_train, y_test

# # ### NEW: Step 3 - Modify the Loss calculation to include the Surrogate ###
# def train(loader, model, surrogate_model, criterion, optimizer, lambda_phys):
#     model.train()
#     epoch_loss = 0.0
#     for inputs, labels in loader:
#         optimizer.zero_grad()
        
#         # 1. Inverse Model Predicts Velocity
#         pred_v = model(inputs) 
        
#         # Supervised Loss (V_pred vs V_true)
#         # Note: I removed .squeeze() to ensure channel dimensions don't get lost, 
#         # which can break the concatenated surrogate input below.
#         loss_v = criterion(pred_v, labels) 
        
#         # 2. Physics-Informed Loss via Surrogate
#         if lambda_phys > 0:
#             # Reconstruct inputs for the Surrogate: Predicted Velocity + True Area
#             # Inputs shape is (Batch, Slices+Area, Seq)
#             num_signal_channels = inputs.shape[1] - 2 
            
#             true_signal = inputs[:, :num_signal_channels, :]
#             area_context = inputs[:, num_signal_channels:, :]
            
#             # Concat Predicted V and Area along the channel dimension
#             surrogate_in = torch.cat([pred_v, area_context], dim=1)
            
#             # Predict the signal using the frozen surrogate
#             pred_signal = surrogate_model(surrogate_in)
            
#             # Physics Loss (S_pred vs S_true)
#             loss_physics = criterion(pred_signal, true_signal)
            
#             # Total Loss
#             total_loss = loss_v + (lambda_phys * loss_physics)
#         else:
#             total_loss = loss_v

#         total_loss.backward()
#         optimizer.step()
#         epoch_loss += total_loss.item() * inputs.size(0)
        
#     avg_loss = epoch_loss / len(loader.dataset)
#     return avg_loss


# def test(loader, model, surrogate_model, criterion, lambda_phys):
#     model.eval()
#     epoch_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in loader:
#             pred_v = model(inputs)
#             loss_v = criterion(pred_v, labels)
            
#             # Evaluate Physics loss just like in training
#             if lambda_phys > 0:
#                 num_signal_channels = inputs.shape[1] - 2
#                 true_signal = inputs[:, :num_signal_channels, :]
#                 area_context = inputs[:, num_signal_channels:, :]
                
#                 surrogate_in = torch.cat([pred_v, area_context], dim=1)
#                 pred_signal = surrogate_model(surrogate_in)
#                 loss_physics = criterion(pred_signal, true_signal)
                
#                 total_loss = loss_v + (lambda_phys * loss_physics)
#             else:
#                 total_loss = loss_v
                
#             epoch_loss += total_loss.item() * inputs.size(0)
            
#     avg_loss = epoch_loss / len(loader.dataset)
#     return avg_loss

# # def train(loader, model, criterion, optimizer):
# #     model.train()
# #     epoch_loss = 0.0
# #     for inputs, labels in loader:
# #         optimizer.zero_grad()
# #         outputs = model(inputs)
# #         loss = criterion(outputs.squeeze(), labels.squeeze())
# #         loss.backward()
# #         optimizer.step()
# #         epoch_loss += loss.item() * inputs.size(0)
# #     avg_loss = epoch_loss / len(loader.dataset)
# #     return avg_loss


# # def test(loader, model, criterion):
# #     model.eval()
# #     epoch_loss = 0.0
# #     with torch.no_grad():
# #         for inputs, labels in loader:
# #             outputs = model(inputs)
# #             loss = criterion(outputs.squeeze(), labels.squeeze())
# #             epoch_loss += loss.item() * inputs.size(0)
# #     avg_loss = epoch_loss / len(loader.dataset)
# #     return avg_loss

