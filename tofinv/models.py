# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as torch 


# create class that inherits from Module
class TOFinverse(nn.Module):
    # init method where the layers are defined
    def __init__(self, nfeatures, feature_size, output_size):
        super(TOFinverse, self).__init__()
        
        self.down1 = nn.Conv1d(in_channels=nfeatures, out_channels=16, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(16)
        self.down2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(32)
        self.down3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(64)
        self.up1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm1d(32)
        self.up2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm1d(16)
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * feature_size, 800)
        self.fc2 = nn.Linear(800, output_size)

    def forward(self, x):
        x = self.down1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.down2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.down3(x)
        x = self.norm3(x)
        x = self.relu(x)        
        x = self.up1(x)
        x = self.norm4(x)
        x = self.relu(x)
        x = self.up2(x)
        x = self.norm5(x)
        x = self.relu(x)        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# create class that inherits from Module
class TOFinverse0(nn.Module):
    # init method where the layers are defined
    def __init__(self, nfeatures, feature_size, output_size):
        super(TOFinverse0, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=nfeatures, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * feature_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    