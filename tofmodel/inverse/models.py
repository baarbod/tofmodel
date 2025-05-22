# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as torch 


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
    
class TOFinverse(nn.Module):
    
    def __init__(self, nfeature_in, nfeature_out, input_size, output_size):
        super(TOFinverse, self).__init__()
        
        self.down1 = (Down(nfeature_in, 16))
        self.down2 = (Down(16, 32))
        self.down3 = (Down(32, 64))
        self.up1 = (Up(64, 64))
        self.up2 = (Up(64, 32))
        self.up3 = (Up(32, 16))

        self.output_size = output_size
        self.nfeature_out = nfeature_out
        
        self.fc1 = nn.Linear(16 * input_size, 800)
        self.fc2 = nn.Linear(800, nfeature_out * output_size)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), self.nfeature_out, self.output_size) 
        return x

    