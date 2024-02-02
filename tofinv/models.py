# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as torch 


# create class that inherits from Module
class TOFinverse(nn.Module):
    # init method where the layers are defined
    def __init__(self, nfeatures, feature_size, output_size):
        super(TOFinverse, self).__init__()
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
    