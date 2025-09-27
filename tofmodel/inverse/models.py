import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class TOFinverse(nn.Module):
    def __init__(self, nfeature_in, nfeature_out, input_size, output_size):
        super().__init__()

        self.down1 = Conv(nfeature_in, 16, dilation=1)
        self.down2 = Conv(16, 32, dilation=2)
        self.down3 = Conv(32, 64, dilation=4)
        self.up1 = Conv(64, 32, dilation=2)
        self.up2 = Conv(32, 16, dilation=1)
        self.out = nn.Conv1d(16, nfeature_out, kernel_size=1)

        self.output_size = output_size
        self.nfeature_out = nfeature_out

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.out(x)
        return x
