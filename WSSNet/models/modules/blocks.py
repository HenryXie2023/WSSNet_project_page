import torch
import torch.nn as nn


class MEMCAUUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        y = torch.cat([inputs1, self.up(inputs2)], 1)
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        return y


class EncoderDownBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        y = self.relu(self.conv1(inputs))
        y = self.relu(self.conv2(y))
        return self.pool(y)
