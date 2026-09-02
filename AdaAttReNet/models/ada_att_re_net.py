import torch
import torch.nn as nn

from .att_unet import AttUNet
from .mask_net import MaskNet
from .pca_unet import PCAUNet


class _OutputFusionConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x + residual


class AdaAttReNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.att_unet = AttUNet(in_channels, out_channels)
        self.pca_unet = PCAUNet(in_channels, out_channels)
        self.mask_net = MaskNet(in_channels, out_channels=1)
        self.output_fusion = _OutputFusionConv(in_channels, out_channels)

    def forward(self, x):
        att_output = self.att_unet(x)
        pca_output = self.pca_unet(x)
        mask = self.mask_net(x)
        combined_output = att_output * mask + pca_output * (1 - mask)
        return self.output_fusion(combined_output)
