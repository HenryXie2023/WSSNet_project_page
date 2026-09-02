import torch
import torch.nn as nn

class Attention(nn.Module):


    def __init__(self, in_channels, reduction=16):
        super(Attention, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=True)
        )
        # 全局空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ---------- Channel Attention ----------
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1)
        x = x * ca  # 通道加权

        # ---------- Global Spatial Attention ----------
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid(self.spatial_conv(sa))
        x = x * sa  # 空间加权

        return x

