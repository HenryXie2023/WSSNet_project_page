import torch
import torch.nn as nn


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class CASAtt(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.0):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.oper_k = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        return self.proj_drop(self.proj(self.dwc(q + k) * v))


class HPA(nn.Module):
    def __init__(self, channels, c2=None, factor=8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.max_w = nn.AdaptiveMaxPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        y_h = self.max_h(group_x)
        y_w = self.max_w(group_x).permute(0, 1, 3, 2)
        yhw = self.conv1x1(torch.cat([y_h, y_w], dim=2))
        y_h, y_w = torch.split(yhw, [h, w], dim=2)
        y1 = self.gn(group_x * y_h.sigmoid() * y_w.permute(0, 1, 3, 2).sigmoid())

        y11 = y1.reshape(b * self.groups, c // self.groups, -1)
        y12 = self.softmax(self.map(y1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x11 = x1.reshape(b * self.groups, c // self.groups, -1)
        x12 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        weights = (torch.matmul(x12, y11) + torch.matmul(y12, x11)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class MSSAtt(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=7, padding=3, bias=False)
        self.hpa = HPA(channels=in_channels // 2)
        self.casatt = CASAtt(dim=in_channels // 8)

    def forward(self, x):
        c = x.size(1)
        x1 = x[:, : c // 8, :, :]
        x2 = x[:, c // 8 : c // 4, :, :]
        x3 = x[:, c // 4 : 3 * c // 8, :, :]
        x4 = x[:, 3 * c // 8 : 4 * c // 8, :, :]
        x5 = x[:, 4 * c // 8 :, :, :]
        out1 = self.casatt(self.conv3x3(x1))
        out2 = self.casatt(self.conv5x5(x2))
        out3 = self.casatt(self.conv7x7(x3))
        out5 = self.hpa(x5)
        return torch.cat([out1, out2, out3, x4, out5], dim=1)
