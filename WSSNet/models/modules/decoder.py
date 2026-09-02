import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_space import StateSpaceBlock


class MEMCAUMaskAdapter(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(inc, outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU(),
        )
        self.predtrans = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(outc, 1, kernel_size=1),
        )
        self.rc2 = nn.Sequential(
            nn.Conv2d(outc * 2, outc * 2, kernel_size=3, padding=1, groups=outc * 2),
            nn.BatchNorm2d(outc * 2),
            nn.GELU(),
            nn.Conv2d(outc * 2, outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU(),
        )
        self.mask_conv = nn.Conv2d(outc, 1, kernel_size=1)

    def forward(self, x1, x2):
        x2_rc = self.rc(self.upsample2(x2))
        x_forward = self.rc2(torch.cat((x1, x2_rc), dim=1))
        x_forward = x_forward + x2_rc
        mask = self.mask_conv(x_forward)
        return mask, x_forward


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MEMCAUBridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=None):
        super().__init__()
        if d_list is None:
            d_list = [1, 2, 5, 7]
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(group_size + 1, data_format="channels_first"),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size + 1),
        )
        self.g1 = nn.Sequential(
            LayerNorm(group_size + 1, data_format="channels_first"),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size + 1),
        )
        self.g2 = nn.Sequential(
            LayerNorm(group_size + 1, data_format="channels_first"),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size + 1),
        )
        self.g3 = nn.Sequential(
            LayerNorm(group_size + 1, data_format="channels_first"),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size + 1),
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(dim_xl * 2 + 4, data_format="channels_first"),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1),
        )
        self.mamba = StateSpaceBlock(dim=dim_xh)

    def forward(self, xh, xl, mask):
        xh = self.pre_project(self.mamba(xh))
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode="bilinear", align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        return self.tail_conv(torch.cat((x0, x1, x2, x3), dim=1))


class MEMCAU(nn.Module):
    def __init__(self, up_in, up_out, mask_in, mask_out, bridge_high, bridge_low):
        super().__init__()
        from .blocks import MEMCAUUpBlock

        self.mask_adapter = MEMCAUMaskAdapter(mask_in, mask_out)
        self.bridge = MEMCAUBridge(bridge_high, bridge_low)
        self.up = MEMCAUUpBlock(up_in, up_out)

    def forward(self, low_feature, high_feature, up_reference):
        mask, adapted = self.mask_adapter(low_feature, up_reference)
        fused = adapted + self.bridge(high_feature, low_feature, mask)
        return self.up(fused, up_reference)
