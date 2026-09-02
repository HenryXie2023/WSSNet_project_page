import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward as _DWTForward
except Exception:
    _DWTForward = None


class HaarDWTForward(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        yl = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        yh = [torch.stack([lh, hl, hh], dim=2)]
        return yl, yh


def _make_dwt():
    if _DWTForward is not None:
        return _DWTForward(J=1, wave="haar")
    return HaarDWTForward()


def dynamic_tanh(x, alpha, weight, bias):
    return weight * torch.tanh(alpha * x) + bias


class DERWE(nn.Module):
    def __init__(self, in_channels, n=1):
        super().__init__()
        self.identety = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * n,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.DWT = _make_dwt()
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.tanh_weight = nn.Parameter(
            torch.ones(1, in_channels * 4, 1, 1)
        )
        self.tanh_bias = nn.Parameter(
            torch.zeros(1, in_channels * 4, 1, 1)
        )
        self.dconv_encode = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * n, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def _transformer(self, DMT1_yl, DMT1_yh):
        high = DMT1_yh[0]
        parts = [DMT1_yl]
        for i in range(3):
            parts.append(high[:, :, i, :, :])
        x_wavelet = torch.cat(parts, 1)
        return dynamic_tanh(
            x_wavelet,
            self.alpha,
            self.tanh_weight,
            self.tanh_bias,
        )

    def forward(self, x):
        yl, yh = self.DWT(x)
        encoded = self.dconv_encode(self._transformer(yl, yh))
        residual = self.identety(x)
        return encoded + residual
