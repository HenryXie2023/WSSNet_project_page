import torch.nn as nn

from .modules import DERWE, EncoderDownBlock, MEMCAUBridge, MEMCAUMaskAdapter, MEMCAUUpBlock, MSSAtt


class WSSNetTiny(nn.Module):
    """Small Wavelet-State-Space Network.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.unetdown1 = EncoderDownBlock(3, 32)
        self.unetdown2 = DERWE(32, n=2)
        self.up_concat1 = MEMCAUUpBlock(96, 64)
        self.final = nn.Conv2d(64, num_classes, 1)
        self.mcm4 = MEMCAUMaskAdapter(inc=64, outc=32)
        self.b4 = MEMCAUBridge(dim_xh=64, dim_xl=32)
        self.att1 = MSSAtt(in_channels=32)

    def forward(self, inputs):
        feat2 = self.att1(self.unetdown1(inputs))
        feat3 = self.unetdown2(feat2)
        m4, M4 = self.mcm4(feat2, feat3)
        up1 = self.up_concat1(M4 + self.b4(feat3, feat2, m4), feat3)
        return self.final(up1)
