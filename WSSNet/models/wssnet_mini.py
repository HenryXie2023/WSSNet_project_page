import torch.nn as nn

from .modules import DERWE, EncoderDownBlock, MEMCAUBridge, MEMCAUMaskAdapter, MEMCAUUpBlock, MSSAtt


class WSSNetMini(nn.Module):
    """Medium Wavelet-State-Space Network.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        in_filters = [32, 64, 128]
        out_filters = [64, 128]
        self.unetdown1 = EncoderDownBlock(3, in_filters[0])
        self.unetdown2 = DERWE(in_filters[0], n=2)
        self.unetdown3 = DERWE(in_filters[1], n=2)
        self.up_concat2 = MEMCAUUpBlock(192, out_filters[1])
        self.up_concat1 = MEMCAUUpBlock(160, out_filters[0])
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.mcm3 = MEMCAUMaskAdapter(inc=128, outc=64)
        self.mcm4 = MEMCAUMaskAdapter(inc=128, outc=32)
        self.b3 = MEMCAUBridge(dim_xh=128, dim_xl=64)
        self.b4 = MEMCAUBridge(dim_xh=128, dim_xl=32)
        self.att1 = MSSAtt(in_channels=32)

    def forward(self, inputs):
        feat2 = self.att1(self.unetdown1(inputs))
        feat3 = self.unetdown2(feat2)
        feat4 = self.unetdown3(feat3)
        m3, M3 = self.mcm3(feat3, feat4)
        up2 = self.up_concat2(M3 + self.b3(feat4, feat3, m3), feat4)
        m4, M4 = self.mcm4(feat2, up2)
        up1 = self.up_concat1(M4 + self.b4(up2, feat2, m4), up2)
        return self.final(up1)
