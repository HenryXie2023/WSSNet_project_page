import torch.nn as nn

from .modules import DERWE, EncoderDownBlock, MEMCAUBridge, MEMCAUMaskAdapter, MEMCAUUpBlock, MSSAtt


class WSSNet(nn.Module):
    """Full Wavelet-State-Space Network.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        in_filters = [32, 64, 128, 256]
        out_filters = [64, 128, 256]
        self.unetdown1 = EncoderDownBlock(3, in_filters[0])
        self.unetdown2 = DERWE(in_filters[0], n=2)
        self.unetdown3 = DERWE(in_filters[1], n=2)
        self.unetdown4 = DERWE(in_filters[2], n=2)
        self.up_concat3 = MEMCAUUpBlock(384, out_filters[2])
        self.up_concat2 = MEMCAUUpBlock(320, out_filters[1])
        self.up_concat1 = MEMCAUUpBlock(160, out_filters[0])
        self.mcm4 = MEMCAUMaskAdapter(inc=256, outc=128)
        self.mcm3 = MEMCAUMaskAdapter(inc=256, outc=64)
        self.mcm2 = MEMCAUMaskAdapter(inc=128, outc=32)
        self.b4 = MEMCAUBridge(dim_xh=256, dim_xl=128)
        self.b3 = MEMCAUBridge(dim_xh=256, dim_xl=64)
        self.b2 = MEMCAUBridge(dim_xh=128, dim_xl=32)
        self.att1 = MSSAtt(in_channels=32)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat2 = self.att1(self.unetdown1(inputs))
        feat3 = self.unetdown2(feat2)
        feat4 = self.unetdown3(feat3)
        feat5 = self.unetdown4(feat4)
        m4, M4 = self.mcm4(feat4, feat5)
        up3 = self.up_concat3(M4 + self.b4(feat5, feat4, m4), feat5)
        m3, M3 = self.mcm3(feat3, up3)
        up2 = self.up_concat2(M3 + self.b3(up3, feat3, m3), up3)
        m2, M2 = self.mcm2(feat2, up2)
        up1 = self.up_concat1(M2 + self.b2(up2, feat2, m2), up2)
        return self.final(up1)
