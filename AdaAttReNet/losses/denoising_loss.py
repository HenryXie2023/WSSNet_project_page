import torch
import torch.nn as nn


class HDRLoss(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self, denoised, target):
        loss = ((denoised - target) ** 2) / (denoised + self.eps) ** 2
        return torch.mean(loss.view(-1))


def build_loss(name, is_mc=False):
    if name == "hdr":
        assert is_mc, "HDR loss requires Monte Carlo inputs"
        return HDRLoss()
    if name == "l2":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"invalid loss: {name}")


def denoising_loss(loss_fn, model, source, target):
    output = model(source)
    att_output = model.att_unet(source)
    pca_output = model.pca_unet(source)
    loss = loss_fn(output, target)
    loss = loss + loss_fn(att_output, target) / 10
    loss = loss + loss_fn(pca_output, target) / 10
    return loss, output
