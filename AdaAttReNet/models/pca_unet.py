import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_circle(size):
    p = (size - 1) / 2
    x = np.arange(-p, p + 1) / p
    grid_x, grid_y = np.meshgrid(x, x)
    circle = grid_x**2 + grid_y**2
    if size > 4:
        mask = np.exp(-np.maximum(circle - 1, 0) / 0.2)
    else:
        mask = np.exp(-np.maximum(circle - 1, 0) / 2)
    return grid_x, grid_y, mask


def _pca_basis(kernel_size, transform_count=8, input_size=None, smooth=True):
    if input_size is None:
        input_size = kernel_size
    in_x, in_y, mask = _mask_circle(kernel_size)
    x0 = np.expand_dims(in_x, 2)
    y0 = np.expand_dims(in_y, 2)
    mask = np.expand_dims(mask, 2)
    theta = np.arange(transform_count) / transform_count * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
    x = np.cos(theta) * x0 - np.sin(theta) * y0
    y = np.cos(theta) * y0 + np.sin(theta) * x0
    x = np.expand_dims(np.expand_dims(x, 3), 4)
    y = np.expand_dims(np.expand_dims(y, 3), 4)
    v = np.pi / input_size * (input_size - 1)
    p = input_size / 2
    k = np.reshape(np.arange(input_size), [1, 1, 1, input_size, 1])
    l = np.reshape(np.arange(input_size), [1, 1, 1, 1, input_size])

    basis_c = np.cos((k - input_size * (k > p)) * v * x + (l - input_size * (l > p)) * v * y)
    basis_s = np.sin((k - input_size * (k > p)) * v * x + (l - input_size * (l > p)) * v * y)

    basis_c = np.reshape(basis_c, [kernel_size, kernel_size, transform_count, input_size * input_size])
    basis_c = basis_c * np.expand_dims(mask, 3)
    basis_s = np.reshape(basis_s, [kernel_size, kernel_size, transform_count, input_size * input_size])
    basis_s = basis_s * np.expand_dims(mask, 3)

    basis_c = np.reshape(basis_c, [kernel_size * kernel_size * transform_count, input_size * input_size])
    basis_s = np.reshape(basis_s, [kernel_size * kernel_size * transform_count, input_size * input_size])
    basis = np.concatenate((basis_c, basis_s), axis=1)

    u, s, _ = np.linalg.svd(np.matmul(basis.T, basis))
    rank = np.sum(s > 0.0001)
    basis = np.matmul(np.matmul(basis, u[:, :rank]), np.diag(1 / np.sqrt(s[:rank] + 0.0000000001)))
    basis = np.reshape(basis, [kernel_size, kernel_size, transform_count, rank])

    temp = np.reshape(basis, [kernel_size * kernel_size, transform_count, rank])
    var = (
        np.std(np.sum(temp, axis=0) ** 2, axis=0)
        + np.std(np.sum(temp**2 * kernel_size * kernel_size, axis=0), axis=0)
    ) / np.mean(np.sum(temp, axis=0) ** 2 + np.sum(temp**2 * kernel_size * kernel_size, axis=0), axis=0)
    rank = np.sum(var < 1)
    weight = 1 / np.maximum(var, 0.04) / 25
    if smooth:
        basis = np.expand_dims(np.expand_dims(np.expand_dims(weight, 0), 0), 0) * basis

    return torch.FloatTensor(basis), rank, weight


def _init_regularized(basis_count, in_channels, out_channels, expand, weight=1):
    values = (np.random.rand(out_channels, in_channels, expand, basis_count) - 0.5) * 2
    values = values * 2.4495 / np.sqrt(in_channels * basis_count)
    values = values * np.expand_dims(np.expand_dims(np.expand_dims(weight, axis=0), axis=0), axis=0)
    return torch.FloatTensor(values)


class _PCAConv(nn.Module):
    def __init__(
        self,
        kernel_size,
        in_channels,
        out_channels,
        transform_count=8,
        input_size=None,
        padding=None,
        init_single_transform=0,
        bias=True,
        smooth=True,
        init_scale=0.4,
    ):
        super().__init__()
        if input_size is None:
            input_size = kernel_size
        self.transform_count = transform_count
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        basis, _, weight = _pca_basis(kernel_size, transform_count, input_size, smooth=smooth)
        self.register_buffer("basis", basis)
        self.expand = 1 if init_single_transform else transform_count
        init_weights = _init_regularized(basis.size(3), in_channels, out_channels, self.expand, weight) * init_scale
        self.weights = nn.Parameter(init_weights, requires_grad=True)
        self.padding = 0 if padding is None else padding
        self.c = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=bias)

    def _kernel_and_bias(self):
        temp_w = torch.einsum("ijok,mnak->monaij", self.basis, self.weights)
        num = self.transform_count // self.expand
        temp_w_list = [
            torch.cat(
                [
                    temp_w[:, i * num : (i + 1) * num, :, -i:, :, :],
                    temp_w[:, i * num : (i + 1) * num, :, :-i, :, :],
                ],
                dim=3,
            )
            for i in range(self.expand)
        ]
        temp_w = torch.cat(temp_w_list, dim=1)
        conv_filter = temp_w.reshape(
            [self.out_channels * self.transform_count, self.in_channels * self.expand, self.kernel_size, self.kernel_size]
        )
        conv_bias = self.c.repeat([1, 1, self.transform_count, 1]).reshape(
            [1, self.out_channels * self.transform_count, 1, 1]
        )
        return conv_filter, conv_bias

    def forward(self, x):
        if self.training:
            conv_filter, conv_bias = self._kernel_and_bias()
        else:
            conv_filter, conv_bias = self.filter, self.bias
        return F.conv2d(x, conv_filter, padding=self.padding, dilation=1, groups=1) + conv_bias

    def train(self, mode=True):
        if mode:
            if hasattr(self, "filter"):
                del self.filter
                del self.bias
        elif self.training:
            conv_filter, conv_bias = self._kernel_and_bias()
            self.register_buffer("filter", conv_filter)
            self.register_buffer("bias", conv_bias)
        return super().train(mode)


class _PCAConvOut(nn.Module):
    def __init__(
        self,
        kernel_size,
        in_channels,
        out_channels,
        transform_count=8,
        input_size=None,
        padding=None,
        bias=True,
        smooth=True,
        init_scale=0.4,
    ):
        super().__init__()
        if input_size is None:
            input_size = kernel_size
        self.transform_count = transform_count
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        basis, _, weight = _pca_basis(kernel_size, transform_count, input_size, smooth=smooth)
        self.register_buffer("basis", basis)
        init_weights = _init_regularized(basis.size(3), in_channels, out_channels, 1, weight) * init_scale
        self.weights = nn.Parameter(init_weights, requires_grad=True)
        self.padding = 0 if padding is None else padding
        self.c = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=bias)

    def _kernel(self):
        temp_w = torch.einsum("ijok,mnak->manoij", self.basis, self.weights)
        return temp_w.reshape([self.out_channels, self.in_channels * self.transform_count, self.kernel_size, self.kernel_size])

    def forward(self, x):
        conv_filter = self._kernel() if self.training else self.filter
        return F.conv2d(x, conv_filter, padding=self.padding, dilation=1, groups=1) + self.c

    def train(self, mode=True):
        if mode:
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            self.register_buffer("filter", self._kernel())
        return super().train(mode)


class PCAUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self._block1 = nn.Sequential(
            _PCAConv(3, in_channels, 12, 4, input_size=3, init_single_transform=1, padding=1),
            nn.ReLU(inplace=True),
            _PCAConv(3, 12, 12, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self._block2 = nn.Sequential(
            _PCAConv(3, 12, 12, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self._block3 = nn.Sequential(
            _PCAConv(3, 12, 12, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self._block4 = nn.Sequential(
            _PCAConv(3, 12, 12, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self._block5 = nn.Sequential(
            _PCAConv(3, 12, 12, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self._block6 = nn.Sequential(
            _PCAConv(3, 12, 12, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self._block7 = nn.Sequential(
            _PCAConv(3, 24, 24, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            _PCAConv(3, 24, 24, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self._block8 = nn.Sequential(
            _PCAConv(3, 36, 24, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            _PCAConv(3, 24, 24, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self._block9 = nn.Sequential(
            _PCAConv(3, 36, 24, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            _PCAConv(3, 24, 24, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self._block10 = nn.Sequential(
            _PCAConv(3, 36, 24, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            _PCAConv(3, 24, 24, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self._block11 = nn.Sequential(
            _PCAConv(3, 96 + in_channels, 16, 4, input_size=3, init_single_transform=1, padding=1),
            nn.ReLU(inplace=True),
            _PCAConv(3, 16, 8, 4, input_size=3, padding=1),
            nn.ReLU(inplace=True),
            _PCAConvOut(3, 8, out_channels, 4, input_size=3, padding=1),
        )

    def forward(self, x):
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block3(pool2)
        pool4 = self._block4(pool3)
        pool5 = self._block5(pool4)

        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block8(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        return self._block11(concat1)
