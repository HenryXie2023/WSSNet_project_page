import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def seed_all(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def time_elapsed_since(start):
    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)
    return string, ms


def psnr(input_tensor, target_tensor):
    return 10 * torch.log10(1 / F.mse_loss(input_tensor, target_tensor))


def pad_to_multiple(tensor, multiple=32):
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, height, width
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, height, width


def tensor_to_image(tensor):
    array = tensor.detach().cpu().numpy().astype(np.float32)
    array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    if array.shape[0] == 1:
        return Image.fromarray(array[0], mode="L")
    if array.shape[0] == 3:
        return Image.fromarray(np.transpose(array, (1, 2, 0)), mode="RGB")
    return Image.fromarray(array[0], mode="L")
