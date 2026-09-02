from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .image_io import IMAGE_EXTENSIONS, cvt_color, preprocess_input, resize_image
from .metrics import read_binary_mask


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, input_shape=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.input_shape = tuple(input_shape)
        images = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )
        masks_by_stem = {
            p.stem: p
            for p in self.mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
        self.samples = [(p, masks_by_stem[p.stem]) for p in images if p.stem in masks_by_stem]
        if not self.samples:
            raise ValueError(f"No matched image/mask pairs under {self.image_dir} and {self.mask_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = cvt_color(Image.open(image_path))
        mask = Image.fromarray(read_binary_mask(mask_path) * 255)
        image, _, _ = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        mask = mask.resize((self.input_shape[1], self.input_shape[0]), Image.NEAREST)
        image = np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1))
        mask = (np.array(mask) > 127).astype(np.int64)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()
