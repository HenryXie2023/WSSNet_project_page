import os

import numpy as np
import torchvision.transforms.functional as tvF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SARDataset(Dataset):
    def __init__(self, root_dir, size=0, crop_size=128, clean_targets=False, noise_type="speckle2", noise_param=50.0, seed=None):
        super().__init__()
        self.root_dir = root_dir
        self.size = size
        self.crop_size = crop_size
        self.clean_targets = clean_targets
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.seed = seed
        self.imgs = []
        if root_dir:
            self.imgs = sorted(
                name
                for name in os.listdir(root_dir)
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            )
        if size:
            self.imgs = self.imgs[:size]

    def _random_crop(self, img_list):
        width, height = img_list[0].size
        assert width >= self.crop_size and height >= self.crop_size, (
            f"crop_size={self.crop_size}, image_size=({width}, {height})"
        )
        top = np.random.randint(0, height - self.crop_size + 1)
        left = np.random.randint(0, width - self.crop_size + 1)
        return [tvF.crop(img, top, left, self.crop_size, self.crop_size) for img in img_list]

    def _add_noise(self, img):
        width, height = img.size
        channels = len(img.getbands())
        img_arr = np.array(img).astype(np.float32)

        if self.noise_type == "poisson":
            noise = np.random.poisson(img_arr)
            noise_img = img_arr + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))
        elif self.noise_type == "speckle":
            looks = 1
            noise = np.random.gamma(looks, 1.0 / looks, (height, width, channels))
            noise_img = img_arr * noise
        elif self.noise_type == "speckle2":
            looks = np.random.uniform(1.0, 2.0)
            noise = np.random.gamma(looks, 1.0 / looks, (height, width, channels)).astype(np.float32)
            noise_img = img_arr * noise
        elif self.noise_type == "gaussian":
            std = 25 if self.seed else np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (height, width, channels))
            noise_img = img_arr + noise
        else:
            raise ValueError(f"invalid noise_type: {self.noise_type}")

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img = Image.open(img_path).convert("RGB")
        if self.crop_size != 0:
            img = self._random_crop([img])[0]
        source = tvF.to_tensor(self._add_noise(img))
        target = tvF.to_tensor(img)
        return source, target

    def __len__(self):
        return len(self.imgs)


def load_dataset(root_dir, size, config, shuffled=False, single=False):
    dataset = SARDataset(
        root_dir=root_dir,
        size=size,
        crop_size=config["data"]["crop_size"],
        clean_targets=config["training"]["clean_targets"],
        noise_type=config["noise"]["type"],
        noise_param=config["noise"]["param"],
        seed=config["training"]["seed"],
    )
    batch_size = 1 if single else config["training"]["batch_size"]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffled)
