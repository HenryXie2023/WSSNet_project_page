from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(image_dir, limit=None) -> List[Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return []
    images = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )
    if limit is not None:
        images = images[: int(limit)]
    return images


def cvt_color(image: Image.Image) -> Image.Image:
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    return image.convert("RGB")


def resize_image(image: Image.Image, size: Tuple[int, int]):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh


def preprocess_input(image: np.ndarray) -> np.ndarray:
    return image / 255.0


def predict_mask(model, image: Image.Image, input_shape: Sequence[int], device: torch.device) -> np.ndarray:
    image = cvt_color(image)
    original_h = np.array(image).shape[0]
    original_w = np.array(image).shape[1]
    image_data, nw, nh = resize_image(image, (int(input_shape[1]), int(input_shape[0])))
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
        0,
    )
    with torch.no_grad():
        images = torch.from_numpy(image_data).to(device=device, dtype=torch.float32)
        pr = model(images)
        if isinstance(pr, (tuple, list)):
            pr = pr[0]
        pr = pr[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).detach().cpu().numpy()
        top = int((int(input_shape[0]) - nh) // 2)
        left = int((int(input_shape[1]) - nw) // 2)
        pr = pr[top : top + nh, left : left + nw]
        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return pr.argmax(axis=-1).astype(np.uint8)


def save_mask(mask: np.ndarray, output_path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(output_path)


def save_visualization(image: Image.Image, mask: np.ndarray, output_path, alpha=0.45) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = rgb.copy()
    overlay[mask > 0] = [255, 64, 64]
    blended = (rgb * (1 - alpha) + overlay * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(output_path)
