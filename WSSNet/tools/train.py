from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import get_model
from utils.dataset import SegmentationDataset
from utils.misc import seed_everything


def load_yaml(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Train WSSNet models.")
    parser.add_argument("--model", required=True, choices=["WSSNet", "WSSNet-Mini", "WSSNet-Tiny"])
    parser.add_argument("--config", default="configs/wssnet.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("training", {})
    input_shape = cfg.get("input_shape", [256, 256])
    num_classes = int(cfg.get("num_classes", 2))
    epochs = args.epochs or int(train_cfg.get("epochs", 50))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 8))
    lr = args.lr or float(train_cfg.get("lr", 1e-4))
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 2))
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 11))
    seed_everything(seed)

    data_root = Path(args.data_root)
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("Expected --data-root to contain images/ and masks/ directories.")
    dataset = SegmentationDataset(image_dir, mask_dir, input_shape=input_shape)
    val_len = max(1, int(len(dataset) * 0.1))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = get_model(args.model, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs} train"):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss = F.cross_entropy(logits, masks, ignore_index=num_classes)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"epoch {epoch + 1}/{epochs} val"):
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = F.cross_entropy(logits, masks, ignore_index=num_classes)
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        torch.save(model.state_dict(), save_dir / "last_epoch_weights.pth")
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / "best_epoch_weights.pth")


if __name__ == "__main__":
    main()
