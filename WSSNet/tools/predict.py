from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import get_model
from utils.checkpoint import load_checkpoint
from utils.image_io import iter_images, predict_mask, save_mask, save_visualization


def parse_shape(text):
    if isinstance(text, (list, tuple)):
        return [int(text[0]), int(text[1])]
    parts = str(text).replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use HxW, e.g. 256x256")
    return [int(parts[0]), int(parts[1])]


def run_prediction(
    model_name,
    checkpoint,
    input_dir,
    output_dir,
    visualization_dir=None,
    log_file=None,
    input_shape=(256, 256),
    num_classes=2,
    device=None,
    limit=None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = get_model(model_name, num_classes=num_classes).to(device)
    load_checkpoint(
        model,
        checkpoint,
        map_location=device,
        strict=True,
    )
    model.eval()
    images = iter_images(input_dir, limit=limit)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if visualization_dir:
        visualization_dir = Path(visualization_dir)
        visualization_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        log_file = output_dir / "prediction_log.csv"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = 0
    start_all = time.perf_counter()
    for image_path in tqdm(images, desc=f"Predict {model_name}", leave=False):
        row = {
            "image_name": image_path.name,
            "output_name": image_path.with_suffix(".png").name,
            "status": "ok",
            "error": "",
            "width": "",
            "height": "",
            "elapsed_ms": "",
        }
        start = time.perf_counter()
        try:
            image = Image.open(image_path)
            row["width"], row["height"] = image.size
            mask = predict_mask(model, image, input_shape, device)
            save_mask(mask, output_dir / image_path.with_suffix(".png").name)
            if visualization_dir:
                save_visualization(image, mask, visualization_dir / image_path.with_suffix(".png").name)
        except Exception as exc:
            failures += 1
            row["status"] = "failed"
            row["error"] = repr(exc)
        row["elapsed_ms"] = f"{(time.perf_counter() - start) * 1000.0:.4f}"
        rows.append(row)
    with log_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "output_name", "status", "error", "width", "height", "elapsed_ms"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "model_name": model_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(images),
        "predicted_count": len(images) - failures,
        "failure_count": failures,
        "elapsed_seconds": time.perf_counter() - start_all,
        "log_file": str(log_file),
    }


def build_argparser():
    parser = argparse.ArgumentParser(description="Run WSSNet directory prediction.")
    parser.add_argument("--model", required=True, choices=["WSSNet", "WSSNet-Mini", "WSSNet-Tiny"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visualization-dir", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--input-shape", type=parse_shape, default=[256, 256])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main():
    args = build_argparser().parse_args()
    summary = run_prediction(
        model_name=args.model,
        checkpoint=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        visualization_dir=args.visualization_dir,
        log_file=args.log_file,
        input_shape=args.input_shape,
        num_classes=args.num_classes,
        device=args.device,
        limit=args.limit,
    )
    print(summary)


if __name__ == "__main__":
    main()
