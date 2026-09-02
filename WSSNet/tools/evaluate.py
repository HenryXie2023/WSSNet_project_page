from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.image_io import IMAGE_EXTENSIONS
from utils.metrics import confusion_from_arrays, metrics_from_counts, read_binary_mask


METRIC_FIELDS = ["OA", "Precision", "Recall", "F1", "IoU", "Kappa", "Specificity", "FPR", "FNR", "F2"]


def _files_by_stem(directory):
    directory = Path(directory)
    if not directory.exists():
        return {}
    return {
        p.stem: p
        for p in sorted(directory.iterdir(), key=lambda x: x.name.lower())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def run_evaluation(pred_dir, gt_dir, output_dir, dataset_name="", model_name=""):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_files = _files_by_stem(pred_dir)
    gt_files = _files_by_stem(gt_dir)
    pred_stems = set(pred_files)
    gt_stems = set(gt_files)
    matched = sorted(pred_stems & gt_stems)
    unmatched_rows = []
    for stem in sorted(gt_stems - pred_stems):
        unmatched_rows.append({"stem": stem, "issue": "missing_prediction", "pred_path": "", "gt_path": str(gt_files[stem])})
    for stem in sorted(pred_stems - gt_stems):
        unmatched_rows.append({"stem": stem, "issue": "missing_gt", "pred_path": str(pred_files[stem]), "gt_path": ""})
    unmatched_csv = output_dir / "unmatched_files.csv"
    with unmatched_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "issue", "pred_path", "gt_path"])
        writer.writeheader()
        writer.writerows(unmatched_rows)

    per_image_rows = []
    size_mismatch_rows = []
    total_tp = total_tn = total_fp = total_fn = 0
    for stem in tqdm(matched, desc=f"Evaluate {dataset_name} {model_name}", leave=False):
        pred = read_binary_mask(pred_files[stem])
        gt = read_binary_mask(gt_files[stem])
        if pred.shape != gt.shape:
            size_mismatch_rows.append({
                "stem": stem,
                "pred_shape": "x".join(map(str, pred.shape)),
                "gt_shape": "x".join(map(str, gt.shape)),
                "pred_path": str(pred_files[stem]),
                "gt_path": str(gt_files[stem]),
            })
            continue
        tp, tn, fp, fn = confusion_from_arrays(pred, gt)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        metrics = metrics_from_counts(tp, tn, fp, fn)
        row = {
            "dataset": dataset_name,
            "model_name": model_name,
            "image_name": gt_files[stem].name,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }
        row.update({k: f"{metrics[k]:.6f}" for k in METRIC_FIELDS})
        per_image_rows.append(row)
    size_csv = output_dir / "size_mismatch_files.csv"
    with size_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "pred_shape", "gt_shape", "pred_path", "gt_path"])
        writer.writeheader()
        writer.writerows(size_mismatch_rows)
    per_image_csv = output_dir / "per_image_metrics.csv"
    with per_image_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["dataset", "model_name", "image_name", "TP", "TN", "FP", "FN"] + METRIC_FIELDS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)
    summary_metrics = metrics_from_counts(total_tp, total_tn, total_fp, total_fn)
    summary = {
        "dataset": dataset_name,
        "model_name": model_name,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "prediction_file_count": len(pred_files),
        "gt_file_count": len(gt_files),
        "matched_file_count": len(matched),
        "evaluated_file_count": len(per_image_rows),
        "unmatched_file_count": len(unmatched_rows),
        "size_mismatch_count": len(size_mismatch_rows),
        "TP": total_tp,
        "TN": total_tn,
        "FP": total_fp,
        "FN": total_fn,
    }
    summary.update({k: f"{summary_metrics[k]:.6f}" for k in METRIC_FIELDS})
    summary_csv = output_dir / "dataset_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return {
        "summary": summary,
        "per_image_csv": str(per_image_csv),
        "summary_csv": str(summary_csv),
        "unmatched_csv": str(unmatched_csv),
        "size_mismatch_csv": str(size_csv),
    }


def build_argparser():
    parser = argparse.ArgumentParser(description="Evaluate binary segmentation masks.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--model-name", default="")
    return parser


def main():
    args = build_argparser().parse_args()
    result = run_evaluation(args.pred_dir, args.gt_dir, args.output_dir, args.dataset_name, args.model_name)
    print(result["summary"])


if __name__ == "__main__":
    main()
