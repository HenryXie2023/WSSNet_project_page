from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def read_binary_mask(path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    return (arr > 127).astype(np.uint8)


def confusion_from_arrays(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    return tp, tn, fp, fn


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    oa = _safe_div(tp + tn, total)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    f2 = _safe_div(5 * precision * recall, 4 * precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    if total:
        po = oa
        pe = _safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), total * total)
        kappa = _safe_div(po - pe, 1 - pe)
    else:
        kappa = 0.0
    return {
        "OA": oa * 100.0,
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1": f1 * 100.0,
        "IoU": iou * 100.0,
        "Kappa": kappa * 100.0,
        "Specificity": specificity * 100.0,
        "FPR": fpr * 100.0,
        "FNR": fnr * 100.0,
        "F2": f2 * 100.0,
    }


def binary_metrics_from_arrays(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    return metrics_from_counts(*confusion_from_arrays(pred, gt))
