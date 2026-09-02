from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WSSNET_ROOT = ROOT / "WSSNet"
if str(WSSNET_ROOT) not in sys.path:
    sys.path.insert(0, str(WSSNET_ROOT))

from tools.evaluate import run_evaluation
from utils.metrics import binary_metrics_from_arrays


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate comparison predictions with the unified segmentation metrics.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="comparison")
    parser.add_argument("--model-name", default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = run_evaluation(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
    )
    print(result["summary"])


__all__ = ["binary_metrics_from_arrays", "run_evaluation"]


if __name__ == "__main__":
    main()
