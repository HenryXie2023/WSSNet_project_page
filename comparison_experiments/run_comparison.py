from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_baseline(model_key: str, sources_path: Path) -> dict:
    sources = load_yaml(sources_path).get("baselines", {})
    try:
        entry = sources[model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(sources))
        raise ValueError(f"Unknown model '{model_key}'. Valid choices: {valid}") from exc
    return entry


def command_from_template(template: str, values: dict) -> list[str]:
    rendered = template.format(**values)
    return shlex.split(rendered, posix=os.name != "nt")


def build_values(model_key: str, baseline: dict, protocol: dict, repository_path: Path, output_dir: Path) -> dict:
    dataset = protocol["dataset"]
    train_set = resolve_path(dataset["train_set_path"])
    val_set = resolve_path(dataset["validation_set_path"])
    test_set = resolve_path(dataset["test_set_path"])
    image_dir_name = dataset.get("image_dir_name", "images")
    mask_dir_name = dataset.get("mask_dir_name", "masks")
    return {
        "model": model_key,
        "model_name": baseline["name"],
        "repository_path": str(repository_path),
        "train_set_path": str(train_set),
        "validation_set_path": str(val_set),
        "test_set_path": str(test_set),
        "train_images": str(train_set / image_dir_name),
        "train_masks": str(train_set / mask_dir_name),
        "validation_images": str(val_set / image_dir_name),
        "validation_masks": str(val_set / mask_dir_name),
        "test_images": str(test_set / image_dir_name),
        "test_masks": str(test_set / mask_dir_name),
        "input_height": protocol["input_size"][0],
        "input_width": protocol["input_size"][1],
        "batch_size": protocol["batch_size"],
        "max_epochs": protocol["max_epochs"],
        "optimizer": protocol["optimizer"],
        "initial_learning_rate": protocol["initial_learning_rate"],
        "minimum_learning_rate": protocol["minimum_learning_rate"],
        "learning_rate_schedule": protocol["learning_rate_schedule"],
        "random_seed": protocol["random_seed"],
        "output_dir": str(output_dir),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "prediction_dir": str(output_dir / "predictions"),
        "metric_dir": str(output_dir / "metrics"),
    }


def get_template(args: argparse.Namespace, model_key: str, phase: str) -> str | None:
    explicit = getattr(args, f"{phase}_command")
    if explicit:
        return explicit
    env_key = f"COMPARISON_{model_key.upper()}_{phase.upper()}_COMMAND"
    return os.environ.get(env_key)


def run_command(template: str, values: dict, cwd: Path) -> None:
    command = command_from_template(template, values)
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an external baseline under the unified comparison protocol.")
    parser.add_argument("--model", required=True, choices=[
        "unet",
        "deeplabv3plus",
        "transunet",
        "emcad",
        "selfreg_unet",
        "efficientvim",
        "transoilseg",
        "oilspillnet",
    ])
    parser.add_argument("--sources", default=str(EXPERIMENT_DIR / "baseline_sources.yaml"))
    parser.add_argument("--protocol", default=str(EXPERIMENT_DIR / "protocol.yaml"))
    parser.add_argument("--repository-path", default=None)
    parser.add_argument("--train-command", default=None)
    parser.add_argument("--predict-command", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    protocol = load_yaml(Path(args.protocol))
    baseline = load_baseline(args.model, Path(args.sources))
    repository_path = resolve_path(args.repository_path or baseline["local_repository_path"])
    if not repository_path.exists():
        raise FileNotFoundError(
            f"External repository not found: {repository_path}. "
            "Run prepare_baselines.py or set --repository-path."
        )

    output_dir = resolve_path(protocol["output_directory"]) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    values = build_values(args.model, baseline, protocol, repository_path, output_dir)
    Path(values["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(values["prediction_dir"]).mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        train_template = get_template(args, args.model, "train")
        if not train_template:
            raise ValueError(
                "No training command template was provided. Use --train-command or "
                f"COMPARISON_{args.model.upper()}_TRAIN_COMMAND."
            )
        run_command(train_template, values, repository_path)

    if not args.skip_predict:
        predict_template = get_template(args, args.model, "predict")
        if not predict_template:
            raise ValueError(
                "No prediction command template was provided. Use --predict-command or "
                f"COMPARISON_{args.model.upper()}_PREDICT_COMMAND."
            )
        run_command(predict_template, values, repository_path)

    if not args.skip_evaluate:
        eval_script = EXPERIMENT_DIR / "evaluate_predictions.py"
        subprocess.run(
            [
                os.environ.get("PYTHON", "python"),
                str(eval_script),
                "--pred-dir",
                values["prediction_dir"],
                "--gt-dir",
                values["test_masks"],
                "--output-dir",
                values["metric_dir"],
                "--model-name",
                args.model,
            ],
            cwd=str(ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
