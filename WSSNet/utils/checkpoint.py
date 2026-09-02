from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch


_STATE_KEYS = ("state_dict", "model_state_dict", "model")


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values())


def _extract_state_dict(checkpoint: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if _looks_like_state_dict(checkpoint):
        return checkpoint, "root"
    if isinstance(checkpoint, dict):
        for key in _STATE_KEYS:
            value = checkpoint.get(key)
            if _looks_like_state_dict(value):
                return value, key
    raise ValueError("Checkpoint does not contain a recognizable state_dict.")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True, map_location="cpu"):
    checkpoint = torch.load(str(Path(checkpoint_path)), map_location=map_location)
    state_dict, source_key = _extract_state_dict(checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return {
        "source_key": source_key,
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }
