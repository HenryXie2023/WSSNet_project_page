from .wssnet import WSSNet
from .wssnet_mini import WSSNetMini
from .wssnet_tiny import WSSNetTiny

MODEL_REGISTRY = {
    "WSSNet": WSSNet,
    "WSSNet-Mini": WSSNetMini,
    "WSSNet-Tiny": WSSNetTiny,
}


def get_model(model_name: str, num_classes: int = 2):
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        valid = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{model_name}'. Valid choices: {valid}") from exc
    return model_cls(num_classes=num_classes)


__all__ = ["WSSNet", "WSSNetMini", "WSSNetTiny", "get_model", "MODEL_REGISTRY"]
