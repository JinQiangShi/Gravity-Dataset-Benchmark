from .unet import UNet
from .save import save_checkpoint
from .load import load_checkpoint

__all__ = [
    "get_model",
    "save_checkpoint",
    "load_checkpoint",
]

def get_model(
    model_type: str,
    model_kwargs: dict,
):
    if model_type == "unet":
        return UNet(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
