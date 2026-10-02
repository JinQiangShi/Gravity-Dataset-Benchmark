import torch
from torch import nn


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: str = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint
