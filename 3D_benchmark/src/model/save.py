import torch
from torch import nn

def _to_cpu(state_dict: dict) -> dict:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in state_dict.items()
    }

def save_checkpoint(
    epoch: int, 
    model: nn.Module, 
    metric_value_dict: dict,
    path: str,
):
    state = {
        "epoch": epoch,
        "model_state_dict": _to_cpu(model.state_dict()),
        "metric_value_dict": metric_value_dict,
    }
    torch.save(state, path)