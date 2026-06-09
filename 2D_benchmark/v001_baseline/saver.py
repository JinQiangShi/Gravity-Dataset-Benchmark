import torch


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric_value, path,
                    best_metric_name=None, metrics=None):
    """Unified checkpoint saving for best model, periodic, and last model.

    Args:
        model: the model.
        optimizer: the optimizer.
        scheduler: the scheduler (or None).
        epoch: current epoch index.
        best_metric_value: the metric value used for best-model selection.
        path: file path to save the checkpoint.
        best_metric_name: name of the selection metric (e.g. "loss", "psnr").
        metrics: dict of current metrics (optional, only for best model).
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }

    if metrics is not None:
        state["metrics"] = metrics

    torch.save(state, path)
