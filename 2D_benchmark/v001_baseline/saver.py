import torch


def save_checkpoint(model, optimizer, scheduler, epoch, improvement_score, path,
                    tracker=None, metrics=None):
    """Unified checkpoint saving for best model, periodic, and last model.

    Args:
        model: the model.
        optimizer: the optimizer.
        scheduler: the scheduler (or None).
        epoch: current epoch index.
        improvement_score: improvement score from adaptive tracker.
        path: file path to save the checkpoint.
        tracker: AdaptiveMetricTracker instance (optional, saved for resume).
        metrics: dict of current metrics (optional, only for best model).
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": improvement_score,
    }

    if tracker is not None:
        state["tracker_best"] = dict(tracker.best)
        state["tracker_n"] = tracker.n
        state["tracker_mean"] = dict(tracker._mean)
        state["tracker_m2"] = dict(tracker._m2)

    if metrics is not None:
        state["metrics"] = metrics

    torch.save(state, path)
