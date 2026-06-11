import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_scheduler(optimizer, name, patience=10, factor=0.1, warmup_epochs=10):
    """Build a learning rate scheduler.

    Supports: "plateau", "warmup_plateau", "none".

    For "warmup_plateau", returns a tuple (LinearLR, ReduceLROnPlateau).
    The caller should step LinearLR during warmup and ReduceLROnPlateau afterwards.
    """
    if name == "none":
        return None

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience,
        )

    if name == "warmup_plateau":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs,
        )
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience,
        )
        return (warmup, plateau)

    raise ValueError(f"Unknown scheduler: {name}")


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric_value, path,
                    best_metric_name=None, metrics=None):
    """Save a training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            {"warmup": scheduler[0].state_dict(), "plateau": scheduler[1].state_dict()}
            if isinstance(scheduler, tuple)
            else scheduler.state_dict() if scheduler else None
        ),
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, path)


class TrainingLogger:
    """Unified training logger wrapping tensorboard and swanlab."""

    def __init__(self, backend, log_dir=None, project=None, workspace=None, config=None):
        self.backend = backend
        self._writer = None

        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir)
        elif backend == "swanlab":
            import swanlab
            self._swanlab = swanlab
            swanlab.init(project=project, workspace=workspace, config=config or {})

    def log_scalars(self, metrics, step):
        """Log a dict of scalar metrics."""
        if self.backend == "tensorboard":
            for name, value in metrics.items():
                self._writer.add_scalar(name, value, step)
        elif self.backend == "swanlab":
            self._swanlab.log(metrics, step=step)

    def close(self):
        if self.backend == "tensorboard" and self._writer:
            self._writer.close()
        elif self.backend == "swanlab":
            self._swanlab.finish()


class EarlyStopping:
    """Early stopping to halt training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' (lower is better) or 'max' (higher is better).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

        if mode == "min":
            self._is_better = lambda cur, best: cur < best - min_delta
        elif mode == "max":
            self._is_better = lambda cur, best: cur > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def __call__(self, current_value: float) -> bool:
        """Update state. Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = current_value
            return False

        if self._is_better(current_value, self.best_score):
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop