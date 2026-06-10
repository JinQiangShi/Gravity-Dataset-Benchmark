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


def build_scheduler(optimizer, name, epochs, step_size=30, gamma=0.1, warmup_epochs=10):
    """Build a learning rate scheduler.

    Supports: "cosine", "step", "warmup_cosine", "warmup_step", "none".
    """
    sched = torch.optim.lr_scheduler

    base_schedulers = {
        "cosine": lambda: sched.CosineAnnealingLR(optimizer, T_max=epochs),
        "step": lambda: sched.StepLR(optimizer, step_size=step_size, gamma=gamma),
        "none": lambda: None,
    }

    if name in base_schedulers:
        return base_schedulers[name]()

    if name in ("warmup_cosine", "warmup_step"):
        warmup = sched.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
        main = (sched.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
                if name == "warmup_cosine"
                else sched.StepLR(optimizer, step_size=step_size, gamma=gamma))
        return sched.SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])

    raise ValueError(f"Unknown scheduler: {name}")


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric_value, path,
                    best_metric_name=None, metrics=None):
    """Save a training checkpoint."""
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
