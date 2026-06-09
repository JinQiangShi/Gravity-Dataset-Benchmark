class TrainingLogger:
    """Unified training logger wrapping tensorboard and swanlab.

    Usage:
        logger = TrainingLogger("tensorboard", log_dir="logs")
        logger.log_scalars({"loss": 0.5, "psnr": 30.0}, step=0)
        logger.close()
    """

    def __init__(self, backend, log_dir=None, project=None, workspace=None, config=None):
        """
        Args:
            backend: "tensorboard" or "swanlab"
            log_dir: tensorboard log directory
            project: swanlab project name
            workspace: swanlab workspace name
            config: dict of hyperparameters (passed to swanlab.init)
        """
        self.backend = backend
        self._writer = None

        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir)
        elif backend == "swanlab":
            import swanlab
            self._swanlab = swanlab
            swanlab.init(
                project=project,
                workspace=workspace,
                config=config or {},
            )

    def log_scalars(self, metrics, step):
        """Log a dict of scalar metrics.

        Args:
            metrics: dict of {name: value}
            step: global step (epoch)
        """
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

    Args:
        optimizer: PyTorch optimizer
        name: scheduler name ("cosine", "step", "warmup_cosine", "warmup_step", "none")
        epochs: total training epochs
        step_size: step size for step-based schedulers
        gamma: decay factor for step-based schedulers
        warmup_epochs: number of warmup epochs for warmup variants

    Returns:
        lr_scheduler instance, or None if name == "none"
    """
    import torch

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "warmup_cosine":
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[warmup_epochs])
    elif name == "warmup_step":
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
        main_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[warmup_epochs])
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")
