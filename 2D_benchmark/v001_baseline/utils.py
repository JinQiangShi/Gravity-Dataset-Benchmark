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


class AdaptiveMetricTracker:
    """Adaptive best-model selector using normalized improvement scores.

    For each tracked metric, maintains a running mean/std (Welford's algorithm).
    Improvement for each metric is measured as:
        (current_value - current_best) / running_std   (for higher-is-better)
        (current_best - current_value) / running_std   (for lower-is-better)

    This is weight-free: metrics are automatically normalized by their own
    volatility. A stable metric that suddenly improves significantly contributes
    a large score (small std), while a noisy metric contributes little (large std).

    The total improvement score is the sum of per-metric normalized improvements.
    A new best model is saved when this score exceeds a threshold (after warmup).
    """

    def __init__(self, warmup_epochs=5, threshold=0.5, higher_is_better=None):
        self.warmup_epochs = warmup_epochs
        self.threshold = threshold
        self.higher_is_better = higher_is_better or {}
        self.n = 0
        # Welford running stats: {metric_name: (mean, M2)}
        self._mean = {}
        self._m2 = {}
        # Current best per metric
        self.best = {}

    def _running_std(self, name):
        if self.n < 2 or name not in self._m2:
            return 1.0
        variance = self._m2[name] / (self.n - 1)
        return max(variance ** 0.5, 1e-8)

    def evaluate(self, metrics):
        """Evaluate whether the current epoch should be saved as best model.

        Args:
            metrics: dict of {name: value}, e.g. {"psnr": 30.5, "ssim": 0.92, ...}

        Returns:
            (should_save: bool, improvement_score: float)
        """
        self.n += 1

        # Update Welford running statistics for all metrics
        for name, value in metrics.items():
            if name not in self._mean:
                self._mean[name] = value
                self._m2[name] = 0.0
            else:
                old_mean = self._mean[name]
                self._mean[name] = old_mean + (value - old_mean) / self.n
                self._m2[name] += (value - old_mean) * (value - self._mean[name])

        # Update best (always track)
        for name, value in metrics.items():
            hib = self.higher_is_better.get(name, True)
            if name not in self.best:
                self.best[name] = value
            elif hib and value > self.best[name]:
                self.best[name] = value
            elif not hib and value < self.best[name]:
                self.best[name] = value

        # During warmup, always save (use simple sum-of-bests as score)
        if self.n <= self.warmup_epochs:
            return True, self.n

        # Compute normalized improvement score
        score = 0.0
        for name, value in metrics.items():
            hib = self.higher_is_better.get(name, True)
            std = self._running_std(name)
            if hib:
                improvement = value - self.best[name]
            else:
                improvement = self.best[name] - value
            score += improvement / std

        return score > self.threshold, score


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
