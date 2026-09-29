import torch.optim as optim
import math


class WarmupCosineScheduler(optim.lr_scheduler.LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing whose
    restart peaks decay by a factor after each warm restart.

    Parameters
    ----------
    optimizer (optim.Optimizer): optimizer whose learning rate is scheduled.
    warmup_epochs (int): number of warmup epochs, default is 20.
    T_0 (int): number of epochs for the first cosine restart cycle, default is 20.
    T_mult (float): multiplicative factor for the cycle length after each restart, default is 1.
    eta_min (float): minimum learning rate, default is 1e-6.
    start_factor (float): starting factor of the warmup relative to the base lr, default is 0.1.
    restart_decay (float): factor applied to the peak learning rate at each restart, default is 0.5.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 20,
        T_0: int = 20,
        T_mult: float = 1.0,
        eta_min: float = 1e-6,
        start_factor: float = 0.1,
        restart_decay: float = 0.5,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.start_factor = start_factor
        self.restart_decay = restart_decay
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        lrs = []

        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                # linear warmup from start_factor * base_lr to base_lr
                progress = epoch / self.warmup_epochs
                factor = self.start_factor + (1.0 - self.start_factor) * progress
                lrs.append(base_lr * factor)
                continue

            # locate current cosine cycle
            t = epoch - self.warmup_epochs
            cycle = 0
            cycle_start = 0
            cycle_length = self.T_0
            while t >= cycle_start + cycle_length:
                cycle_start += cycle_length
                cycle_length *= self.T_mult
                cycle += 1

            # decayed peak for this cycle: base, base*decay, base*decay^2, ...
            peak_lr = base_lr * (self.restart_decay ** cycle)

            progress = (t - cycle_start) / cycle_length
            lr = self.eta_min + (peak_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress)) / 2.0
            lrs.append(lr)

        return lrs