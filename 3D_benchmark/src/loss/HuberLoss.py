import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        # pred shape (batch, 1, nz, ny, nx)
        # target shape (batch, 1, nz, ny, nx)
        return F.huber_loss(pred, target, delta=self.delta)