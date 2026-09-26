import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, z0=1.0, beta=2.0):
        super(DepthWeightedHuberLoss, self).__init__()
        self.delta = delta
        self.z0 = z0
        self.beta = beta

    def forward(self, pred, target):
        # pred shape (batch, 1, nz, nx)
        # target shape (batch, 1, nz, nx)
        nz = pred.shape[2]

        # Depth weighting w(z) = (z + z0)^(-beta/2) counters the skin effect of
        # potential-field inversion, which otherwise over-focuses on shallow cells.
        z = torch.arange(nz, device=pred.device, dtype=pred.dtype) + self.z0
        weight = z ** (-self.beta / 2)

        # Normalize so the mean weight is 1, keeping the loss scale comparable.
        weight = weight / weight.mean()
        weight = weight.view(1, 1, nz, 1)

        return F.huber_loss(pred, target, delta=self.delta, weight=weight.expand_as(pred))