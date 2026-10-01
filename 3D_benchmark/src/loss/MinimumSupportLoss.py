import torch
import torch.nn as nn


class MinimumSupportLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(MinimumSupportLoss, self).__init__()
        self.beta = beta

    def forward(self, pred):
        # pred shape (batch, 1, nz, nx)
        # Minimum support functional: sum(pred^2 / (pred^2 + beta^2))
        # Penalizes the volume of the anomalous region, producing compact bodies.
        msf = torch.sum(pred ** 2 / (pred ** 2 + self.beta ** 2))
        return msf / pred.numel()