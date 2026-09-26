import torch
import torch.nn as nn


class MinimumGradientSupportLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(MinimumGradientSupportLoss, self).__init__()
        self.beta = beta

    def forward(self, pred):
        # pred shape (batch, 1, nz, nx)
        # Differences along nx and nz directions
        diff_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # nx differences
        diff_z = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # nz differences

        # Minimum gradient support functional: sum(g^2 / (g^2 + beta^2))
        # Shrinks the support of model gradients, sharpening the interfaces.
        mgs = torch.sum(diff_x ** 2 / (diff_x ** 2 + self.beta ** 2))
        mgs += torch.sum(diff_z ** 2 / (diff_z ** 2 + self.beta ** 2))

        return mgs / pred.numel()