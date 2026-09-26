import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, pred):
        # pred shape (batch, 1, nz, nx)

        # Calculate gradients along nx and nz directions
        diff_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # nx differences
        diff_z = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # nz differences

        # Anisotropic L1 TV: sum of gradient magnitudes per direction
        # sqrt(d^2 + eps) is a smooth (differentiable) approximation of |d|
        tv_loss = torch.sum(torch.sqrt(diff_x ** 2 + self.epsilon))
        tv_loss += torch.sum(torch.sqrt(diff_z ** 2 + self.epsilon))

        return tv_loss / pred.numel()