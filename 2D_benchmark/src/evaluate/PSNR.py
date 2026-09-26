import torch
import torch.nn as nn
import torch.nn.functional as F


class PSNR(nn.Module):
    def __init__(self, data_range=None):
        super(PSNR, self).__init__()
        self.data_range = data_range

    def forward(self, pred, target):
        # pred shape (batch, 1, nz, nx)
        # target shape (batch, 1, nz, nx)
        data_range = self.data_range
        if data_range is None:
            data_range = target.max() - target.min()

        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float("inf"), device=pred.device)
        return 10 * torch.log10((data_range ** 2) / mse)