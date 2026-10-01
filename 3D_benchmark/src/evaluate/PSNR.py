import torch
import torch.nn as nn
import torch.nn.functional as F


class PSNR(nn.Module):
    def __init__(self, data_range=1.0, max_psnr=100.0):
        super(PSNR, self).__init__()
        self.data_range = data_range
        self.max_psnr = max_psnr

    def forward(self, pred, target):
        # pred shape (batch, 1, nz, ny, nx)
        # target shape (batch, 1, nz, ny, nx)
        data_range = self.data_range
        if data_range is None:
            data_range = target.max() - target.min()

        mse = F.mse_loss(pred, target)
        if mse == 0:
            # perfect reconstruction: return a large finite value instead of inf
            return torch.tensor(float(self.max_psnr), device=pred.device)
        psnr = 10 * torch.log10((data_range ** 2) / mse)
        return torch.clamp(psnr, max=self.max_psnr)