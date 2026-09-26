import torch.nn as nn
import torch.nn.functional as F


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, pred, target):
        # pred shape (batch, 1, nz, nx)
        # target shape (batch, 1, nz, nx)
        return F.l1_loss(pred, target)