import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, pred):
        # Calculate gradients along x and y directions using L2 norm
        diff_i = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # horizontal differences
        diff_j = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # vertical differences
        
        # Compute L2 norm of gradients separately for each direction
        tv_loss = torch.sum(torch.sqrt(diff_i**2 + self.epsilon)) + torch.sum(torch.sqrt(diff_j**2 + self.epsilon))
        
        return tv_loss / pred.numel()

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, tv_weight=0.01):
        super(CombinedLoss, self).__init__()
        self.mse_loss = MSELoss()
        self.mse_weight = mse_weight

        self.tv_loss = TotalVariationLoss()
        self.tv_weight = tv_weight

    def forward(self, pred, target):
        total_loss = 0.0
        total_loss += self.mse_weight * self.mse_loss(pred, target)
        total_loss += self.tv_weight * self.tv_loss(pred)
        return total_loss