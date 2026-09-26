import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=7, sigma=1.5, data_range=1.0, k1=0.01, k2=0.03):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.register_buffer("window", self._gaussian_window(window_size, sigma))

    @staticmethod
    def _gaussian_window(window_size, sigma):
        # 1D Gaussian, then outer product to build a 2D window
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g[:, None] * g[None, :]
        return window.view(1, 1, window_size, window_size)

    def forward(self, pred, target):
        # pred shape (batch, channels, nz, nx)
        # target shape (batch, channels, nz, nx)
        data_range = self.data_range
        if data_range is None:
            data_range = target.max() - target.min()

        channels = pred.shape[1]
        window = self.window.to(device=pred.device, dtype=pred.dtype).expand(channels, 1, -1, -1).contiguous()
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_target = F.conv2d(target, window, padding=pad, groups=channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross

        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2

        numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
        denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

        ssim_map = numerator / denominator

        return ssim_map.mean()