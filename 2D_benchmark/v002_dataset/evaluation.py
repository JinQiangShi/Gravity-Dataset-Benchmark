import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = None) -> torch.Tensor:
    """
    peak signal to noise ratio (PSNR)
    """
    if data_range is None:
        data_range = target.max() - target.min()
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 10 * torch.log10((data_range ** 2) / mse)


_gaussian_window_cache: dict[tuple, torch.Tensor] = {}


def _get_gaussian_window(window_size: int, channels: int, device: torch.device) -> torch.Tensor:
    """获取缓存的高斯卷积核，避免重复计算。"""
    key = (window_size, channels, device)
    if key not in _gaussian_window_cache:
        coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)  # [ws, ws]
        window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]
        window = window.expand(channels, 1, -1, -1).contiguous()
        _gaussian_window_cache[key] = window
    return _gaussian_window_cache[key]


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = None,
        window_size: int = 3, k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    """
    structural similarity index (SSIM) for 2D images [B, C, H, W].
    """
    if data_range is None:
        data_range = target.max() - target.min()

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    channels = pred.shape[1]
    window = _get_gaussian_window(window_size, channels, pred.device)
    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross

    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

    ssim_map = numerator / denominator
    return ssim_map.mean()


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    mean absolute error (MAE)
    """
    return F.l1_loss(pred, target)