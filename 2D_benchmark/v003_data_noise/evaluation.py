import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = None) -> torch.Tensor:
    """
    计算峰值信噪比 (PSNR)。

    Params:
    -----
        pred: 预测张量
        target: 真值张量
        data_range: 数据范围，若为 None 则自动从 target 计算

    Returns:
    --------
        PSNR 值（标量）
    """
    if data_range is None:
        data_range = target.max() - target.min()
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 10 * torch.log10((data_range ** 2) / mse)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = None,
    window_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    计算结构相似性 (SSIM)，支持 2D 输入 [B, C, H, W]。

    Params:
    -----
        pred: 预测张量 [B, C, H, W]
        target: 真值张量 [B, C, H, W]
        data_range: 数据范围，若为 None 则自动从 target 计算
        window_size: 高斯窗口大小
        k1, k2: 稳定性常数

    Returns:
    --------
        平均 SSIM 值（标量）
    """
    if data_range is None:
        data_range = target.max() - target.min()

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # 高斯核
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)  # [ws, ws]
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]

    channels = pred.shape[1]
    window = window.expand(channels, 1, -1, -1)
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
