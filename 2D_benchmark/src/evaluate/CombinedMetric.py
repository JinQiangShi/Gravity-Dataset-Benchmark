import torch.nn as nn

from .MAE import MAE
from .PSNR import PSNR
from .SSIM import SSIM


class CombinedMetric(nn.Module):
    def __init__(
        self,
        mae_weight: float = 0.4,
        psnr_weight: float = 0.4,
        ssim_weight: float = 0.2,
        mae_scale: float = 1.0,
        psnr_scale: float = 100.0,
    ):
        super(CombinedMetric, self).__init__()
        self.mae_weight = mae_weight
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        self.mae_scale = mae_scale
        self.psnr_scale = psnr_scale

        self.mae_metric = MAE()
        self.total_mae_value = 0.0
        self.avg_mae_value = 0.0

        self.psnr_metric = PSNR()
        self.total_psnr_value = 0.0
        self.avg_psnr_value = 0.0

        self.ssim_metric = SSIM()
        self.total_ssim_value = 0.0
        self.avg_ssim_value = 0.0

        self.best_metric_value = None

        self.batch_num = 0

    @classmethod
    def copy(cls, other: "CombinedMetric") -> "CombinedMetric":
        return cls(
            mae_weight=other.mae_weight,
            psnr_weight=other.psnr_weight,
            ssim_weight=other.ssim_weight,
            mae_scale=other.mae_scale,
            psnr_scale=other.psnr_scale,
        )

    def forward(self, pred, target):
        mae_value = self.mae_metric(pred, target)
        psnr_value = self.psnr_metric(pred, target)
        ssim_value = self.ssim_metric(pred, target)
        # update class attributes
        self.total_mae_value += mae_value.item()
        self.total_psnr_value += psnr_value.item()
        self.total_ssim_value += ssim_value.item()
        # update batch number
        self.batch_num += 1

    def update_avg(self):
        self.avg_mae_value = self.total_mae_value / self.batch_num
        self.avg_psnr_value = self.total_psnr_value / self.batch_num
        self.avg_ssim_value = self.total_ssim_value / self.batch_num

    def metric(self) -> float:
        return (
            self.mae_weight * (self.avg_mae_value / self.mae_scale)
            - self.psnr_weight * (self.avg_psnr_value / self.psnr_scale)
            - self.ssim_weight * self.avg_ssim_value
        )

    def is_better(self) -> bool:
        self.update_avg()
        if self.best_metric_value is None:
            return True
        return self.metric() < self.best_metric_value

    def update_best(self):
        self.update_avg()
        self.best_metric_value = self.metric()

    def value_dict(self, prefix: str = "val") -> dict:
        self.update_avg()
        return {
            f"{prefix}_mae": self.avg_mae_value,
            f"{prefix}_psnr": self.avg_psnr_value,
            f"{prefix}_ssim": self.avg_ssim_value,
            f"{prefix}_metric": self.metric(),
        }

    def reset(self):
        self.total_mae_value = 0.0
        self.avg_mae_value = 0.0

        self.total_psnr_value = 0.0
        self.avg_psnr_value = 0.0

        self.total_ssim_value = 0.0
        self.avg_ssim_value = 0.0

        self.batch_num = 0