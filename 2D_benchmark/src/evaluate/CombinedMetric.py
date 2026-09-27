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
        psnr_max: float = 20.0,
        data_range: float = 1.0,
    ):
        super(CombinedMetric, self).__init__()
        self.mae_weight = mae_weight
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        self.psnr_max = psnr_max
        self.data_range = data_range

        self.mae_metric = MAE()
        self.total_mae_value = 0.0
        self.avg_mae_value = 0.0

        self.psnr_metric = PSNR(data_range=data_range)
        self.total_psnr_value = 0.0
        self.avg_psnr_value = 0.0

        self.ssim_metric = SSIM(data_range=data_range)
        self.total_ssim_value = 0.0
        self.avg_ssim_value = 0.0

        self.best_metric_value = None

        self.batch_num = 0

    def copy(self) -> "CombinedMetric":
        return CombinedMetric(
            mae_weight=self.mae_weight,
            psnr_weight=self.psnr_weight,
            ssim_weight=self.ssim_weight,
            psnr_max=self.psnr_max,
            data_range=self.data_range,
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
        self.update_avg()
        return (
            self.mae_weight * self.avg_mae_value
            + self.psnr_weight * (1.0 - self.avg_psnr_value / self.psnr_max)
            + self.ssim_weight * (1.0 - self.avg_ssim_value)
        )

    def is_better(self) -> bool:
        if self.best_metric_value is None:
            return True
        return self.metric() < self.best_metric_value

    def update_best(self):
        self.best_metric_value = self.metric()

    def value_dict(self, prefix: str = "val") -> dict:
        self.update_avg()
        return {
            f"{prefix}/metric/mae": self.avg_mae_value,
            f"{prefix}/metric/psnr": self.avg_psnr_value,
            f"{prefix}/metric/ssim": self.avg_ssim_value,
            f"{prefix}/metric/combined": self.metric(),
        }

    def reset(self):
        self.total_mae_value = 0.0
        self.avg_mae_value = 0.0

        self.total_psnr_value = 0.0
        self.avg_psnr_value = 0.0

        self.total_ssim_value = 0.0
        self.avg_ssim_value = 0.0

        self.batch_num = 0