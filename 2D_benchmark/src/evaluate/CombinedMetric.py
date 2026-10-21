import torch.nn as nn

from .MAE import MAE
from .PSNR import PSNR
from .SSIM import SSIM


class CombinedMetric(nn.Module):
    @staticmethod
    def _is_enabled(weight) -> bool:
        return weight is not None and float(weight) != 0.0

    @staticmethod
    def _normalize_weight(weight) -> float:
        return 0.0 if weight is None else float(weight)

    def __init__(
        self,
        mae_weight: float = 1.0,
        psnr_weight: float = None,
        ssim_weight: float = None,
        psnr_max: float = 20.0,
        data_range: float = 1.0,
    ):
        super(CombinedMetric, self).__init__()
        self.mae_weight = self._normalize_weight(mae_weight)
        self.psnr_weight = self._normalize_weight(psnr_weight)
        self.ssim_weight = self._normalize_weight(ssim_weight)
        self.psnr_max = psnr_max
        self.data_range = data_range

        self.mae_metric = MAE() if self._is_enabled(mae_weight) else None
        self.total_mae_value = 0.0
        self.avg_mae_value = 0.0

        self.psnr_metric = PSNR(data_range=data_range) if self._is_enabled(psnr_weight) else None
        self.total_psnr_value = 0.0
        self.avg_psnr_value = 0.0

        self.ssim_metric = SSIM(data_range=data_range) if self._is_enabled(ssim_weight) else None
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
        # calculate each enabled metric, skip disabled ones
        if self.mae_metric is not None:
            self.total_mae_value += self.mae_metric(pred, target).item()
        if self.psnr_metric is not None:
            self.total_psnr_value += self.psnr_metric(pred, target).item()
        if self.ssim_metric is not None:
            self.total_ssim_value += self.ssim_metric(pred, target).item()
        # update batch number
        self.batch_num += 1

    def update_avg(self):
        if self.batch_num == 0:
            return
        self.avg_mae_value = self.total_mae_value / self.batch_num
        self.avg_psnr_value = self.total_psnr_value / self.batch_num
        self.avg_ssim_value = self.total_ssim_value / self.batch_num

    def metric(self) -> float:
        self.update_avg()
        value = 0.0
        if self.mae_metric is not None:
            value += self.mae_weight * self.avg_mae_value
        if self.psnr_metric is not None:
            value += self.psnr_weight * (1.0 - self.avg_psnr_value / self.psnr_max)
        if self.ssim_metric is not None:
            value += self.ssim_weight * (1.0 - self.avg_ssim_value)
        return value

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