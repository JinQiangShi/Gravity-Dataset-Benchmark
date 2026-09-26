import torch.nn as nn

from .DepthWeightedHuberLoss import DepthWeightedHuberLoss
from .HuberLoss import HuberLoss
from .MinimumGradientSupportLoss import MinimumGradientSupportLoss
from .MinimumSupportLoss import MinimumSupportLoss
from .SSIMLoss import SSIMLoss
from .TotalVariationLoss import TotalVariationLoss


class CombinedLoss(nn.Module):
    def __init__(
        self, 
        huber_weight: float = 1.0, 
        depth_weight: float = 1.0, 
        ssim_weight: float = 0.01,
        tv_weight: float = 0.01, 
        ms_weight: float = 0.01, 
        mgs_weight: float = 0.01, 
        data_range: float = 1.0,
    ):
        super(CombinedLoss, self).__init__()
        self.data_range = data_range
        self.huber_loss = HuberLoss()
        self.huber_weight = huber_weight
        self.total_huber_loss_value = 0.0
        self.avg_huber_loss_value = 0.0

        self.depth_loss = DepthWeightedHuberLoss()
        self.depth_weight = depth_weight
        self.total_depth_loss_value = 0.0
        self.avg_depth_loss_value = 0.0

        self.ssim_loss = SSIMLoss(data_range=data_range)
        self.ssim_weight = ssim_weight
        self.total_ssim_loss_value = 0.0
        self.avg_ssim_loss_value = 0.0

        self.tv_loss = TotalVariationLoss()
        self.tv_weight = tv_weight
        self.total_tv_loss_value = 0.0
        self.avg_tv_loss_value = 0.0

        self.ms_loss = MinimumSupportLoss()
        self.ms_weight = ms_weight
        self.total_ms_loss_value = 0.0
        self.avg_ms_loss_value = 0.0

        self.mgs_loss = MinimumGradientSupportLoss()
        self.mgs_weight = mgs_weight
        self.total_mgs_loss_value = 0.0
        self.avg_mgs_loss_value = 0.0

        self.total_loss_value = 0.0
        self.avg_loss_value = 0.0

        self.batch_num = 0

    def copy(self) -> "CombinedLoss":
        return CombinedLoss(
            huber_weight=self.huber_weight,
            depth_weight=self.depth_weight,
            ssim_weight=self.ssim_weight,
            tv_weight=self.tv_weight,
            ms_weight=self.ms_weight,
            mgs_weight=self.mgs_weight,
            data_range=self.data_range,
        )

    def forward(self, pred, target):
        # calculate each loss component
        huber_loss_value = self.huber_loss(pred, target)
        depth_loss_value = self.depth_loss(pred, target)
        ssim_loss_value = 1.0 - self.ssim_loss(pred, target)
        tv_loss_value = self.tv_loss(pred)
        ms_loss_value = self.ms_loss(pred)
        mgs_loss_value = self.mgs_loss(pred)
        # calculate total loss
        loss = self.huber_weight * huber_loss_value
        loss += self.depth_weight * depth_loss_value
        loss += self.ssim_weight * ssim_loss_value
        loss += self.tv_weight * tv_loss_value
        loss += self.ms_weight * ms_loss_value
        loss += self.mgs_weight * mgs_loss_value
        # update class attributes
        self.total_loss_value += loss.item()
        self.total_huber_loss_value += huber_loss_value.item()
        self.total_depth_loss_value += depth_loss_value.item()
        self.total_ssim_loss_value += ssim_loss_value.item()
        self.total_tv_loss_value += tv_loss_value.item()
        self.total_ms_loss_value += ms_loss_value.item()
        self.total_mgs_loss_value += mgs_loss_value.item()
        # update batch number
        self.batch_num += 1
        return loss

    def update_avg(self):
        if self.batch_num == 0:
            return
        self.avg_loss_value = self.total_loss_value / self.batch_num
        self.avg_huber_loss_value = self.total_huber_loss_value / self.batch_num
        self.avg_depth_loss_value = self.total_depth_loss_value / self.batch_num
        self.avg_ssim_loss_value = self.total_ssim_loss_value / self.batch_num
        self.avg_tv_loss_value = self.total_tv_loss_value / self.batch_num
        self.avg_ms_loss_value = self.total_ms_loss_value / self.batch_num
        self.avg_mgs_loss_value = self.total_mgs_loss_value / self.batch_num

    def value_dict(self, prefix: str = "train") -> dict:
        self.update_avg()
        return {
            f"{prefix}/loss/total": self.avg_loss_value,
            f"{prefix}/loss/huber": self.avg_huber_loss_value,
            f"{prefix}/loss/depth": self.avg_depth_loss_value,
            f"{prefix}/loss/ssim": self.avg_ssim_loss_value,
            f"{prefix}/loss/tv": self.avg_tv_loss_value,
            f"{prefix}/loss/ms": self.avg_ms_loss_value,
            f"{prefix}/loss/mgs": self.avg_mgs_loss_value,
        }


    def reset(self):
        self.total_huber_loss_value = 0.0
        self.avg_huber_loss_value = 0.0

        self.total_depth_loss_value = 0.0
        self.avg_depth_loss_value = 0.0
        
        self.total_ssim_loss_value = 0.0
        self.avg_ssim_loss_value = 0.0
        
        self.total_tv_loss_value = 0.0
        self.avg_tv_loss_value = 0.0
        
        self.total_ms_loss_value = 0.0
        self.avg_ms_loss_value = 0.0
        
        self.total_mgs_loss_value = 0.0
        self.avg_mgs_loss_value = 0.0
        
        self.total_loss_value = 0.0
        self.avg_loss_value = 0.0
        
        self.batch_num = 0