import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, linear=True):
        super().__init__()
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        upsample feature map and concat with downsample feature map

        Params:
        -----
            x1: upsampled feature map
            x2: downsampled feature map
        """
        x1 = self.up(x1) # 上采样
        x1 = self._align(x1, x2) # 对齐
        x = torch.cat([x2, x1], dim=1) # 拼接
        return self.conv(x) # 卷积层

    def _align(self, x1, x2):
        """
        align upsampled feature map with downsampled feature map by padding

        Params:
        -----
            x1: upsampled feature map
            x2: downsampled feature map
        """
        diff_l = x2.size(2) - x1.size(2)
        pad_left = diff_l // 2
        pad_right = diff_l - pad_left
        x1 = F.pad(
            x1,
            [pad_left, pad_right],
            mode="replicate"
        )
        return x1


class Gravity_Inverse_UNet_2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=126, linear=False):
        """
        2D gravity inverse UNet model

        Params:
        -----
            in_channels: input channels number, defined by gravity data
            out_channels: output channels number, defined by density data
            linear: use linear upsample
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = linear
        
        self.in_block = ConvBlock(in_channels, 64) # input block
        self.down1 = DownBlock(64, 128) # first downsample block
        self.down2 = DownBlock(128, 256) # second downsample block
        self.down3 = DownBlock(256, 512) # third downsample block
        factor = 2 if linear else 1 # sample factor
        self.down4 = DownBlock(512, 1024 // factor) # fourth downsample block
        self.up1 = UpBlock(1024, 512 // factor, linear) # first upsample block
        self.up2 = UpBlock(512, 256 // factor, linear) # second upsample block
        self.up3 = UpBlock(256, 128 // factor, linear) # third upsample block
        self.up4 = UpBlock(128, out_channels, linear) # fourth upsample block
        self.out_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        # x.shape: [batch_size, in_channels, nx]
        x1 = self.in_block(x) # x1.shape: [batch_size, 64, nx]

        x2 = self.down1(x1) # x2.shape: [batch_size, 128, nx//2]
        x3 = self.down2(x2) # x3.shape: [batch_size, 256, nx//4]
        x4 = self.down3(x3) # x4.shape: [batch_size, 512, nx//8]
        x5 = self.down4(x4) # x5.shape: [batch_size, 1024, nx//16]

        x = self.up1(x5, x4) # x.shape: [batch_size, 512, nx//8]
        x = self.up2(x, x3) # x.shape: [batch_size, 256, nx//4]
        x = self.up3(x, x2) # x.shape: [batch_size, 128, nx//2]
        x = self.up4(x, x1) # x.shape: [batch_size, out_channels, nx]

        x = x.unsqueeze(1) # [batch_size, out_channels, nx] -> [batch_size, 1, nz, nx]
        output = self.out_block(x) # [batch_size, 1, nz, nx]

        return output
