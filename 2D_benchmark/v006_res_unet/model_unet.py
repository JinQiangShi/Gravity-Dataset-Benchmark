import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + res
        out = self.relu(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv = ResConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None, linear=True):
        super().__init__()
        if skip_channels is None:
            skip_channels = out_channels

        if linear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_channels = in_channels // 2

        if up_channels != skip_channels:
            self.proj = nn.Sequential(
                nn.Conv1d(up_channels, skip_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(skip_channels),
            )
        else:
            self.proj = nn.Identity()

        self.conv = ResConvBlock(skip_channels, out_channels)

    def forward(self, x1, x2):
        """
        upsample feature map and add with downsampled feature map (residual skip)

        Params:
        -----
            x1: upsampled feature map
            x2: downsampled feature map (skip connection)
        """
        x1 = self.up(x1) # 上采样
        x1 = self._align(x1, x2) # 对齐
        x1 = self.proj(x1) # 投影以匹配通道数
        x = x1 + x2 # 残差相加
        return self.conv(x)

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


class ResOutBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: 1 -> 32 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.residual1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
        )

        # Block 2: 32 -> 16 -> 16
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.residual2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
        )

        # Final
        self.conv5 = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Block 1
        res = self.residual1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + res
        out = self.relu(out)

        # Block 2
        res = self.residual2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = out + res
        out = self.relu(out)

        # Final
        out = self.conv5(out)
        out = self.sigmoid(out)
        return out


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
        
        self.in_block = ResConvBlock(in_channels, 64) # input block
        self.down1 = DownBlock(64, 128) # first downsample block
        self.down2 = DownBlock(128, 256) # second downsample block
        self.down3 = DownBlock(256, 512) # third downsample block
        factor = 2 if linear else 1 # sample factor
        self.down4 = DownBlock(512, 1024 // factor) # fourth downsample block
        self.up1 = UpBlock(1024 // factor, 512 // factor, 512, linear) # first upsample block
        self.up2 = UpBlock(512 // factor, 256 // factor, 256, linear) # second upsample block
        self.up3 = UpBlock(256 // factor, 128 // factor, 128, linear) # third upsample block
        self.up4 = UpBlock(128 // factor, out_channels, 64, linear) # fourth upsample block
        self.out_block = ResOutBlock()

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
