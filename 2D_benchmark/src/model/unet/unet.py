import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, linear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: shape(batch, channel, nx1), the decoder up feature
        x2: shape(batch, channel, nx2), the encoder down feature
        """
        x1 = self.up(x1)

        # align the feature maps in size before concatenation
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        diffX = x2.size()[2] - x1.size()[2] # 2 means nx dimension
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.out_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.Softplus()
            # softplus activation motivated by
            # Deep Learning-Based 3D Gravity Inversion: A Comparative Analysis of CNN Architectures for Density Estimation
        )

    def forward(self, x):
        return self.out_block(x)


class GravityInverseUNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, linear=False):
        """
        2D gravity inverse UNet model

        Parameters:
        -----------
            in_channels: input channels number, defined by gravity data channels
            out_channels: output channels number, defined by density model nz
            linear: use linear upsample or not
        """
        super(GravityInverseUNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = linear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if linear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, linear)
        self.up2 = Up(512, 256 // factor, linear)
        self.up3 = Up(256, 128 // factor, linear)
        self.up4 = Up(128, out_channels, linear)
        self.outc = OutConv(1, 1)

    def forward(self, x):
        # x shape: [batch_size, in_channels, nx]

        x1 = self.inc(x) # x1 shape: [batch_size, 64, nx]
        x2 = self.down1(x1) # x2 shape: [batch_size, 128, nx//2]
        x3 = self.down2(x2) # x3 shape: [batch_size, 256, nx//4]
        x4 = self.down3(x3) # x4 shape: [batch_size, 512, nx//8]
        x5 = self.down4(x4) # x5 shape: [batch_size, 1024, nx//16]

        x = self.up1(x5, x4) # x shape: [batch_size, 512, nx//8]
        x = self.up2(x, x3) # x shape: [batch_size, 256, nx//4]
        x = self.up3(x, x2) # x shape: [batch_size, 128, nx//2]
        x = self.up4(x, x1) # x shape: [batch_size, out_channels, nx]

        x = x.unsqueeze(1) # x shape: [batch_size, 1, out_channels, nx] <=> [batch_size, 1, nz, nx]
        output = self.outc(x) # output shape: [batch_size, 1, nz, nx]
        return output