import torch
import torch.nn as nn

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            # pre-activation
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn1d = nn.BatchNorm1d(out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv2d = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.softplus = nn.Softplus()
        # softplus activation motivated by
        # Deep Learning-Based 3D Gravity Inversion: A Comparative Analysis of CNN Architectures for Density Estimation

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn1d(x)
        x = self.leakyrelu(x)
        x = x.unsqueeze(1) # (batch, nz, nx) -> (batch, 1, nz, nx)
        x = self.conv2d(x)
        x = self.softplus(x)
        return x


class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels, linear=False):
        super(ResUnet, self).__init__()
        feature_map_nums=[64, 128, 256, 512]

        self.input_layer = nn.Sequential(
            nn.Conv1d(in_channels, feature_map_nums[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_map_nums[0]),
            nn.ReLU(),
            nn.Conv1d(feature_map_nums[0], feature_map_nums[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv1d(in_channels, feature_map_nums[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(feature_map_nums[0], feature_map_nums[1], 2, 1)
        self.residual_conv_2 = ResidualConv(feature_map_nums[1], feature_map_nums[2], 2, 1)

        self.bridge = ResidualConv(feature_map_nums[2], feature_map_nums[3], 2, 1)

        self.upsample_1 = Upsample(feature_map_nums[3], feature_map_nums[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(feature_map_nums[3] + feature_map_nums[2], feature_map_nums[2], 1, 1)

        self.upsample_2 = Upsample(feature_map_nums[2], feature_map_nums[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(feature_map_nums[2] + feature_map_nums[1], feature_map_nums[1], 1, 1)

        self.upsample_3 = Upsample(feature_map_nums[1], feature_map_nums[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(feature_map_nums[1] + feature_map_nums[0], feature_map_nums[0], 1, 1)

        self.output_layer = OutConv(feature_map_nums[0], out_channels)

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output