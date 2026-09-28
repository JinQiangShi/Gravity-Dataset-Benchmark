import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(mid_channels, eps=1e-5, affine=True, momentum=0.1),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(out_channels, eps=1e-5, affine=True, momentum=0.1),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

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

class GravityInverseUNetPlusPlus2D(nn.Module):
    def __init__(self, in_channels, out_channels, linear=False, deep_supervision=False):
        """
        2D gravity inverse UNet++ model with deep supervision

        Parameters:
        -----------
            in_channels: input channels number, defined by gravity data channels
            out_channels: output channels number, defined by density model nz
            linear: use linear upsample or not
            deep_supervision: use deep supervision or not
        """
        super(GravityInverseUNetPlusPlus2D, self).__init__()

        # the channel number of feature map
        feature_map_nums = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision

        self.down = nn.MaxPool1d(kernel_size=2, stride=2)
        
        if linear:
            self.up4 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up4 = nn.ConvTranspose1d(feature_map_nums[4], feature_map_nums[4], kernel_size=2, stride=2, bias=True)
            self.up3 = nn.ConvTranspose1d(feature_map_nums[3], feature_map_nums[3], kernel_size=2, stride=2, bias=True)
            self.up2 = nn.ConvTranspose1d(feature_map_nums[2], feature_map_nums[2], kernel_size=2, stride=2, bias=True)
            self.up1 = nn.ConvTranspose1d(feature_map_nums[1], feature_map_nums[1], kernel_size=2, stride=2, bias=True)
        
        # {depth}_{stage}
        self.conv0_0 = DoubleConv(in_channels, feature_map_nums[0], feature_map_nums[0])
        self.conv1_0 = DoubleConv(feature_map_nums[0], feature_map_nums[1], feature_map_nums[1])
        self.conv2_0 = DoubleConv(feature_map_nums[1], feature_map_nums[2], feature_map_nums[2])
        self.conv3_0 = DoubleConv(feature_map_nums[2], feature_map_nums[3], feature_map_nums[3])
        self.conv4_0 = DoubleConv(feature_map_nums[3], feature_map_nums[4], feature_map_nums[4])

        self.conv0_1 = DoubleConv(feature_map_nums[0]+feature_map_nums[1], feature_map_nums[0], feature_map_nums[0])
        self.conv1_1 = DoubleConv(feature_map_nums[1]+feature_map_nums[2], feature_map_nums[1], feature_map_nums[1])
        self.conv2_1 = DoubleConv(feature_map_nums[2]+feature_map_nums[3], feature_map_nums[2], feature_map_nums[2])
        self.conv3_1 = DoubleConv(feature_map_nums[3]+feature_map_nums[4], feature_map_nums[3], feature_map_nums[3])

        self.conv0_2 = DoubleConv(feature_map_nums[0]*2+feature_map_nums[1], feature_map_nums[0], feature_map_nums[0])
        self.conv1_2 = DoubleConv(feature_map_nums[1]*2+feature_map_nums[2], feature_map_nums[1], feature_map_nums[1])
        self.conv2_2 = DoubleConv(feature_map_nums[2]*2+feature_map_nums[3], feature_map_nums[2], feature_map_nums[2])

        self.conv0_3 = DoubleConv(feature_map_nums[0]*3+feature_map_nums[1], feature_map_nums[0], feature_map_nums[0])
        self.conv1_3 = DoubleConv(feature_map_nums[1]*3+feature_map_nums[2], feature_map_nums[1], feature_map_nums[1])

        self.conv0_4 = DoubleConv(feature_map_nums[0]*4+feature_map_nums[1], feature_map_nums[0], feature_map_nums[0])

        if self.deep_supervision:
            self.final1 = OutConv(feature_map_nums[0], out_channels)
            self.final2 = OutConv(feature_map_nums[0], out_channels)
            self.final3 = OutConv(feature_map_nums[0], out_channels)
            self.final4 = OutConv(feature_map_nums[0], out_channels)
        else:
            self.final = OutConv(feature_map_nums[0], out_channels)

    def forward(self, x):
        # x shape (batch, 2, nx)
        x0_0 = self.conv0_0(x) # (batch, 64, nx)
        x1_0 = self.conv1_0(self.down(x0_0)) # (batch, 128, nx//2)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1(x1_0)], 1)) # (batch, 64, nx)

        x2_0 = self.conv2_0(self.down(x1_0)) # (batch, 256, nx//4)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2(x2_0)], 1)) # (batch, 128, nx//2)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1(x1_1)], 1)) # (batch, 64, nx)

        x3_0 = self.conv3_0(self.down(x2_0)) # (batch, 512, nx//8)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3(x3_0)], 1)) # (batch, 256, nx//4)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2(x2_1)], 1)) # (batch, 128, nx//2)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1(x1_2)], 1)) # (batch, 64, nx)

        x4_0 = self.conv4_0(self.down(x3_0)) # (batch, 1024, nx//16)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4(x4_0)], 1)) # (batch, 512, nx//8)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3(x3_1)], 1)) # (batch, 256, nx//4)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2(x2_2)], 1)) # (batch, 128, nx//2)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1(x1_3)], 1)) # (batch, 64, nx)

        if self.deep_supervision:
            output1 = self.final1(x0_1) # (batch, 1, out_channels, nx) <=> (batch, 1, nz, nx)
            output2 = self.final2(x0_2) # (batch, 1, out_channels, nx)
            output3 = self.final3(x0_3) # (batch, 1, out_channels, nx)
            output4 = self.final4(x0_4) # (batch, 1, out_channels, nx)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4) # (batch, 1, out_channels, nx)
            return output
       