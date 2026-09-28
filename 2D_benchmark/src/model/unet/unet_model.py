from .unet_parts import *

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
