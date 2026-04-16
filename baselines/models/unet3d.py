"""
3D U-Net for CFD WSS prediction.

Standard encoder-decoder architecture with skip connections.
Input: voxelized velocity + pressure fields (9 channels)
Output: WSS magnitude field (1 channel)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Two 3D conv layers with BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Down3D(nn.Module):
    """Downsampling: MaxPool + DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up3D(nn.Module):
    """Upsampling: Upsample + concat skip + DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from non-power-of-2 inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D_WSS(nn.Module):
    """
    3D U-Net for predicting WSS from voxelized flow fields.

    Architecture (for resolution 48 or 64):
        Encoder: 9 -> 32 -> 64 -> 128 -> 256
        Bottleneck: 256
        Decoder: 256 -> 128 -> 64 -> 32
        Output: 32 -> 1
    """

    def __init__(self,
                 in_channels: int = 9,
                 base_channels: int = 32,
                 depth: int = 4):
        super().__init__()

        channels = [base_channels * (2 ** i) for i in range(depth)]
        # channels = [32, 64, 128, 256]

        # Encoder
        self.inc = DoubleConv3D(in_channels, channels[0])
        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(Down3D(channels[i-1], channels[i]))

        # Decoder
        self.ups = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.ups.append(Up3D(channels[i] + channels[i-1], channels[i-1]))

        # Output
        self.outc = nn.Conv3d(channels[0], 1, 1)

    def forward(self, x):
        """
        Args:
            x: [B, C_in, D, H, W]
        Returns:
            y: [B, 1, D, H, W]
        """
        # Encoder
        skips = []
        h = self.inc(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)

        # Decoder (skip last element which is the bottleneck)
        h = skips.pop()
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip)

        return self.outc(h)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = UNet3D_WSS(in_channels=9, base_channels=32, depth=4)
    print(f"UNet3D_WSS parameters: {count_parameters(model):,}")

    x = torch.randn(2, 9, 48, 48, 48)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
