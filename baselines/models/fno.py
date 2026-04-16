"""
3D Fourier Neural Operator (FNO) for CFD WSS prediction.

Uses the neuralop library's FNO with 3D spectral convolutions.
Input: voxelized velocity + pressure fields (9 channels)
Output: WSS magnitude field (1 channel)
"""
import torch
import torch.nn as nn
from neuralop.models import FNO


class FNO3D_WSS(nn.Module):
    """
    3D FNO for predicting WSS from voxelized flow fields.

    Architecture:
        - Lifting: in_channels -> hidden_channels
        - N Fourier layers with spectral convolutions
        - Projection: hidden_channels -> 1 (WSS magnitude)
    """

    def __init__(self,
                 in_channels: int = 9,
                 hidden_channels: int = 32,
                 n_modes: tuple = (16, 16, 16),
                 n_layers: int = 4,
                 use_mlp: bool = True,
                 mlp_expansion: float = 0.5,
                 norm: str = None):
        super().__init__()

        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            lifting_channels=hidden_channels * 2,
            projection_channels=hidden_channels * 2,
            n_layers=n_layers,
            use_mlp=use_mlp,
            mlp_expansion=mlp_expansion,
            norm=norm,
            non_linearity=torch.nn.functional.gelu,
            fno_skip='linear',
        )

    def forward(self, x):
        """
        Args:
            x: [B, C_in, D, H, W] voxelized input
        Returns:
            y: [B, 1, D, H, W] predicted WSS field
        """
        return self.fno(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = FNO3D_WSS(in_channels=9, hidden_channels=32, n_modes=(12, 12, 12), n_layers=4)
    print(f"FNO3D_WSS parameters: {count_parameters(model):,}")

    x = torch.randn(2, 9, 48, 48, 48)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
