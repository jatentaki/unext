from typing import Callable, Optional

from torch import nn
from torch.nn import functional as F

class LayerNorm2d(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        **kwargs,
    ):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        kernel_size: int = 7,
        blowup: int = 2,
        skip_connection: bool = True,
        out_channels: Optional[int] = None,
        norm_fn: Callable[[int, bool], nn.Module] = LayerNorm2d,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = n_channels

        if skip_connection and out_channels != n_channels:
            raise AssertionError(f"cannot have skip connections with {out_channels=} != {n_channels}.")

        self.conv_wide = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            groups=n_channels
        )
        self.norm = norm_fn(
            n_channels,
            elementwise_affine=False,
        )
        self.conv_1x1_1 = nn.Conv2d(
            n_channels,
            blowup * n_channels,
            kernel_size=(1, 1),
        )
        self.activation = nn.GELU()
        self.conv_1x1_2 = nn.Conv2d(
            blowup * n_channels,
            out_channels,
            kernel_size=(1, 1),
        )
        self.skip_connection = skip_connection
    
    def forward(self, x):
        y = self.conv_wide(x)
        y = self.norm(y)
        y = self.conv_1x1_1(y)
        y = self.activation(y)
        y = self.conv_1x1_2(y)

        if self.skip_connection:
            return x + y
        else:
            return y

