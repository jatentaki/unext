from typing import Callable, List

from torch import nn

from .blocks import ConvNeXtBlock

class DownStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        block: Callable[[int, ], nn.Module] = ConvNeXtBlock,
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[block(in_channels) for _ in range(n_blocks)])
        self.downsample = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(2, 2),
            stride=(2, 2),
        )
    
    def forward(self, x):
        y = self.blocks(x)
        return y, self.downsample(y)

class UpStage(nn.Module):
    def __init__(
        self,
        low_res_channels: int,
        high_res_channels: int,
        n_blocks: int = 2,
        block: Callable[[int, ], nn.Module] = ConvNeXtBlock,
    ):
        super().__init__()

        self.low_res_channels = low_res_channels
        self.high_res_channels = high_res_channels

        self.upsample = nn.ConvTranspose2d(
            low_res_channels,
            high_res_channels,
            kernel_size=(2, 2),
            stride=(2, 2)
        )

        self.blocks = nn.Sequential(*[block(high_res_channels) for _ in range(n_blocks)])
    
    def forward(self, low_res, high_res):
        # skip connection features are as expected
        assert low_res.shape[1] == self.low_res_channels
        assert high_res.shape[1] == self.high_res_channels

        # upsampling by 2x
        assert high_res.shape[2] == 2 * low_res.shape[2]
        assert high_res.shape[3] == 2 * low_res.shape[3]

        y = self.upsample(low_res) + high_res
        return self.blocks(y)

class UNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        widths: List[int],
        n_blocks: int = 2,
        block: Callable[[int, ], nn.Module] = ConvNeXtBlock,
    ):
        super().__init__()
        # mapping from input dimensionality to 1st stages' dimensionality
        self.entry = nn.Conv2d(in_channels, widths[0], kernel_size=(1, 1))

        # downsampling stages
        self.down_stages = nn.ModuleList()
        for in_width, out_width in zip(widths[:-1], widths[1:]):
            self.down_stages.append(DownStage(
                in_width,
                out_width,
                n_blocks=n_blocks,
                block=block,
            ))
        
        # upsampling stages
        w_rev = list(reversed(widths))
        self.up_stages = nn.ModuleList()
        for low_res_w, high_res_w in zip(w_rev[:-1], w_rev[1:]):
            self.up_stages.append(UpStage(
                low_res_channels=low_res_w,
                high_res_channels=high_res_w,
                n_blocks=n_blocks,
                block=block,
            ))

        # a final block to map from 1st (and thus also last) stage dimensionality
        # to output dimensionality.
        # Because ConvNeXtBlock uses pre-norm formulation, the activations at this
        # point can be very large, therefore we need to use at least norm + linear.
        # In order to give the network a bit more expressive power, and simplify
        # implementation we reuse the code already present in ConvNeXtBlock instead.
        # Unlike self.up_stages[-1], this one does not use a skip-connection, since we
        # want the outputs to be completely free-form. alternatively, we could
        # add some special logic in creation of self.up_stages to make the last
        # blocko of the last stage have skip_connection=False
        self.exit_block = ConvNeXtBlock(
            widths[0],
            skip_connection=False,
            out_channels=out_channels,
        )

    def forward(self, x):
        y = self.entry(x)
        latents = []
        for stage in self.down_stages:
            latent, y = stage(y)
            latents.append(latent)
        
        for stage in self.up_stages:
            low = latents.pop()
            y = stage(y, low)

        return self.exit_block(y)
