from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# Number of stride 2 convolutions needed to reach a 4 by 4 feature map.
_NUM_STAGES: dict[int, int] = {64: 4, 128: 5, 256: 6}


def init_weights(module: nn.Module) -> None:
    """DCGAN style weight initialization."""
    classname = module.__class__.__name__
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if "Conv" in classname:
        if isinstance(weight, torch.Tensor):
            nn.init.normal_(weight, mean=0.0, std=0.02)
        if isinstance(bias, torch.Tensor):
            nn.init.zeros_(bias)
    elif "BatchNorm" in classname:
        if isinstance(weight, torch.Tensor):
            nn.init.normal_(weight, mean=1.0, std=0.02)
        if isinstance(bias, torch.Tensor):
            nn.init.zeros_(bias)


class Generator(nn.Module):
    """DCGAN generator mapping a latent vector to an RGB square image."""

    def __init__(
        self,
        latent_dim: int = 128,
        channels: int = 3,
        image_size: int = 128,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if image_size not in _NUM_STAGES:
            raise ValueError(f"image_size must be one of {sorted(_NUM_STAGES)}")

        self.latent_dim = latent_dim
        self.image_size = image_size

        num_stages = _NUM_STAGES[image_size]
        max_channels = base_channels * 8
        start_channels = max_channels

        # Project the 1 by 1 latent to a 4 by 4 feature map.
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                latent_dim,
                start_channels,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(start_channels),
            nn.ReLU(inplace=True),
        ]

        # Repeatedly upsample by 2x until the second to last stage.
        in_ch = start_channels
        for _ in range(num_stages - 1):
            out_ch = max(in_ch // 2, base_channels)
            layers += [
                nn.ConvTranspose2d(
                    in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        # Final upsample produces the RGB image in [neg one, one] via Tanh.
        layers += [
            nn.ConvTranspose2d(
                in_ch, channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)


class Discriminator(nn.Module):
    """Spectral normalized discriminator. Outputs raw scores for hinge loss."""

    def __init__(
        self,
        channels: int = 3,
        image_size: int = 128,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if image_size not in _NUM_STAGES:
            raise ValueError(f"image_size must be one of {sorted(_NUM_STAGES)}")

        num_stages = _NUM_STAGES[image_size]
        max_channels = base_channels * 8

        def sn_conv(in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> nn.Module:
            return spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False)
            )

        # First downsample skips normalization, following DCGAN convention.
        layers: list[nn.Module] = [
            sn_conv(channels, base_channels, kernel=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_ch = base_channels
        for _ in range(num_stages - 1):
            out_ch = min(in_ch * 2, max_channels)
            layers += [
                sn_conv(in_ch, out_ch, kernel=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch

        # Final 4 by 4 conv collapses the 4 by 4 map to a single logit per image.
        layers.append(sn_conv(in_ch, 1, kernel=4, stride=1, padding=0))

        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0))
