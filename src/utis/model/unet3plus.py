"""UNet 3+ (Huang et al., 2020) adapted for flux/background separation.

The network maps a grayscale target image ``I`` (``B x 1 x H x W``) to two
channels: the relative flux density ``X_r`` and the background ``B``
(see Eq. 4 of the paper). ``H`` and ``W`` must be divisible by 16.

The implementation is numerically identical to the original UNet 3+ code used
for the paper; only the module layout is expressed with loops instead of
~200 hand-written layer definitions. Checkpoints in the original layout can be
converted with :func:`utis.model.legacy.convert_legacy_state_dict`.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

NUM_LEVELS = 5


class ConvBNReLU(nn.Module):
    """3x3 convolution followed by batch normalisation and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DoubleConv(nn.Sequential):
    """Two :class:`ConvBNReLU` blocks, the building block of the encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(ConvBNReLU(in_channels, out_channels), ConvBNReLU(out_channels, out_channels))


class FullScaleSkip(nn.Module):
    """Brings a feature map of an arbitrary level to the resolution of a decoder stage.

    Finer levels are max-pooled, coarser levels are bilinearly upsampled and the
    same level is passed through. A :class:`ConvBNReLU`-like projection to
    ``out_channels`` follows.
    """

    def __init__(self, in_channels: int, out_channels: int, source_level: int, target_level: int) -> None:
        super().__init__()
        if source_level < target_level:
            factor = 2 ** (target_level - source_level)
            self.resample: nn.Module = nn.MaxPool2d(factor, factor, ceil_mode=True)
        elif source_level > target_level:
            self.resample = nn.Upsample(scale_factor=2 ** (source_level - target_level), mode="bilinear")
        else:
            self.resample = nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(self.resample(x))))


class DecoderStage(nn.Module):
    """Decoder stage that fuses the features of all five levels (full-scale skip connections)."""

    def __init__(self, level: int, encoder_channels: List[int], cat_channels: int) -> None:
        super().__init__()
        up_channels = cat_channels * NUM_LEVELS
        skips = []
        for source in range(1, NUM_LEVELS + 1):
            # Encoder output (h1..h4 and hd5) or output of a coarser decoder stage.
            from_encoder = source <= level or source == NUM_LEVELS
            in_channels = encoder_channels[source - 1] if from_encoder else up_channels
            skips.append(FullScaleSkip(in_channels, cat_channels, source, level))
        self.skips = nn.ModuleList(skips)
        self.fuse = ConvBNReLU(up_channels, up_channels)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        resampled: List[torch.Tensor] = []
        for i, skip in enumerate(self.skips):
            resampled.append(skip(features[i]))
        return self.fuse(torch.cat(resampled, dim=1))


class UNet3Plus(nn.Module):
    """UNet 3+ with five levels.

    Args:
        in_channels: Number of input channels (1 for grayscale target images).
        out_channels: Number of output channels (2: relative flux and background).
        base_channels: Feature channels of the first level. Level ``i`` uses
            ``base_channels * 2**i`` channels, e.g. ``[32, 64, 128, 256, 512]``
            for the default (as in the paper).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        filters = [base_channels * 2**i for i in range(NUM_LEVELS)]
        self.encoders = nn.ModuleList(
            DoubleConv(c_in, c_out) for c_in, c_out in zip([in_channels] + filters[:-1], filters)
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Decoder stages from coarse to fine: hd4, hd3, hd2, hd1.
        self.decoders = nn.ModuleList(
            DecoderStage(level, filters, cat_channels=filters[0]) for level in range(NUM_LEVELS - 1, 0, -1)
        )
        self.head = nn.Conv2d(filters[0] * NUM_LEVELS, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: List[torch.Tensor] = []
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                x = self.pool(x)
            x = encoder(x)
            features.append(x)

        # Each decoder stage replaces the encoder output of its level so that
        # finer stages see the decoded (instead of encoded) coarser features.
        finest_decoded = len(features) - 2  # hd4 replaces h4, then hd3 replaces h3, ...
        for i, decoder in enumerate(self.decoders):
            features[finest_decoded - i] = decoder(features)

        return torch.sigmoid(self.head(features[0]))
