"""Random transformations used to simulate training images.

All functions operate on single samples (``C x H x W``) and draw their random
numbers from an optional :class:`torch.Generator`, which makes validation
samples reproducible.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def uniform(low: float, high: float, generator: Optional[torch.Generator] = None) -> float:
    """Draw a float from ``U(low, high)``."""
    return low + (high - low) * torch.rand((), generator=generator).item()


def bernoulli(p: float, generator: Optional[torch.Generator] = None) -> bool:
    """Return ``True`` with probability ``p``."""
    return torch.rand((), generator=generator).item() < p


def random_flips(x: torch.Tensor, p: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Independently apply a horizontal flip, a vertical flip and a transpose, each with probability ``p``."""
    if bernoulli(p, generator):
        x = x.flip(-1)
    if bernoulli(p, generator):
        x = x.flip(-2)
    if bernoulli(p, generator):
        x = x.transpose(-1, -2)
    return x


def shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Shift the content of ``x`` by ``(-dx, -dy)`` pixels, filling with zeros.

    ``out[..., i, j] = x[..., i + dy, j + dx]`` (zero outside of ``x``).
    """
    out = torch.zeros_like(x)
    h, w = x.shape[-2:]
    if abs(dx) >= w or abs(dy) >= h:
        return out
    src_y, dst_y = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    src_x, dst_x = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[..., dst_y, dst_x] = x[..., src_y, src_x]
    return out


def random_shift(
    x: torch.Tensor, std: float, max_shift: int, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Shift by a normally distributed offset ``N(0, std^2)`` per axis, clipped to ``+-max_shift``."""
    dx, dy = (torch.randn(2, generator=generator) * std).clamp(-max_shift, max_shift).tolist()
    return shift(x, int(dx), int(dy))


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Normalised 2D Gaussian kernel of shape ``1 x 1 x size x size``."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    values = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = values.view(1, 1, size, 1) * values.view(1, 1, 1, size)
    return kernel / kernel.sum()


def gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Blur a ``1 x H x W`` (or ``B x 1 x H x W``) image with zero padding."""
    kernel = gaussian_kernel(kernel_size, sigma).to(x)
    batched = x if x.dim() == 4 else x.unsqueeze(0)
    out = F.conv2d(batched, kernel, padding=kernel_size // 2)
    return out if x.dim() == 4 else out.squeeze(0)


def random_crop_resize(
    x: torch.Tensor, scale_range: Tuple[float, float], generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Crop a random window of ``scale * size`` and resize it back to the original size (zoom-in)."""
    h, w = x.shape[-2:]
    scale = uniform(*scale_range, generator=generator)
    crop_h, crop_w = int(h * scale), int(w * scale)
    top = int(torch.rand((), generator=generator).item() * (h - crop_h))
    left = int(torch.rand((), generator=generator).item() * (w - crop_w))
    cropped = x[..., top : top + crop_h, left : left + crop_w].unsqueeze(0)
    return F.interpolate(cropped, size=(h, w), mode="bilinear").squeeze(0)
