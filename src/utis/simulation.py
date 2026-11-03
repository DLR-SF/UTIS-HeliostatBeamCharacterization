"""Generation of synthetic target images with known flux and background (paper Sec. 3.1).

A simulated image is composed according to Eq. (1)::

    I = R * X_r + B

with the spatial target reflectivity ``R`` (Eq. 5), a raytraced focal spot
``X_r`` and an image of the empty target ``B``. The network learns to recover
``X_r`` and ``B`` from ``I``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from utis import augmentation as aug
from utis.config import SimulationConfig

_EPS = 1e-12


def reflectivity_map(backgrounds: torch.Tensor) -> torch.Tensor:
    """Relative target reflectivity from many empty-target images (Eq. 5).

    Args:
        backgrounds: ``N x 1 x H x W`` images of the empty target.

    Returns:
        ``1 x H x W`` map, normalised to a maximum of 1.
    """
    mean = backgrounds.mean(dim=0)
    return mean / mean.max()


def compose_image(
    background: torch.Tensor,
    flux: torch.Tensor,
    reflectivity: torch.Tensor,
    peak_range: Tuple[float, float],
    u: float,
) -> torch.Tensor:
    """Superpose a reflected focal spot on a background image (Eq. 1).

    The flux is scaled such that the peak of the composed image,
    approximated by ``max(B) + s * max(R * X_r)``, lies at the position ``u``
    within ``peak_range``. The scale ``s`` is limited to ``[0, 1]`` so the
    reflected flux never exceeds its simulated intensity.

    Args:
        background: ``1 x H x W`` empty-target image ``B``.
        flux: ``1 x H x W`` relative flux ``X_r``.
        reflectivity: ``1 x H x W`` reflectivity ``R``.
        peak_range: ``(low, high)`` range of the targeted image peak value.
        u: Position within the range, in ``[0, 1]``.
    """
    reflected = flux * reflectivity
    background_peak = background.max()
    reflected_peak = reflected.max() + _EPS
    low, high = peak_range
    scale_low = ((low - background_peak) / reflected_peak).clamp(0, 1)
    scale_high = ((high - background_peak) / reflected_peak).clamp(0, 1)
    scale = scale_low + (scale_high - scale_low) * u
    return background + scale * reflected


@dataclass
class SimulatedSample:
    """A synthetic image together with its ground truth."""

    image: torch.Tensor
    """Network input ``I`` (``1 x H x W``)."""
    flux: torch.Tensor
    """Ground truth relative flux ``X_r`` (``1 x H x W``)."""
    background: torch.Tensor
    """Ground truth background ``B`` (``1 x H x W``)."""

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {"image": self.image, "flux": self.flux, "background": self.background}


class TargetImageSimulator:
    """Creates training/validation samples from a background and a raytraced flux.

    Training samples (``augment=True``) are subject to

    * random flips/transpose and displacement of the focal spot,
    * random zoom (crop + resize) of background and reflectivity map,
    * random brightness of the background,
    * a wide range of focal-spot intensities and, optionally, samples without focal spot.

    Validation samples (``augment=False``) only place the focal spot randomly
    (flips and displacement, as real focal spots are not centred) and use the
    realistic intensity range ``val_peak_range``. With a seeded generator they
    are fully reproducible.

    Args:
        config: Simulation parameters.
        shift_std: Standard deviation of the focal spot displacement (pixels).
        augment: Whether to create training (``True``) or validation samples.
    """

    def __init__(self, config: SimulationConfig, shift_std: float, augment: bool) -> None:
        self.config = config
        self.shift_std = shift_std
        self.augment = augment

    def place_flux(self, flux: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Randomly orient and displace the focal spot, then smooth it."""
        cfg = self.config
        flux = aug.random_flips(flux, cfg.flip_prob, generator)
        flux = aug.random_shift(flux, self.shift_std, cfg.max_shift, generator)
        return aug.gaussian_blur(flux, cfg.blur_kernel_size, cfg.blur_sigma)

    def augment_background(
        self, background: torch.Tensor, reflectivity: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the same random zoom to background and reflectivity and vary the brightness."""
        cfg = self.config
        if aug.bernoulli(cfg.crop_prob, generator):
            stacked = aug.random_crop_resize(torch.cat([background, reflectivity]), cfg.crop_scale_range, generator)
            background, reflectivity = stacked[:1], stacked[1:]
        background = background * aug.uniform(*cfg.brightness_range, generator=generator)
        return background, reflectivity

    def __call__(
        self,
        background: torch.Tensor,
        flux: torch.Tensor,
        reflectivity: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> SimulatedSample:
        cfg = self.config
        flux = self.place_flux(flux, generator)

        if not self.augment:
            u = torch.rand((), generator=generator).item()
            image = compose_image(background, flux, reflectivity, cfg.val_peak_range, u)
            return SimulatedSample(image, flux, background)

        background, reflectivity = self.augment_background(background, reflectivity, generator)

        if aug.bernoulli(cfg.zero_flux_prob, generator):
            # Empty target: rescale the background to a random peak in [max(B), zero_flux_max_peak].
            peak = background.max()
            target_peak = peak + (cfg.zero_flux_max_peak - peak) * torch.rand((), generator=generator)
            background = background * target_peak / (peak + _EPS)
            return SimulatedSample(background.clone(), torch.zeros_like(flux), background)

        u = torch.rand((), generator=generator).item()
        image = compose_image(background, flux, reflectivity, cfg.train_peak_range, u)
        return SimulatedSample(image, flux, background)
