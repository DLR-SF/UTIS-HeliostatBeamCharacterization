"""Datasets of synthetic target images."""

from __future__ import annotations

import bisect
import zlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from utis.simulation import TargetImageSimulator


@dataclass
class TargetSource:
    """Images belonging to one calibration target (one split)."""

    name: str
    backgrounds: torch.Tensor
    """``N x 1 x H x W`` empty-target images."""
    fluxes: torch.Tensor
    """``M x 1 x H x W`` simulated focal spots matching the target size."""
    reflectivity: torch.Tensor
    """``1 x H x W`` relative reflectivity of the target."""
    simulator: TargetImageSimulator
    num_samples: int
    """Number of samples this target contributes to the dataset."""


def split_indices(n: int, val_fraction: float, seed: int, key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministically split ``range(n)`` into train and validation indices.

    The permutation depends on ``seed`` and ``key`` (e.g. the directory name),
    so every image directory is split independently and reproducibly.
    """
    generator = torch.Generator().manual_seed(seed + zlib.crc32(key.encode()))
    perm = torch.randperm(n, generator=generator)
    n_val = int(round(n * val_fraction))
    if val_fraction > 0 and n_val == 0:
        n_val = 1
    if n_val >= n and n > 0:
        raise ValueError(f"val_fraction={val_fraction} leaves no training images for {key!r} (n={n})")
    return perm[n_val:].sort().values, perm[:n_val].sort().values


class SyntheticTargetDataset(Dataset):
    """Composes target images on the fly from backgrounds and simulated fluxes.

    Sample ``i`` of a target uses flux ``i mod M`` and a random background.
    With ``seed=None`` (training) all random numbers come from the global torch
    RNG, so each epoch yields new combinations. With an integer ``seed``
    (validation) every sample is generated from its own seeded generator and is
    identical in each epoch.

    Each item is a dict with ``image``, ``flux``, ``background`` (``1 x H x W``)
    and the integer ``target`` index.
    """

    def __init__(self, sources: Sequence[TargetSource], seed: Optional[int] = None) -> None:
        self.sources = list(sources)
        self.seed = seed
        self._offsets: List[int] = []
        total = 0
        for source in self.sources:
            self._offsets.append(total)
            total += source.num_samples
        self._length = total

    @property
    def target_names(self) -> List[str]:
        return [s.name for s in self.sources]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self._length:
            raise IndexError(index)
        target = bisect.bisect_right(self._offsets, index) - 1
        source = self.sources[target]
        local = index - self._offsets[target]

        generator = torch.Generator().manual_seed(self.seed + index) if self.seed is not None else None
        background_index = int(torch.randint(len(source.backgrounds), (), generator=generator))
        sample = source.simulator(
            source.backgrounds[background_index],
            source.fluxes[local % len(source.fluxes)],
            source.reflectivity,
            generator=generator,
        )
        return {**sample.as_dict(), "target": target}
