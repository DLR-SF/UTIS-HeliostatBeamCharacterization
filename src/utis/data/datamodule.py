"""Lightning data module for training on one or several calibration targets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from utis.config import DataConfig, SimulationConfig, TargetConfig
from utis.data.dataset import SyntheticTargetDataset, TargetSource, split_indices
from utis.data.io import list_images, load_image_folder
from utis.simulation import TargetImageSimulator, reflectivity_map

log = logging.getLogger(__name__)

VAL_SEED = 1234
"""Fixed seed of the validation samples, independent of the training seed."""


def empty_target_dir(root: Path, target: str) -> Path:
    """Directory with the empty-target images of ``target``."""
    return root / "empty_targets" / target


def flux_dir(root: Path, flux_set: str) -> Path:
    """Directory with the simulated fluxes of ``flux_set``."""
    return root / "fluxes" / flux_set


def real_samples_dir(root: Path, target: str) -> Path:
    """Directory with real images of ``target``."""
    return root / "real_samples" / target


class TargetDataModule(pl.LightningDataModule):
    """Loads empty-target images and simulated fluxes and serves synthetic samples.

    Directory layout below ``config.root``::

        empty_targets/<target>/*.png   images of the empty target (backgrounds)
        fluxes/<flux_set>/*.png        raytraced focal spots
        real_samples/<target>/*.png    optional real images for visual monitoring

    Backgrounds and fluxes are split into training and validation images per
    directory (see :func:`split_indices`). Targets sharing a flux set also share
    its split. The reflectivity map is computed from all empty-target images,
    as it is a property of the target.
    """

    def __init__(self, config: DataConfig, simulation: SimulationConfig) -> None:
        super().__init__()
        self.config = config
        self.simulation = simulation
        self.root = Path(config.root)
        self.targets: List[TargetConfig] = [t.resolved() for t in config.targets]
        self.train_dataset: Optional[SyntheticTargetDataset] = None
        self.val_dataset: Optional[SyntheticTargetDataset] = None

    @property
    def target_names(self) -> List[str]:
        return [t.name for t in self.targets]

    @property
    def has_validation(self) -> bool:
        return self.config.val_fraction > 0

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return
        cfg = self.config
        flux_cache: Dict[str, torch.Tensor] = {}
        train_sources, val_sources = [], []

        for target in self.targets:
            assert target.flux_set is not None and target.shift_std is not None
            backgrounds, _ = load_image_folder(empty_target_dir(self.root, target.name))
            if target.flux_set not in flux_cache:
                flux_cache[target.flux_set], _ = load_image_folder(flux_dir(self.root, target.flux_set))
            fluxes = flux_cache[target.flux_set]
            reflectivity = reflectivity_map(backgrounds)

            bg_train, bg_val = split_indices(len(backgrounds), cfg.val_fraction, cfg.split_seed, target.name)
            fx_train, fx_val = split_indices(len(fluxes), cfg.val_fraction, cfg.split_seed, target.flux_set)
            log.info(
                "Target %s: %d/%d backgrounds, %d/%d fluxes (%s) for training/validation",
                target.name,
                len(bg_train),
                len(bg_val),
                len(fx_train),
                len(fx_val),
                target.flux_set,
            )

            train_sources.append(
                TargetSource(
                    name=target.name,
                    backgrounds=backgrounds[bg_train],
                    fluxes=fluxes[fx_train],
                    reflectivity=reflectivity,
                    simulator=TargetImageSimulator(self.simulation, target.shift_std, augment=True),
                    num_samples=target.samples_per_epoch or len(fx_train),
                )
            )
            if self.has_validation:
                val_sources.append(
                    TargetSource(
                        name=target.name,
                        backgrounds=backgrounds[bg_val],
                        fluxes=fluxes[fx_val],
                        reflectivity=reflectivity,
                        simulator=TargetImageSimulator(self.simulation, target.shift_std, augment=False),
                        num_samples=len(fx_val),
                    )
                )

        self.train_dataset = SyntheticTargetDataset(train_sources)
        self.val_dataset = SyntheticTargetDataset(val_sources, seed=VAL_SEED) if val_sources else None

    def _loader(self, dataset: SyntheticTargetDataset, train: bool) -> DataLoader:
        workers = self.config.num_workers
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "call setup() first"
        return self._loader(self.train_dataset, train=True)

    def val_dataloader(self) -> List[DataLoader]:
        if self.val_dataset is None:
            return []
        return [self._loader(self.val_dataset, train=False)]

    def real_samples(self) -> Dict[str, torch.Tensor]:
        """Real target images per target (if available), used for visual monitoring."""
        samples = {}
        for name in self.target_names:
            directory = real_samples_dir(self.root, name)
            if directory.is_dir() and list_images(directory):
                samples[name], _ = load_image_folder(directory)
        return samples
