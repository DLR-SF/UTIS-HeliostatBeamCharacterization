"""Typed training configuration, loadable from YAML files.

Example::

    cfg = load_config("configs/paper.yaml", overrides=["trainer.max_epochs=10"])
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, get_args, get_origin, get_type_hints

import yaml

from utis.targets import STJ_LOWER, get_target


@dataclass
class TargetConfig:
    """A calibration target used for training.

    ``flux_set`` and ``shift_std`` default to the values in :data:`utis.targets.TARGETS`.
    """

    name: str
    flux_set: Optional[str] = None
    shift_std: Optional[float] = None
    samples_per_epoch: Optional[int] = None
    """Training samples drawn from this target per epoch (default: number of training fluxes)."""

    def resolved(self) -> TargetConfig:
        """Return a copy with all defaults filled in from the target registry."""
        target = get_target(self.name)
        return dataclasses.replace(
            self,
            name=target.name,
            flux_set=self.flux_set or target.flux_set,
            shift_std=target.shift_std if self.shift_std is None else self.shift_std,
        )


@dataclass
class DataConfig:
    """Training data: targets, validation split and data loading."""

    root: str = "data"
    """Data root containing ``empty_targets/``, ``fluxes/`` and ``real_samples/``."""
    targets: List[TargetConfig] = field(default_factory=lambda: [TargetConfig(STJ_LOWER)])
    val_fraction: float = 0.1
    """Fraction of background *and* flux images held out for validation (per target / flux set)."""
    split_seed: int = 0
    """Seed of the train/validation split; independent of the training seed."""
    batch_size: int = 32
    num_workers: int = 8


@dataclass
class SimulationConfig:
    """Parameters of the synthetic image generation (paper Sec. 3.1, Fig. 2)."""

    train_peak_range: Tuple[float, float] = (0.7, 1.1)
    """Range for the peak pixel value of simulated training images (background + reflected flux)."""
    val_peak_range: Tuple[float, float] = (0.8, 1.0)
    """Range for the peak pixel value of validation images (realistic, no augmentation)."""
    flip_prob: float = 0.5
    max_shift: int = 80
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    crop_prob: float = 0.75
    crop_scale_range: Tuple[float, float] = (0.95, 0.99)
    brightness_range: Tuple[float, float] = (0.95, 1.05)
    zero_flux_prob: float = 0.0
    """Probability of a training sample without focal spot (empty target only)."""
    zero_flux_max_peak: float = 0.95
    """Upper bound for the rescaled peak value of empty-target samples."""


@dataclass
class ModelConfig:
    """Network size."""

    base_channels: int = 32


@dataclass
class OptimizerConfig:
    """AdamP optimizer settings."""

    lr: float = 2e-4
    weight_decay: float = 1e-8


@dataclass
class TrainerConfig:
    """Lightning trainer, logging and output settings."""

    max_epochs: int = 1000
    accelerator: str = "auto"
    devices: Union[int, str, List[int]] = "auto"
    precision: str = "32-true"
    output_dir: str = "runs"
    log_every_n_steps: int = 10
    preview_every_n_epochs: int = 1


@dataclass
class TrainConfig:
    """Complete training configuration."""

    name: str = "utis"
    seed: int = 42
    init_from: Optional[str] = None
    """Initialise the network weights from a checkpoint (``.ckpt``, state dict or TorchScript)."""
    resume_from: Optional[str] = None
    """Resume a training run (weights, optimizer and epoch) from a Lightning checkpoint."""
    data: DataConfig = field(default_factory=DataConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _convert(tp: Any, value: Any) -> Any:
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value)
    origin = get_origin(tp)
    if origin is Union:
        if value is None:
            return None
        candidates = [arg for arg in get_args(tp) if arg is not type(None)]
        return _convert(candidates[0], value) if len(candidates) == 1 else value
    if origin in (list, List):
        (item_type,) = get_args(tp)
        return [_convert(item_type, v) for v in value]
    if origin in (tuple, Tuple):
        return tuple(value)
    if tp is float and isinstance(value, int):
        return float(value)
    return value


def from_dict(cls: type, data: Optional[Dict[str, Any]]) -> Any:
    """Build a (nested) config dataclass from a plain dictionary, rejecting unknown keys."""
    data = data or {}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    return cls(**{key: _convert(hints[key], value) for key, value in data.items()})


def _apply_override(data: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must have the form key.subkey=value, got {override!r}")
    key, raw = override.split("=", 1)
    *parents, leaf = key.strip().split(".")
    node = data
    for parent in parents:
        node = node.setdefault(parent, {})
    node[leaf] = yaml.safe_load(raw)


def load_config(path: Optional[Union[str, Path]] = None, overrides: Sequence[str] = ()) -> TrainConfig:
    """Load a :class:`TrainConfig` from YAML and apply ``key.subkey=value`` overrides."""
    data: Dict[str, Any] = {}
    if path is not None:
        with Path(path).open() as f:
            data = yaml.safe_load(f) or {}
    for override in overrides:
        _apply_override(data, override)
    return from_dict(TrainConfig, data)


def save_config(config: TrainConfig, path: Union[str, Path]) -> None:
    """Write ``config`` to a YAML file that :func:`load_config` can read again."""
    with Path(path).open("w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
