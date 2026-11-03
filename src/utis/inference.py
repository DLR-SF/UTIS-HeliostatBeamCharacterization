"""Loading trained models and predicting flux and background."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import NamedTuple, Sequence, Union

import torch
from torch import nn

from utis.data.io import PathLike, list_images, load_images
from utis.model import UNet3Plus, convert_legacy_state_dict


class Prediction(NamedTuple):
    """Network outputs for a batch of images."""

    relative_flux: torch.Tensor
    """``B x 1 x H x W`` relative flux density ``X_r``."""
    background: torch.Tensor
    """``B x 1 x H x W`` background ``B``."""


def _is_torchscript(path: Path) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return any(name.endswith("constants.pkl") or "/code/" in name for name in archive.namelist())


def load_model(path: PathLike, device: Union[str, torch.device] = "cpu") -> nn.Module:
    """Load a trained network in evaluation mode.

    Supported formats:

    * Lightning checkpoints (``.ckpt``), including the original paper checkpoint,
    * plain state dicts of :class:`UNet3Plus` (new or original layout),
    * TorchScript models such as ``utis_model_scripted.pt`` (returned as-is).
    """
    path = Path(path)
    if _is_torchscript(path):
        return torch.jit.load(str(path), map_location=device).eval()

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older torch versions or checkpoints with non-tensor objects
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = convert_legacy_state_dict(state_dict)

    first_conv = state_dict["encoders.0.0.conv.weight"]
    model = UNet3Plus(
        in_channels=first_conv.shape[1],
        out_channels=state_dict["head.weight"].shape[0],
        base_channels=first_conv.shape[0],
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()


class FluxPredictor:
    """Applies a trained network to rectified target images.

    Example::

        predictor = FluxPredictor.from_file("trained_models/UNet3plus.ckpt")
        prediction = predictor.predict_folder("data/real_samples/solar_tower_juelich_lower")
        prediction.relative_flux  # B x 1 x 256 x 256
    """

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu", batch_size: int = 16) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size

    @classmethod
    def from_file(cls, path: PathLike, device: Union[str, torch.device] = "cpu", batch_size: int = 16) -> FluxPredictor:
        """Create a predictor from a model file (see :func:`load_model`)."""
        return cls(load_model(path, device), device, batch_size)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Prediction:
        """Predict relative flux and background for ``B x 1 x H x W``, ``1 x H x W`` or ``H x W`` images in [0, 1]."""
        if images.dim() == 2:
            images = images[None, None]
        elif images.dim() == 3:
            images = images[None]
        outputs = [self.model(chunk.to(self.device, torch.float32)).cpu() for chunk in images.split(self.batch_size)]
        output = torch.cat(outputs)
        return Prediction(output[:, :1], output[:, 1:])

    def predict_files(self, paths: Sequence[PathLike]) -> Prediction:
        return self.predict(load_images(paths))

    def predict_folder(self, directory: PathLike) -> Prediction:
        return self.predict_files(list_images(directory))

    __call__ = predict
