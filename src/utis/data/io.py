"""Reading and writing of target, background and flux images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

PathLike = Union[str, Path]

IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
_SIXTEEN_BIT_MODES = ("I;16", "I;16B", "I;16L", "I")


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB -> grayscale conversion used in the paper (Sec. 3.2).

    Red and blue are weighted equally and green twice as much, reflecting the
    Bayer pattern of the CMOS sensor: ``(R + 2G + B) / 4``.
    """
    return (rgb[..., 0] + 2.0 * rgb[..., 1] + rgb[..., 2]) / 4.0


def load_image(path: PathLike) -> torch.Tensor:
    """Load an image as a ``1 x H x W`` float tensor with values in ``[0, 1]``.

    8-bit and 16-bit grayscale images are scaled by their full range; RGB images
    are converted with :func:`to_grayscale`.
    """
    with Image.open(path) as img:
        if img.mode in _SIXTEEN_BIT_MODES:
            array = np.asarray(img, dtype=np.float32) / 65535.0
        elif img.mode == "L":
            array = np.asarray(img, dtype=np.float32) / 255.0
        else:
            array = to_grayscale(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def list_images(directory: PathLike) -> List[Path]:
    """Return all image files in ``directory``, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_images(paths: Sequence[PathLike]) -> torch.Tensor:
    """Load several images of equal size into an ``N x 1 x H x W`` tensor."""
    if not paths:
        raise ValueError("No images given")
    return torch.stack([load_image(p) for p in paths])


def load_image_folder(directory: PathLike) -> Tuple[torch.Tensor, List[Path]]:
    """Load all images of a directory. Returns the image tensor and the file paths."""
    paths = list_images(directory)
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return load_images(paths), paths


def save_image_16bit(image: torch.Tensor, path: PathLike) -> None:
    """Save a ``[1 x] H x W`` tensor with values in ``[0, 1]`` as a 16-bit grayscale PNG."""
    array = image.detach().cpu().squeeze().clamp(0, 1).numpy()
    Image.fromarray(np.round(array * 65535.0).astype(np.uint16)).save(path)
