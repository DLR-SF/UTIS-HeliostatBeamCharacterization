import shutil
from pathlib import Path

import pytest
import torch

from tests.paths import DATA, REAL_SAMPLES
from utis.data.io import list_images, load_image_folder
from utis.targets import STJ_LOWER


@pytest.fixture(scope="session")
def real_samples() -> torch.Tensor:
    images, _ = load_image_folder(REAL_SAMPLES)
    return images


@pytest.fixture()
def tiny_data_root(tmp_path: Path) -> Path:
    """A small copy of the data directory (STJ lower only) for fast training tests."""
    pairs = [
        (DATA / "empty_targets" / STJ_LOWER, tmp_path / "empty_targets" / STJ_LOWER, 6),
        (DATA / "fluxes" / "solar_tower_juelich", tmp_path / "fluxes" / "solar_tower_juelich", 6),
        (REAL_SAMPLES, tmp_path / "real_samples" / STJ_LOWER, 2),
    ]
    for source, target, n in pairs:
        target.mkdir(parents=True)
        for path in list_images(source)[:n]:
            shutil.copy(path, target / path.name)
    return tmp_path
