import pytest
import torch
import torch.nn.functional as F

from utis import augmentation as aug
from utis.config import SimulationConfig
from utis.data.dataset import SyntheticTargetDataset, TargetSource, split_indices
from utis.simulation import TargetImageSimulator, compose_image, reflectivity_map


def _legacy_shift(img, shift_x, shift_y):
    """Shift of the original implementation (pad + crop)."""
    pad = max(int(abs(shift_x)), int(abs(shift_y)))
    padded = F.pad(img, (pad, pad, pad, pad), mode="constant")
    sx, sy = pad + int(shift_x), pad + int(shift_y)
    return padded[:, sy : sy + img.size(1), sx : sx + img.size(2)]


@pytest.mark.parametrize("dx,dy", [(0, 0), (5, -7), (-30, 12), (79, -80), (-3.7, 2.9)])
def test_shift_matches_original_implementation(dx, dy):
    x = torch.rand(1, 64, 64)
    assert torch.equal(aug.shift(x, int(dx), int(dy)), _legacy_shift(x, dx, dy))


def test_shift_out_of_range_is_empty():
    assert aug.shift(torch.rand(1, 8, 8), 8, 0).abs().sum() == 0


def test_gaussian_blur_preserves_shape_and_mass():
    x = torch.zeros(1, 32, 32)
    x[0, 16, 16] = 1
    y = aug.gaussian_blur(x)
    assert y.shape == x.shape
    assert y.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_reflectivity_map_is_normalised():
    backgrounds = torch.rand(10, 1, 16, 16)
    reflectivity = reflectivity_map(backgrounds)
    assert reflectivity.shape == (1, 16, 16)
    assert reflectivity.max().item() == pytest.approx(1.0)
    torch.testing.assert_close(reflectivity, backgrounds.mean(0) / backgrounds.mean(0).max())


@pytest.mark.parametrize("u", [0.0, 0.5, 1.0])
def test_compose_image_peak_in_range(u):
    background = torch.full((1, 32, 32), 0.5)
    flux = torch.zeros(1, 32, 32)
    flux[0, 10:20, 10:20] = 1.0
    image = compose_image(background, flux, torch.ones(1, 32, 32), (0.7, 0.9), u)
    assert image.max().item() == pytest.approx(0.7 + 0.2 * u)
    torch.testing.assert_close(image[0, 0, 0], torch.tensor(0.5))


def test_compose_image_never_amplifies_flux():
    image = compose_image(torch.zeros(1, 8, 8), torch.full((1, 8, 8), 0.5), torch.ones(1, 8, 8), (2.0, 3.0), 1.0)
    assert image.max().item() == pytest.approx(0.5)


def _source(simulator, n=4):
    return TargetSource(
        name="t",
        backgrounds=torch.rand(n, 1, 32, 32) * 0.5,
        fluxes=torch.rand(n, 1, 32, 32),
        reflectivity=torch.ones(1, 32, 32),
        simulator=simulator,
        num_samples=n,
    )


def test_validation_samples_are_reproducible():
    simulator = TargetImageSimulator(SimulationConfig(), shift_std=3, augment=False)
    dataset = SyntheticTargetDataset([_source(simulator)], seed=7)
    a, b = dataset[2], dataset[2]
    for key in ("image", "flux", "background"):
        assert torch.equal(a[key], b[key])


def test_image_decomposes_into_reflected_flux_and_background():
    simulator = TargetImageSimulator(SimulationConfig(), shift_std=3, augment=False)
    sample = SyntheticTargetDataset([_source(simulator)], seed=0)[0]
    residual = sample["image"] - sample["background"]
    # I - B is proportional to R * X_r (R = 1 here)
    scale = residual.sum() / sample["flux"].sum()
    torch.testing.assert_close(residual, scale * sample["flux"])


def test_zero_flux_samples_are_consistent():
    simulator = TargetImageSimulator(SimulationConfig(zero_flux_prob=1.0), shift_std=3, augment=True)
    sample = SyntheticTargetDataset([_source(simulator)])[0]
    assert sample["flux"].abs().sum() == 0
    assert torch.equal(sample["image"], sample["background"])
    assert sample["image"].max() <= 0.95 + 1e-6


def test_dataset_indexes_multiple_targets():
    simulator = TargetImageSimulator(SimulationConfig(), shift_std=3, augment=True)
    dataset = SyntheticTargetDataset([_source(simulator, 3), _source(simulator, 5)])
    assert len(dataset) == 8
    assert [dataset[i]["target"] for i in (0, 2, 3, 7)] == [0, 0, 1, 1]


def test_split_indices_is_disjoint_and_deterministic():
    train, val = split_indices(100, 0.1, seed=0, key="a")
    assert len(val) == 10 and len(train) == 90
    assert not set(train.tolist()) & set(val.tolist())
    assert torch.equal(val, split_indices(100, 0.1, seed=0, key="a")[1])
    assert not torch.equal(val, split_indices(100, 0.1, seed=0, key="b")[1])
    assert len(split_indices(10, 0.0, seed=0, key="a")[1]) == 0
