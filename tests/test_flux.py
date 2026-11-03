import math

import pytest
import torch

from utis.flux import STJ_HELIOSTAT_AREA, absolute_flux, angle_between, incidence_angle, spot_power
from utis.targets import STJ_TARGET_AREA


def test_incidence_angle_is_half_the_angle_between_sun_and_aim():
    assert angle_between([1, 0, 0], [0, 1, 0]).item() == pytest.approx(math.pi / 2)
    assert incidence_angle([1, 0, 0], [0, 1, 0]).item() == pytest.approx(math.pi / 4)
    assert incidence_angle([0, 0, 2], [0, 0, 1]).item() == pytest.approx(0.0, abs=1e-7)


def test_absolute_flux_integrates_to_spot_power():
    relative = torch.rand(1, 256, 256)
    sun, aim = torch.tensor([0.3, -0.5, 0.8]), torch.tensor([-1.0, 40.0, 30.0])
    flux = absolute_flux(
        relative,
        dni=900.0,
        sun_direction=sun,
        aim_direction=aim,
        mirror_area=STJ_HELIOSTAT_AREA,
        mirror_reflectivity=0.87,
        target_area=STJ_TARGET_AREA,
    )
    power = spot_power(900.0, STJ_HELIOSTAT_AREA, 0.87, incidence_angle(sun, aim))
    pixel_area = STJ_TARGET_AREA / relative[0].numel()
    assert (flux.double().sum() * pixel_area).item() == pytest.approx(power.item(), rel=1e-5)


def test_absolute_flux_matches_original_evaluation_script():
    # Scaling as in the original evaluation script, which normalised X_r to its maximum first.
    relative = torch.rand(1, 256, 256) * 0.7
    sun, aim, dni, r = torch.tensor([0.2, -0.4, 0.9]), torch.tensor([3.0, 60.0, 20.0]), 913.0, 0.87
    alpha = angle_between(sun, aim)
    normed = relative / relative.max()
    c = r * torch.cos(alpha / 2) * dni * STJ_HELIOSTAT_AREA * 256**2 / (STJ_TARGET_AREA * normed.double().sum())
    expected = normed.double() * c
    flux = absolute_flux(
        relative,
        dni=dni,
        sun_direction=sun,
        aim_direction=aim,
        mirror_area=STJ_HELIOSTAT_AREA,
        mirror_reflectivity=r,
        target_area=STJ_TARGET_AREA,
    )
    torch.testing.assert_close(flux.double(), expected, rtol=1e-5, atol=1e-3)


def test_absolute_flux_batch_with_per_image_parameters():
    relative = torch.rand(3, 1, 16, 16)
    dni = torch.tensor([800.0, 900.0, 1000.0])
    sun = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    aim = torch.tensor([[0.0, 1.0, 1.0]] * 3)
    batch = absolute_flux(
        relative,
        dni=dni,
        sun_direction=sun,
        aim_direction=aim,
        mirror_area=1.0,
        mirror_reflectivity=1.0,
        target_area=1.0,
    )
    for i in range(3):
        single = absolute_flux(
            relative[i],
            dni=dni[i],
            sun_direction=sun[i],
            aim_direction=aim[i],
            mirror_area=1.0,
            mirror_reflectivity=1.0,
            target_area=1.0,
        )
        torch.testing.assert_close(batch[i], single)
