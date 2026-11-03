"""Translation of relative into absolute flux densities (paper Sec. 3.3).

The total power of a focal spot is estimated with a simple heliostat model::

    Q_spot = r * DNI * cos(alpha) * A_H                      (Eqs. 9, 11)
    alpha  = 1/2 * angle(sun direction, aim direction)       (Eq. 10)

and the relative flux ``X_r`` predicted by the network is scaled such that its
integral over the target matches ``Q_spot``::

    c = Q_spot / (A_T * mean(X_r))                           (Eqs. 12, 13)
    X = c * X_r

The model assumes that the whole focal spot lies on the target (no spillage)
and that there is no shading or blocking.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch

ArrayLike = Union[float, torch.Tensor, Sequence[float]]

STJ_HELIOSTAT_AREA = 4 * 1.6 * 1.2
"""Mirror area of a Solar Tower Jülich heliostat in m^2 as used in the paper."""


def _as_tensor(value: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def angle_between(a: ArrayLike, b: ArrayLike) -> torch.Tensor:
    """Angle in radians between vectors ``a`` and ``b`` (shape ``(..., 3)``)."""
    a, b = _as_tensor(a), _as_tensor(b)
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
    return torch.arccos(cos.clamp(-1.0, 1.0))


def incidence_angle(sun_direction: ArrayLike, aim_direction: ArrayLike) -> torch.Tensor:
    """Angle of incidence on the mirror (Eq. 10), assuming the normal bisects sun and aim direction.

    Args:
        sun_direction: Vector from the heliostat towards the sun, e.g. (east, north, up).
        aim_direction: Vector from the heliostat to the centre of the focal spot,
            in the same coordinate system.
    """
    return 0.5 * angle_between(sun_direction, aim_direction)


def spot_power(
    dni: ArrayLike, mirror_area: ArrayLike, mirror_reflectivity: ArrayLike, incidence: ArrayLike
) -> torch.Tensor:
    """Total reflected power ``Q_spot`` in W (Eqs. 9 and 11).

    Args:
        dni: Direct normal irradiance in W/m^2.
        mirror_area: Mirror area ``A_H`` of the heliostat in m^2.
        mirror_reflectivity: Mirror reflectivity ``r`` in ``[0, 1]``.
        incidence: Angle of incidence ``alpha`` in radians.
    """
    return (
        _as_tensor(mirror_reflectivity) * _as_tensor(dni) * torch.cos(_as_tensor(incidence)) * _as_tensor(mirror_area)
    )


def flux_scaling_factor(relative_flux: torch.Tensor, target_area: ArrayLike, power: ArrayLike) -> torch.Tensor:
    """Scaling factor ``c`` (Eq. 13) for one image (``[1 x] H x W``) or a batch (``B x 1 x H x W``)."""
    mean = relative_flux.double().mean(dim=(-2, -1))  # B x 1, 1 or scalar
    mean = mean.flatten(1).squeeze(1) if relative_flux.dim() == 4 else mean.reshape(())
    return _as_tensor(power) / (_as_tensor(target_area) * mean)


def absolute_flux(
    relative_flux: torch.Tensor,
    *,
    dni: ArrayLike,
    sun_direction: ArrayLike,
    aim_direction: ArrayLike,
    mirror_area: ArrayLike,
    mirror_reflectivity: ArrayLike,
    target_area: ArrayLike,
) -> torch.Tensor:
    """Absolute flux density ``X`` in W/m^2 from the network's relative flux ``X_r``.

    ``relative_flux`` is a single image (``[1 x] H x W``) or a batch
    (``B x 1 x H x W``). For a batch, all other arguments may be given per image
    (shape ``(B,)``, or ``(B, 3)`` for the directions) or as scalars.

    Example (values of the paper's evaluation at the Solar Tower Jülich)::

        X = absolute_flux(X_r, dni=900.0, sun_direction=sun, aim_direction=aim,
                          mirror_area=STJ_HELIOSTAT_AREA, mirror_reflectivity=0.87,
                          target_area=STJ_TARGET_AREA)
    """
    alpha = incidence_angle(sun_direction, aim_direction)
    power = spot_power(dni, mirror_area, mirror_reflectivity, alpha)
    scale = flux_scaling_factor(relative_flux, target_area, power)
    if relative_flux.dim() == 4:
        scale = scale.reshape(-1, 1, 1, 1)
    return (relative_flux.double() * scale).to(relative_flux.dtype)
