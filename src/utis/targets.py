"""Calibration targets supported by UTIS.

Target names follow the naming of the `PAINT database
<https://github.com/ARTIST-Association/PAINT>`_ so that images, models and
metadata can be exchanged without translation tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

STJ_UPPER = "solar_tower_juelich_upper"
"""Upper calibration target of the Solar Tower Jülich."""
STJ_LOWER = "solar_tower_juelich_lower"
"""Lower calibration target of the Solar Tower Jülich."""
MFT = "multi_focus_tower"
"""Calibration target of the Multi Focus Tower."""

STJ_TARGET_AREA = 8.625 * (7.5855 - 0.375)
"""Area of each of the two STJ calibration targets in m^2 (both are equally large), as used in the paper."""

MFT_TARGET_AREA = 5.401 * 6.388
"""Area of the MFT calibration target in m^2.

Derived from the corner coordinates in the PAINT tower measurements
(``WRI1030197-tower-measurements.json``). The same derivation reproduces
:data:`STJ_TARGET_AREA` within 0.2 %.
"""


@dataclass(frozen=True)
class Target:
    """Static properties of a calibration target.

    Attributes:
        name: PAINT name of the target.
        flux_set: Name of the simulated flux set that matches the target geometry.
            Raytraced focal spots depend on the physical target size, so targets of
            equal size share one set.
        shift_std: Standard deviation (in pixels) of the random displacement of
            simulated focal spots. It reflects how far the real focal spots scatter
            around the target centre in the rectified 256 x 256 images.
        area: Area in m^2 of the target region shown in the rectified image (between
            the crop markers), required for absolute flux scaling.
    """

    name: str
    flux_set: str
    shift_std: float
    area: Optional[float] = None


TARGETS = {
    STJ_UPPER: Target(STJ_UPPER, flux_set="solar_tower_juelich", shift_std=15.0, area=STJ_TARGET_AREA),
    STJ_LOWER: Target(STJ_LOWER, flux_set="solar_tower_juelich", shift_std=15.0, area=STJ_TARGET_AREA),
    MFT: Target(MFT, flux_set="multi_focus_tower", shift_std=20.0, area=MFT_TARGET_AREA),
}


def get_target(name: str) -> Target:
    """Return the :class:`Target` registered under its PAINT name."""
    if name not in TARGETS:
        raise KeyError(f"Unknown target {name!r}; known targets: {sorted(TARGETS)}")
    return TARGETS[name]
