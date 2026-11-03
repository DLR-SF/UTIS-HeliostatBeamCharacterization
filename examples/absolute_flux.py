"""Predict relative flux distributions and scale them to absolute flux densities.

Run from the repository root:

    python examples/absolute_flux.py

The measurement metadata below (DNI, sun and aim directions) are illustrative
placeholders. Take them from your calibration database, e.g. PAINT:
the sun direction points from the heliostat towards the sun, the aim direction
from the heliostat to the centre of the focal spot (same coordinate system).
"""

from pathlib import Path

import torch

from utis import STJ_LOWER, FluxPredictor, absolute_flux, get_target
from utis.data.io import load_image_folder
from utis.flux import STJ_HELIOSTAT_AREA

ROOT = Path(__file__).resolve().parents[1]

predictor = FluxPredictor.from_file(ROOT / "trained_models" / "UNet3plus.ckpt")
images, paths = load_image_folder(ROOT / "data" / "real_samples" / STJ_LOWER)
prediction = predictor.predict(images)

n = len(paths)
flux = absolute_flux(
    prediction.relative_flux,
    dni=torch.full((n,), 900.0),  # W/m^2, measured at the time of each image
    sun_direction=torch.tensor([[0.0, -0.6, 0.8]]).repeat(n, 1),  # (east, north, up)
    aim_direction=torch.tensor([[0.0, 60.0, 30.0]]).repeat(n, 1),  # heliostat -> focal spot
    mirror_area=STJ_HELIOSTAT_AREA,
    mirror_reflectivity=0.87,
    target_area=get_target(STJ_LOWER).area,
)
peaks = (flux.amax(dim=(1, 2, 3)) / 1e3).tolist()
for path, peak in zip(paths, peaks):
    print(f"{path.name}: peak flux density {peak:.2f} kW/m^2")

try:
    from utis.visualization import plot_flux_overview

    plot_flux_overview(images, prediction.relative_flux, prediction.background, flux, path="flux_overview.png")
    print("Saved flux_overview.png")
except ImportError:
    print("Install matplotlib (pip install 'utis[plot]') to create the overview figure.")
