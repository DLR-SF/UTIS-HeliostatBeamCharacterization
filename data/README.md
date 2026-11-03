# Data

Images used to generate synthetic training data and to inspect the model on real measurements.
All images are rectified and cropped target images of 256 × 256 pixels, recorded at the
Solar Tower Jülich (STJ) and the Multi Focus Tower (MFT).

Target names follow the [PAINT database](https://paint-database.org/) (`solar_tower_juelich_upper`,
`solar_tower_juelich_lower`, `multi_focus_tower`). Numeric target ids are deliberately not used, since older DLR
data sets and PAINT number the targets differently. In the original UTIS data, the directories were
`emptyTargets6cropped` (STJ upper), `emptyTargets7cropped` (STJ lower) and `emptyTargets4cropped` (MFT).

## License

The data in this directory are licensed under the
[Community Data License Agreement – Permissive – Version 2.0](https://cdla.dev/permissive-2-0/)
(SPDX: `CDLA-Permissive-2.0`), the same license as the PAINT database. The code of this repository is licensed
under the MIT license (see [`LICENSE`](../LICENSE)).

## `empty_targets/<target>/`

Images of the empty target without focal spot (8-bit grayscale), taken under a wide range of
irradiation conditions. They serve as backgrounds `B` of the synthetic images, and their
pixel-wise mean defines the relative target reflectivity `R` (Eq. 5 of the paper).

| Target | Images |
|---|---|
| `solar_tower_juelich_upper` | 352 |
| `solar_tower_juelich_lower` | 999 |
| `multi_focus_tower` | 382 |

## `fluxes/<flux set>/`

Focal spots `X_r` simulated with a raytracer using deflectometry measurements of the heliostat
surfaces. Every image is normalised to a maximum of 1 and stored as a 16-bit PNG with
10-bit precision (1023 levels; quantisation error of the flux below 0.15 % in terms of Eq. 14).
The focal spot size depends on the physical size of the target, so there is one set per target
geometry:

| Flux set | Images | Used for |
|---|---|---|
| `solar_tower_juelich` | 1000 | `solar_tower_juelich_upper`, `solar_tower_juelich_lower` (same size, mounted above each other) |
| `multi_focus_tower` | 1000 | `multi_focus_tower` |

## `real_samples/<target>/`

Real target images with focal spots (`solar_tower_juelich_lower`, 9 images), used to monitor the
model during training (TensorBoard) and in the examples.

## Adding a target

1. Put images of the empty target into `empty_targets/<name>/` and, if the target size differs from
   the existing ones, raytraced focal spots into `fluxes/<flux set>/`.
2. Register the target in [`src/utis/targets.py`](../src/utis/targets.py) (flux set, displacement of
   the focal spots, area) or set `flux_set` and `shift_std` directly in the training config.
