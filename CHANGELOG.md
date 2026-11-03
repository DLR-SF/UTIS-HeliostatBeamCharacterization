# Changelog

## 2.0.0 (2026-11-03)

Complete restructuring of the code base. The published models are unchanged and produce identical results.

### Added
- Training on several calibration targets (STJ upper, STJ lower, MFT) with one flux set per target geometry,
  per-target focal spot displacement and optional empty-target samples (`configs/multi_target.yaml`).
- Absolute flux scaling of the paper (Eqs. 9-13): `utis.absolute_flux`.
- Real validation split of backgrounds and fluxes; checkpoint selection by the relative flux deviation (Eq. 14).
- `utis` command line interface (`train`, `predict`, `export`), YAML configs with overrides.
- `FluxPredictor` / `load_model` for Lightning checkpoints, state dicts and TorchScript models.
- TorchScript and ONNX export, Fig.-4-style visualisation, test suite and CI.
- Code style checks with ruff (PEP 8, import sorting, Google-style docstrings, type hints) and pre-commit hooks.
- Empty-target images of STJ upper and MFT, simulated fluxes for the MFT.
- Target areas for all three targets (MFT derived from the PAINT tower measurements).
- Data license: CDLA-Permissive-2.0 (as the PAINT database); code remains MIT.

### Changed
- `src/` layout; package modules reorganised (`model`, `data`, `simulation`, `augmentation`, ...).
- Target names follow the PAINT database: `solar_tower_juelich_upper`, `solar_tower_juelich_lower`,
  `multi_focus_tower`. Data directories renamed accordingly; numeric target ids are no longer used.
- UNet 3+ implemented compactly with loops; original checkpoints are converted on load (bit-identical outputs).
- Simulated fluxes stored as 16-bit PNGs with 10-bit precision instead of 8 bit.
- Augmentations are drawn per sample instead of per batch; images are loaded in sorted order.
- Training outputs go to `runs/` instead of `trained_models/`.

### Fixed
- Broken imports: `python -m utis.train` did not run.
- Validation loss was computed on training batches and over both output channels with the wrong normalisation.
- Hyperparameters and reflectivity map were not stored in checkpoints.
- Multi-target training of the previous code used the reflectivity map of the wrong target for STJ images and an
  inconsistent background ground truth for empty-target samples.
