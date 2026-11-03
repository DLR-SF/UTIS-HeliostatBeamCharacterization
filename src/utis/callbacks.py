"""Training callbacks."""

from __future__ import annotations

from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from utis.visualization import image_grid

NUM_PREVIEW = 9


class PredictionPreview(pl.Callback):
    """Logs image grids of validation and real-world predictions to TensorBoard.

    * ``preview/synthetic``: input, true flux, predicted flux, true background,
      predicted background of the first validation samples.
    * ``preview/real/<target>``: input, predicted flux, predicted background of
      real target images (``data/real_samples/<target>``).
    """

    def __init__(self, real_samples: Optional[Dict[str, torch.Tensor]] = None, every_n_epochs: int = 1) -> None:
        self.real_samples = real_samples or {}
        self.every_n_epochs = max(1, every_n_epochs)
        self._synthetic: Optional[Dict[str, torch.Tensor]] = None

    def _active(self, trainer: pl.Trainer) -> bool:
        return (
            not trainer.sanity_checking
            and trainer.is_global_zero
            and isinstance(trainer.logger, TensorBoardLogger)
            and trainer.current_epoch % self.every_n_epochs == 0
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0 and dataloader_idx == 0 and self._active(trainer):
            n = NUM_PREVIEW
            self._synthetic = {
                "image": batch["image"][:n],
                "flux": batch["flux"][:n],
                "background": batch["background"][:n],
                "prediction": outputs["prediction"][:n],
            }

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._active(trainer):
            return
        writer = trainer.logger.experiment
        step = trainer.current_epoch

        if self._synthetic is not None:
            s = self._synthetic
            rows = [s["image"], s["flux"], s["prediction"][:, :1], s["background"], s["prediction"][:, 1:]]
            writer.add_image("preview/synthetic", image_grid(rows), step)
            self._synthetic = None

        if self.real_samples:
            was_training = pl_module.training
            pl_module.eval()
            with torch.no_grad():
                for name, images in self.real_samples.items():
                    images = images[:NUM_PREVIEW].to(pl_module.device)
                    prediction = pl_module(images)
                    grid = image_grid([images, prediction[:, :1], prediction[:, 1:]])
                    writer.add_image(f"preview/real/{name}", grid, step)
            pl_module.train(was_training)
