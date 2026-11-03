"""Lightning module for training the UTIS network (paper Sec. 3.1.4)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
from torch import nn

from utis.metrics import RelativeFluxDeviation
from utis.model import UNet3Plus
from utis.model.legacy import convert_legacy_state_dict
from utis.optim import AdamP


class UTISModule(pl.LightningModule):
    """Trains :class:`UNet3Plus` to split a target image into relative flux and background.

    The loss is the pixel-wise mean squared error over both output channels
    (Eqs. 6-8). Validation additionally reports the relative flux deviation of
    Eq. (14) as ``val_flux_deviation``.
    """

    def __init__(self, base_channels: int = 32, lr: float = 2e-4, weight_decay: float = 1e-8) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet3Plus(in_channels=1, out_channels=2, base_channels=base_channels)
        self.loss_fn = nn.MSELoss()
        self.val_deviation = RelativeFluxDeviation()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(batch["image"])
        target = torch.cat([batch["flux"], batch["background"]], dim=1)
        return prediction, self.loss_fn(prediction, target)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _, loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_deviation.reset()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        prediction, loss = self._step(batch)
        self.val_deviation.update(prediction[:, :1], batch["flux"])
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        return {"loss": loss, "prediction": prediction.detach()}

    def on_validation_epoch_end(self) -> None:
        strategy = self.trainer.strategy
        abs_error = strategy.reduce(self.val_deviation.abs_error, reduce_op="sum")
        abs_target = strategy.reduce(self.val_deviation.abs_target, reduce_op="sum")
        self.log("val_flux_deviation", (abs_error / abs_target).float(), prog_bar=True)

    def configure_optimizers(self) -> AdamP:
        return AdamP(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Accept checkpoints written by the original implementation (e.g. UNet3plus.ckpt).
        state_dict = convert_legacy_state_dict(checkpoint["state_dict"])
        checkpoint["state_dict"] = {f"model.{key}": value for key, value in state_dict.items()}
