"""Training entry point."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from utis.callbacks import PredictionPreview
from utis.config import TrainConfig, save_config
from utis.data.datamodule import TargetDataModule
from utis.inference import load_model
from utis.lightning_module import UTISModule

log = logging.getLogger(__name__)


def build_module(config: TrainConfig) -> UTISModule:
    """Create the Lightning module, optionally initialised from a pretrained model."""
    module = UTISModule(
        base_channels=config.model.base_channels, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay
    )
    if config.init_from:
        pretrained = load_model(config.init_from)
        module.model.load_state_dict(pretrained.state_dict())
        log.info("Initialised weights from %s", config.init_from)
    return module


def train(config: TrainConfig) -> Optional[str]:
    """Train a model as described by ``config``. Returns the path of the best checkpoint."""
    pl.seed_everything(config.seed, workers=True)

    datamodule = TargetDataModule(config.data, config.simulation)
    datamodule.setup("fit")
    module = build_module(config)

    logger = TensorBoardLogger(save_dir=config.trainer.output_dir, name=config.name)
    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    monitor = "val_flux_deviation" if datamodule.has_validation else "train_loss_epoch"
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="epoch={epoch:04d}-" + monitor + "={" + monitor + ":.5f}",
        auto_insert_metric_name=False,
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    preview = PredictionPreview(datamodule.real_samples(), every_n_epochs=config.trainer.preview_every_n_epochs)

    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        logger=logger,
        callbacks=[checkpoint, preview],
        log_every_n_steps=config.trainer.log_every_n_steps,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.resume_from)
    return checkpoint.best_model_path or None
