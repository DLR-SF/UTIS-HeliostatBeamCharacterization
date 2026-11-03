import torch

from tests.paths import PAPER_CHECKPOINT, REAL_SAMPLES
from utis.cli import main
from utis.config import load_config
from utis.inference import load_model
from utis.training import train


def _tiny_config(data_root, output_dir, **extra):
    overrides = [
        f"data.root={data_root}",
        "data.batch_size=2",
        "data.num_workers=0",
        "data.val_fraction=0.34",
        "model.base_channels=2",
        "trainer.max_epochs=2",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        f"trainer.output_dir={output_dir}",
        "trainer.log_every_n_steps=1",
    ] + [f"{k}={v}" for k, v in extra.items()]
    return load_config("configs/paper.yaml", overrides)


def test_training_writes_loadable_checkpoint(tiny_data_root, tmp_path):
    best = train(_tiny_config(tiny_data_root, tmp_path / "runs"))
    assert best and best.endswith(".ckpt")
    model = load_model(best)
    with torch.no_grad():
        assert model(torch.rand(1, 1, 256, 256)).shape == (1, 2, 256, 256)
    assert (tmp_path / "runs" / "paper_stj_lower" / "version_0" / "config.yaml").exists()


def test_training_can_start_from_paper_checkpoint(tiny_data_root, tmp_path):
    config = _tiny_config(
        tiny_data_root,
        tmp_path / "runs",
        **{"model.base_channels": 32, "trainer.max_epochs": 1, "init_from": str(PAPER_CHECKPOINT)},
    )
    assert train(config)


def test_cli_predict_and_export(tmp_path):
    main(
        [
            "predict",
            str(PAPER_CHECKPOINT),
            str(REAL_SAMPLES),
            "--output",
            str(tmp_path / "grid.png"),
            "--save-tensors",
            str(tmp_path / "pred.pt"),
        ]
    )
    assert (tmp_path / "grid.png").exists()
    assert torch.load(tmp_path / "pred.pt")["relative_flux"].shape == (9, 1, 256, 256)

    main(["export", str(PAPER_CHECKPOINT), str(tmp_path / "model.pt")])
    assert isinstance(load_model(tmp_path / "model.pt"), torch.jit.ScriptModule)
