import numpy as np
import pytest
import torch
from PIL import Image

from utis.config import TrainConfig, load_config, save_config
from utis.data.io import load_image, save_image_16bit, to_grayscale
from utis.targets import MFT, STJ_LOWER, STJ_UPPER, TARGETS, get_target


def test_load_8bit_and_16bit(tmp_path):
    array8 = np.array([[0, 255], [51, 102]], dtype=np.uint8)
    Image.fromarray(array8).save(tmp_path / "a.png")
    torch.testing.assert_close(load_image(tmp_path / "a.png")[0], torch.from_numpy(array8 / 255.0).float())

    image = torch.rand(1, 8, 8)
    save_image_16bit(image, tmp_path / "b.png")
    torch.testing.assert_close(load_image(tmp_path / "b.png"), image, atol=1 / 65535, rtol=0)


def test_rgb_uses_paper_grayscale_weights(tmp_path):
    Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)).save(tmp_path / "rgb.png")
    torch.testing.assert_close(load_image(tmp_path / "rgb.png")[0, 0], torch.tensor([0.25, 0.5]))
    assert to_grayscale(np.ones((1, 1, 3))).item() == 1.0


def test_target_naming_follows_paint():
    assert STJ_UPPER == "solar_tower_juelich_upper"
    assert STJ_LOWER == "solar_tower_juelich_lower"
    assert MFT == "multi_focus_tower"
    assert get_target(STJ_LOWER) is TARGETS[STJ_LOWER]
    assert TARGETS[STJ_UPPER].flux_set == TARGETS[STJ_LOWER].flux_set != TARGETS[MFT].flux_set
    assert all(target.area for target in TARGETS.values())
    with pytest.raises(KeyError):
        get_target("unknown")


@pytest.mark.parametrize("name", ["paper", "multi_target"])
def test_shipped_configs_load(name):
    config = load_config(f"configs/{name}.yaml")
    assert all(t.resolved().flux_set for t in config.data.targets)


def test_overrides_and_roundtrip(tmp_path):
    config = load_config("configs/paper.yaml", ["trainer.max_epochs=3", "simulation.train_peak_range=[0.5, 0.6]"])
    assert config.trainer.max_epochs == 3
    assert config.simulation.train_peak_range == (0.5, 0.6)
    save_config(config, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == config
    assert load_config() == TrainConfig()


def test_unknown_config_key_raises():
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(overrides=["data.batchsize=2"])
