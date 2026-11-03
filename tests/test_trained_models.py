"""The published models must keep working and produce unchanged results."""

import json
from pathlib import Path

import torch

from tests.paths import PAPER_CHECKPOINT, SCRIPTED_MODEL
from utis import FluxPredictor, load_model
from utis.export import export_torchscript
from utis.lightning_module import UTISModule
from utis.model import UNet3Plus, convert_legacy_state_dict

REFERENCE = json.loads((Path(__file__).parent / "data" / "paper_model_reference.json").read_text())


def test_paper_checkpoint_reproduces_original_outputs(real_samples):
    prediction = FluxPredictor.from_file(PAPER_CHECKPOINT).predict(real_samples)
    flux, background = prediction.relative_flux[:, 0].double(), prediction.background[:, 0].double()

    torch.testing.assert_close(flux.sum((1, 2)), torch.tensor(REFERENCE["flux_sum"], dtype=torch.float64))
    torch.testing.assert_close(background.sum((1, 2)), torch.tensor(REFERENCE["background_sum"], dtype=torch.float64))
    rows, cols = zip(*REFERENCE["pixels"])
    torch.testing.assert_close(flux[:, rows, cols], torch.tensor(REFERENCE["flux_pixels"], dtype=torch.float64))
    torch.testing.assert_close(
        background[:, rows, cols], torch.tensor(REFERENCE["background_pixels"], dtype=torch.float64)
    )


def test_scripted_model_weights_load_into_unet3plus(real_samples):
    scripted = load_model(SCRIPTED_MODEL)
    assert isinstance(scripted, torch.jit.ScriptModule)

    model = UNet3Plus()
    model.load_state_dict(convert_legacy_state_dict(scripted.state_dict()))
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(real_samples), scripted(real_samples), rtol=0, atol=0)


def test_reexported_torchscript_equals_published_model(real_samples, tmp_path):
    scripted = load_model(SCRIPTED_MODEL)
    model = UNet3Plus()
    model.load_state_dict(convert_legacy_state_dict(scripted.state_dict()))
    export_torchscript(model, tmp_path / "model.pt")
    reloaded = torch.jit.load(str(tmp_path / "model.pt"))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(real_samples[:2]), scripted(real_samples[:2]), rtol=0, atol=0)


def test_lightning_module_loads_legacy_checkpoint():
    module = UTISModule.load_from_checkpoint(PAPER_CHECKPOINT, map_location="cpu")
    reference = load_model(PAPER_CHECKPOINT)
    for (k1, v1), (k2, v2) in zip(module.model.state_dict().items(), reference.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
