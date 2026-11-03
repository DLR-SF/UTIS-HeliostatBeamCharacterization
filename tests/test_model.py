import pytest
import torch

from utis.model import UNet3Plus, convert_legacy_state_dict
from utis.model.legacy import _PREFIX_MAP


@pytest.mark.parametrize("base_channels", [1, 4, 32])
def test_output_shape_and_range(base_channels):
    model = UNet3Plus(base_channels=base_channels).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 2, 64, 64)
    assert out.min() >= 0 and out.max() <= 1


def test_torchscript_matches_eager():
    model = UNet3Plus(base_channels=4).eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        assert torch.equal(torch.jit.script(model)(x), model(x))


def test_legacy_map_covers_all_parameters():
    new_keys = {k.rpartition(".")[0] for k in UNet3Plus().state_dict()}
    assert new_keys == set(_PREFIX_MAP.values())


def test_conversion_is_idempotent():
    state_dict = UNet3Plus(base_channels=2).state_dict()
    converted = convert_legacy_state_dict({f"model.{k}": v for k, v in state_dict.items()})
    assert converted.keys() == state_dict.keys()


def test_unknown_legacy_key_raises():
    with pytest.raises(KeyError):
        convert_legacy_state_dict({"outconv1.weight": torch.zeros(1), "foo.weight": torch.zeros(1)})
