"""Conversion of state dicts from the original UNet 3+ implementation.

Two legacy layouts exist:

* ``trained_models/UNet3plus.ckpt`` (paper model): encoder blocks named
  ``conv1.conv1.0.weight``; Lightning prefix ``model.``.
* ``trained_models/utis_model_scripted.pt`` (multi-target model used by PAINT):
  encoder blocks named ``conv1.convs.0.0.weight``.

Both share the decoder names of the reference implementation
(``h1_PT_hd4_conv``, ``hd5_UT_hd4_bn``, ``conv4d_1``, ...).
"""

from __future__ import annotations

from typing import Dict, Mapping

import torch

from utis.model.unet3plus import NUM_LEVELS

_BLOCK_PARTS = {"0": "conv", "1": "bn"}


def _legacy_prefix_map() -> Dict[str, str]:
    """Map legacy module prefixes to the module prefixes of :class:`UNet3Plus`."""
    mapping: Dict[str, str] = {}
    for level in range(1, NUM_LEVELS + 1):
        for block in (1, 2):
            for index, part in _BLOCK_PARTS.items():
                new = f"encoders.{level - 1}.{block - 1}.{part}"
                mapping[f"conv{level}.conv{block}.{index}"] = new
                mapping[f"conv{level}.convs.{block - 1}.{index}"] = new

    for stage, level in enumerate(range(NUM_LEVELS - 1, 0, -1)):
        for source in range(1, NUM_LEVELS + 1):
            if source < level:
                legacy = f"h{source}_PT_hd{level}"
            elif source == level:
                legacy = f"h{source}_Cat_hd{level}"
            else:
                legacy = f"hd{source}_UT_hd{level}"
            for part in ("conv", "bn"):
                mapping[f"{legacy}_{part}"] = f"decoders.{stage}.skips.{source - 1}.{part}"
        mapping[f"conv{level}d_1"] = f"decoders.{stage}.fuse.conv"
        mapping[f"bn{level}d_1"] = f"decoders.{stage}.fuse.bn"

    mapping["outconv1"] = "head"
    return mapping


_PREFIX_MAP = _legacy_prefix_map()


def strip_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str = "model.") -> Dict[str, torch.Tensor]:
    """Remove ``prefix`` from all keys that carry it (e.g. the Lightning ``model.`` prefix)."""
    return {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def is_legacy_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Return ``True`` if the (prefix-free) state dict uses the original layer names."""
    return any(key.startswith("outconv1.") for key in state_dict)


def convert_legacy_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a state dict to the key layout of :class:`utis.model.UNet3Plus`.

    State dicts that already use the new layout are returned unchanged (apart
    from removing a ``model.`` prefix).
    """
    state_dict = strip_prefix(state_dict)
    if not is_legacy_state_dict(state_dict):
        return dict(state_dict)

    converted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        module, _, param = key.rpartition(".")
        if module not in _PREFIX_MAP:
            raise KeyError(f"Unexpected key {key!r} in legacy UNet 3+ state dict")
        converted[f"{_PREFIX_MAP[module]}.{param}"] = value
    return converted
