"""Export of trained networks to TorchScript and ONNX."""

from __future__ import annotations

from typing import Union

import torch
from torch import nn

from utis.data.io import PathLike


def export_torchscript(model: nn.Module, path: PathLike) -> torch.jit.ScriptModule:
    """Script ``model`` and save it. The result can be loaded without UTIS via ``torch.jit.load``."""
    scripted = torch.jit.script(model.cpu().eval())
    scripted.save(str(path))
    return scripted


def export_onnx(model: nn.Module, path: PathLike, image_size: Union[int, tuple] = 256, opset_version: int = 17) -> None:
    """Export ``model`` to ONNX with a dynamic batch dimension (input ``image``, output ``prediction``)."""
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    dummy = torch.zeros(1, 1, height, width)
    torch.onnx.export(
        model.cpu().eval(),
        dummy,
        str(path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["prediction"],
        dynamic_axes={"image": {0: "batch"}, "prediction": {0: "batch"}},
    )
