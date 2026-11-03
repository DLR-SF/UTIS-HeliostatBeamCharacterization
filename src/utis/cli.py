"""Command line interface: ``utis train | predict | export``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional


def _train(args: argparse.Namespace) -> None:
    from utis.config import load_config
    from utis.training import train

    config = load_config(args.config, args.set)
    best = train(config)
    if best:
        print(f"Best checkpoint: {best}")


def _predict(args: argparse.Namespace) -> None:
    import torch

    from utis.data.io import list_images, load_images
    from utis.inference import FluxPredictor
    from utis.visualization import save_result_grid

    inputs = [Path(p) for p in args.inputs]
    paths = [p for i in inputs for p in (list_images(i) if i.is_dir() else [i])]
    predictor = FluxPredictor.from_file(args.model, device=args.device)
    images = load_images(paths)
    prediction = predictor.predict(images)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_result_grid(output, images, prediction.relative_flux, prediction.background)
    print(f"Saved result grid of {len(paths)} images to {output}")
    if args.save_tensors:
        torch.save(
            {
                "files": [str(p) for p in paths],
                "relative_flux": prediction.relative_flux,
                "background": prediction.background,
            },
            args.save_tensors,
        )
        print(f"Saved predictions to {args.save_tensors}")


def _export(args: argparse.Namespace) -> None:
    import torch

    from utis.export import export_onnx, export_torchscript
    from utis.inference import load_model

    model = load_model(args.model)
    if isinstance(model, torch.jit.ScriptModule):
        raise SystemExit("The model is already a TorchScript model; export a checkpoint instead.")
    if args.format == "torchscript":
        export_torchscript(model, args.output)
    else:
        export_onnx(model, args.output, image_size=args.image_size)
    print(f"Exported {args.model} to {args.output} ({args.format})")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser of the ``utis`` command."""
    parser = argparse.ArgumentParser(prog="utis", description="UNet-based heliostat beam characterization")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model on synthetic target images")
    p.add_argument("--config", type=Path, help="YAML config, e.g. configs/paper.yaml")
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. trainer.max_epochs=10 data.batch_size=16",
    )
    p.set_defaults(func=_train)

    p = sub.add_parser("predict", help="predict relative flux and background for target images")
    p.add_argument("model", type=Path, help="checkpoint (.ckpt) or TorchScript model (.pt)")
    p.add_argument("inputs", nargs="+", help="image files and/or directories")
    p.add_argument("--output", default="results.png", help="result grid (input / flux / background)")
    p.add_argument("--save-tensors", metavar="FILE", help="additionally save the predictions with torch.save")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=_predict)

    p = sub.add_parser("export", help="export a checkpoint to TorchScript or ONNX")
    p.add_argument("model", type=Path, help="checkpoint (.ckpt) or state dict")
    p.add_argument("output", type=Path)
    p.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    p.add_argument("--image-size", type=int, default=256, help="input size for ONNX export")
    p.set_defaults(func=_export)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point of the ``utis`` command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
