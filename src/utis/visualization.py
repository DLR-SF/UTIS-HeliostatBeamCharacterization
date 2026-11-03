"""Visualisation of predictions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from torchvision.utils import make_grid, save_image

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def image_grid(rows: Sequence[torch.Tensor], padding: int = 2) -> torch.Tensor:
    """Arrange equally sized batches (``B x 1 x H x W``) as rows of one grid image.

    Values are clipped to ``[0, 1]`` (no per-image normalisation).
    """
    rows = [r.detach().float().cpu().clamp(0, 1) for r in rows]
    n = rows[0].shape[0]
    return make_grid(torch.cat(rows, dim=0), nrow=n, padding=padding)


def save_result_grid(
    path: Union[str, Path], images: torch.Tensor, flux: torch.Tensor, background: torch.Tensor
) -> None:
    """Save a grid with the input images (first row), relative flux (second) and background (third)."""
    save_image(image_grid([images, flux, background]), str(path))


def plot_flux_overview(
    images: torch.Tensor,
    relative_flux: torch.Tensor,
    background: torch.Tensor,
    absolute_flux: Optional[torch.Tensor] = None,
    path: Optional[Union[str, Path]] = None,
    flux_unit_scale: float = 1e-3,
) -> Figure:
    """Figure in the style of Fig. 4 of the paper (requires ``matplotlib``).

    Rows: input ``I``, relative flux ``X_r``, background ``B`` (gray) and,
    if given, the absolute flux ``X`` in kW/m^2 (hot colormap).
    """
    import matplotlib.pyplot as plt

    rows = [("$I$", images), ("$X_r$", relative_flux), ("$B_{pred}$", background)]
    if absolute_flux is not None:
        rows.append(("$X$", absolute_flux * flux_unit_scale))
    n = images.shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.6 * n + 1, 1.6 * len(rows)), squeeze=False)
    vmax_abs = float(rows[-1][1].max()) if absolute_flux is not None else 1.0

    for r, (label, batch) in enumerate(rows):
        is_abs = absolute_flux is not None and r == len(rows) - 1
        for c in range(n):
            ax = axes[r, c]
            im = ax.imshow(
                batch[c].squeeze().detach().cpu().numpy(),
                cmap="hot" if is_abs else "gray",
                vmin=0,
                vmax=vmax_abs if is_abs else 1,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(label)
        cbar = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("kW m$^{-2}$" if is_abs else "[-]")

    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
