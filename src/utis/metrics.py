"""Evaluation metrics."""

from __future__ import annotations

import torch


def relative_flux_deviation(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative mean pixel deviation of the flux (Eq. 14): ``sum|X - X_hat| / sum|X_hat|``."""
    return (prediction - target).abs().sum() / target.abs().sum()


class RelativeFluxDeviation:
    """Accumulates Eq. (14) over several batches (numerator and denominator separately)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.abs_error = torch.zeros((), dtype=torch.float64)
        self.abs_target = torch.zeros((), dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.abs_error = self.abs_error.to(prediction.device) + (prediction - target).abs().sum().double()
        self.abs_target = self.abs_target.to(prediction.device) + target.abs().sum().double()

    def compute(self) -> torch.Tensor:
        return (self.abs_error / self.abs_target).float()
