"""AdamP optimizer.

AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant
Weights (Heo et al., ICLR 2021). Original implementation:
https://github.com/clovaai/AdamP

Copyright (c) 2020-present NAVER Corp. MIT license.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

ViewFunction = Callable[[torch.Tensor], torch.Tensor]


def _channel_view(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def _layer_view(x: torch.Tensor) -> torch.Tensor:
    return x.view(1, -1)


def _cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float, view: ViewFunction) -> torch.Tensor:
    x, y = view(x), view(y)
    x_norm = x.norm(dim=1).add_(eps)
    y_norm = y.norm(dim=1).add_(eps)
    return (x * y).sum(dim=1).abs() / x_norm / y_norm


class AdamP(Optimizer):
    """Adam with projection of updates for scale-invariant weights (e.g. before batch norm)."""

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "delta": delta,
            "wd_ratio": wd_ratio,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _projection(
        p: torch.Tensor, grad: torch.Tensor, perturb: torch.Tensor, delta: float, wd_ratio: float, eps: float
    ) -> Tuple[torch.Tensor, float]:
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view in (_channel_view, _layer_view):
            cosine_sim = _cosine_similarity(grad, p.data, eps, view)
            if cosine_sim.max() < delta / math.sqrt(view(p.data).size(1)):
                p_n = p.data / view(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view(p_n * perturb).sum(dim=1).view(expand_size)
                return perturb, wd_ratio
        return perturb, 1.0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Adam
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                numerator = beta1 * exp_avg + (1 - beta1) * grad if group["nesterov"] else exp_avg
                perturb = numerator / denom

                # Projection for scale-invariant weights
                wd_ratio = 1.0
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(
                        p, grad, perturb, group["delta"], group["wd_ratio"], group["eps"]
                    )

                # Weight decay
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss
