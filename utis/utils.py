import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import math
import random

class AdamP(Optimizer):
    """
    AdamP optimizer. Copyright (c) 2020-present NAVER Corp.
    AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)
        x_norm = x.norm(dim=1).add_(eps)
        y_norm = y.norm(dim=1).add_(eps)
        dot = (x * y).sum(dim=1)
        return dot.abs() / x_norm / y_norm

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:
            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)
            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio
                return perturb, wd
        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])
                else:
                    wd_ratio = 1

                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                p.data.add_(perturb, alpha=-step_size)
        return loss

def gaussian_kernel(size, sigma):
    kernel_range = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel_values = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
    kernel = kernel_values.view(1, 1, size, 1) * kernel_values.view(1, 1, 1, size)
    return kernel / torch.sum(kernel)

def gaussian_filter_batch(input_batch, kernel_size=5, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma).to(input_batch.device)
    filtered_batch = F.conv2d(input_batch, kernel, padding=kernel_size // 2)
    return filtered_batch

def random_float(lo, hi):
    return lo + (hi - lo) * random.random()

def random_crop_and_resize(tensor, scale):
    _, _, h, w = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random.random() * delta)
    w_delta = int(random.random() * delta)
    cropped = tensor[:, :, h_delta:h_delta + new_width, w_delta:w_delta + new_width].clone()
    return F.interpolate(cropped, size=(h, w), mode="bilinear")

def random_vflip(tensor, prob):
    return tensor if prob > random.random() else torch.flip(tensor, dims=(2,))

def random_hflip(tensor, prob):
    return tensor if prob > random.random() else torch.flip(tensor, dims=(3,))

def random_transpose(tensor, prob):
    return tensor if prob > random.random() else tensor.permute(0, 1, 3, 2)

def calc_flux_loss(f1, f2):
    abs_diff = torch.abs(f1 - f2)
    loss = torch.sum(abs_diff) / torch.sum(f1)
    return loss
