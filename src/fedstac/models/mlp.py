"""MLP body + linear head for tabular flow features."""
from __future__ import annotations

import torch
from torch import nn


def _norm(kind: str, d: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(d)
    if kind == "layer":
        return nn.LayerNorm(d)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden=(256, 128, 64), dropout: float = 0.1, norm: str = "batch"):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), _norm(norm, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def mlp_flops(in_dim: int, n_classes: int, hidden) -> int:
    """Multiply-accumulate x2 for linear layers (norm/activation ignored, <1%)."""
    dims = [in_dim, *hidden, n_classes]
    return int(sum(2 * a * b for a, b in zip(dims[:-1], dims[1:])))


def is_bn_key(name: str, model: nn.Module) -> bool:
    mod = name.rsplit(".", 1)[0]
    m = dict(model.named_modules()).get(mod)
    return isinstance(m, nn.BatchNorm1d)
