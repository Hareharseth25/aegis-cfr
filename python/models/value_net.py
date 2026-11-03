"""Counterfactual value network (DeepStack-style) in PyTorch.

Input  x = [board one-hot (52) | r̂_0 (1326) | r̂_1 (1326) | pot share | log(1+SPR)]
Output y = [v_0 (1326) | v_1 (1326)]  - per-hand counterfactual values of both
players, normalised by pot and opponent range mass. The layout is shared with
the C++ runtime (cpp/value_net/mlp.h) which loads exported weights.
"""
from __future__ import annotations

import torch
from torch import nn

import aegis

INPUT = aegis.NN_INPUT_SIZE
OUTPUT = aegis.NN_OUTPUT_SIZE
NC = 1326
BOARD = 52


class ValueNet(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (512, 512)):
        super().__init__()
        layers, d = [], INPUT
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, OUTPUT))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def linear_layers(self):
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def to_aegis(self) -> "aegis.MLP":
        ws, bs, shapes = [], [], []
        for lin in self.linear_layers():
            w = lin.weight.detach().cpu().float().contiguous()
            ws.append(w.reshape(-1).tolist())
            bs.append(lin.bias.detach().cpu().float().tolist())
            shapes.append((w.shape[0], w.shape[1]))
        return aegis.MLP.from_arrays(ws, bs, shapes)


def ranges_from_features(x: torch.Tensor):
    return x[:, BOARD:BOARD + NC], x[:, BOARD + NC:BOARD + 2 * NC]


def zero_sum_residual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Σ r̂_0 v_0 + Σ r̂_1 v_1 per sample (0 for exactly zero-sum values)."""
    r0, r1 = ranges_from_features(x)
    return (r0 * y[:, :NC]).sum(1) + (r1 * y[:, NC:]).sum(1)


def zero_sum_project(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Shift each player's values by half the residual (DeepStack correction)."""
    r0, r1 = ranges_from_features(x)
    e = zero_sum_residual(x, y)
    m0 = r0.sum(1).clamp_min(1e-12)
    m1 = r1.sum(1).clamp_min(1e-12)
    out = y.clone()
    out[:, :NC] -= (0.5 * e / m0)[:, None]
    out[:, NC:] -= (0.5 * e / m1)[:, None]
    return out
