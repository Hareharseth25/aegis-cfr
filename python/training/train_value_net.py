"""Train the counterfactual value network on solver-generated river positions.

Loss (masked to live combos):
    L = Σ_h w_h (V_θ(x, h) - V_solver(x, h))²  +  λ_zs * (Σ r̂_0 v_0 + Σ r̂_1 v_1)²
The second term regularises towards zero-sum aggregate values (the payoff
convention is zero-sum, so exact values satisfy it).

Usage (library): train(data_train, data_val, ...) -> (model, history)
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from models.value_net import NC, ValueNet, zero_sum_project, zero_sum_residual


def _tensors(d, device):
    return (torch.tensor(d["features"], device=device), torch.tensor(d["targets"], device=device),
            torch.tensor(d["mask"], device=device))


def evaluate(model, data, device="cpu", project=False, batch=1024) -> dict:
    model.eval()
    x, y, m = _tensors(data, device)
    se = ae = n = 0.0
    zs_raw = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb, yb, mb = x[i:i + batch], y[i:i + batch], m[i:i + batch]
            p = model(xb)
            zs_raw.append(zero_sum_residual(xb, p).abs().cpu())
            if project:
                p = zero_sum_project(xb, p)
            d = (p - yb) * mb
            se += float((d ** 2).sum())
            ae += float(d.abs().sum())
            n += float(mb.sum())
    tgt = y[m > 0]
    return {"rmse": math.sqrt(se / n), "mae": ae / n, "target_std": float(tgt.std()),
            "zero_sum_residual_mean_abs": float(torch.cat(zs_raw).mean())}


def train(train_data, val_data, hidden=(512, 512), epochs=40, batch=256, lr=1e-3, zs_weight=1.0, device=None,
          seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cpu")
    model = ValueNet(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    x, y, m = _tensors(train_data, device)
    n = x.shape[0]
    history = []
    best, best_state = float("inf"), None
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb, mb = x[idx], y[idx], m[idx]
            p = model(xb)
            mse = (((p - yb) * mb) ** 2).sum() / mb.sum()
            zs = (zero_sum_residual(xb, p) ** 2).mean()
            loss = mse + zs_weight * zs
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * xb.shape[0]
        sched.step()
        val = evaluate(model, val_data, device)
        history.append({"epoch": ep + 1, "train_loss": tot / n, **{f"val_{k}": v for k, v in val.items()},
                        "seconds": time.perf_counter() - t0})
        if val["rmse"] < best:
            best = val["rmse"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 4 or ep == epochs - 1):
            print(f"  epoch {ep + 1:3d}  train {tot / n:.5f}  val rmse {val['rmse']:.5f}  "
                  f"(target std {val['target_std']:.4f})", flush=True)
    model.load_state_dict(best_state)
    return model, history


def split(data, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == data["features"].shape[0] else v)
            for k, v in data.items()}


__all__ = ["train", "evaluate", "split", "NC"]
