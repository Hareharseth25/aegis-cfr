"""Shared utilities for AEGIS experiments: paths, JSON results, statistics and
a consistent plotting style.

Plot conventions (from the validated reference palette):
  * one y-axis per chart, recessive hairline grid, 2px lines, markers >= 8px;
  * every entity (algorithm / method) keeps one colour across all charts;
  * categorical slots are assigned in fixed order, never cycled;
  * >= 2 series always get a legend; text uses ink colours, never series colours.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import aegis  # noqa: E402,F401

RESULTS = os.path.join(ROOT, "results")

# Reference palette (light mode), categorical slots 1..8 in fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Entity -> slot. Colour follows the entity everywhere in the report.
ENTITY_SLOT = {
    "CFR": 0, "CFR(alt)": 0, "CFR+": 1, "DCFR": 2, "LCFR": 3, "CFR(sim)": 4, "ES-MCCFR": 5, "OS-MCCFR": 6,
    "ES-MCCFR x8": 7,
    # Experiment A modes
    "external": 0, "random": 1, "reach": 2, "adaptive": 3, "adaptive-noρ": 4, "adaptive-noU": 5,
    "adaptive-noG": 6, "adaptive-noC": 7,
    # resolving
    "blueprint": 0, "naive": 7, "resolve": 2, "maxmargin": 6, "translation": 1, "nested": 3, "reference-solve": 5,
    # refinement
    "hole": 3, "largest-gap": 1, "uniform": 0, "learned": 2,
    # budgets
    "weighted": 1,
    # backends
    "C++": 0, "torch-cpu": 1, "torch-mps": 2, "C++ (no SIMD)": 4,
    "pseudo-harmonic": 2, "nearest": 1, "abstract": 0,
}


def color(name: str, fallback: int = 0) -> str:
    return SLOTS[ENTITY_SLOT.get(name, fallback) % len(SLOTS)]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "figure.dpi": 110,
    })


style()


def outdir(name: str) -> str:
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    return d


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(name: str, file: str, data) -> str:
    path = os.path.join(outdir(name), file)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=_json_default)
    return path


def save_fig(fig, name: str, file: str) -> str:
    path = os.path.join(outdir(name), file)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def mean_ci(samples, axis=0):
    """Mean and 95% confidence half-width (Student t) along `axis`."""
    x = np.asarray(samples, dtype=float)
    n = x.shape[axis]
    m = x.mean(axis=axis)
    if n < 2:
        return m, np.zeros_like(m)
    from scipy import stats

    se = x.std(axis=axis, ddof=1) / math.sqrt(n)
    return m, stats.t.ppf(0.975, n - 1) * se


def log_checkpoints(total: int, first: int = 1, factor: float = 1.5) -> list[int]:
    pts, x = [], first
    while x < total:
        pts.append(int(x))
        x = max(x + 1, x * factor)
    pts.append(total)
    return sorted(set(pts))


def environment() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpu": None, "cores": os.cpu_count(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vectorized_build": bool(aegis.core.vectorized_build)}
    try:
        info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:  # noqa: BLE001
        pass
    return info


@contextmanager
def timed(label: str):
    t = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - t:.1f}s", flush=True)


def line_plot(ax, xs, ys, label, ci=None, marker="o", direct_label=False):
    c = color(label)
    ax.plot(xs, ys, color=c, label=label, marker=marker, markersize=4, markevery=max(1, len(xs) // 8))
    if ci is not None:
        ys = np.asarray(ys)
        ci = np.asarray(ci)
        ax.fill_between(xs, np.maximum(ys - ci, 1e-12), ys + ci, color=c, alpha=0.15, linewidth=0)
    if direct_label and len(xs):
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8,
                    va="center")
