"""Experiment 2 - Leduc comparison: exploitability vs time (algorithmic efficiency).

Same game, same wall-clock budget for every method. Deterministic CFR variants
run once; sampled methods (external / outcome sampling MCCFR, and 8-thread
batch-synchronous external sampling) run over 5 seeds and report the mean with
a 95% confidence interval. Exploitability is the exact NashConv/2 of the
average strategy (evaluation time excluded from the budget).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import line_plot, mean_ci, plt, save_fig, save_json  # noqa: E402

import aegis  # noqa: E402

NAME = "exp02_leduc_comparison"
SEEDS = [1, 2, 3, 4, 5]


def run_deterministic(tree, cfg, budget_s):
    cfr = aegis.TabularCFR(tree, cfg)
    t_used, chunk, pts = 0.0, 1, []
    while t_used < budget_s:
        t = time.perf_counter()
        cfr.iterate(chunk)
        t_used += time.perf_counter() - t
        pts.append((t_used, cfr.nodes_touched, cfr.iteration, aegis.exploitability(tree, cfr.average_policy())))
        chunk = max(1, int(chunk * 1.5))
    return pts


def run_sampled(tree, scheme, budget_s, seed, threads=1):
    c = aegis.MCCFRConfig()
    c.scheme = scheme
    c.seed = seed
    m = aegis.MCCFR(tree, c)
    t_used, chunk, pts = 0.0, 200, []
    while t_used < budget_s:
        t = time.perf_counter()
        if threads > 1:
            m.iterate_parallel(max(1, chunk // (50 * threads)), 50, threads)
        else:
            m.iterate(chunk)
        t_used += time.perf_counter() - t
        pts.append((t_used, m.nodes_touched, m.iteration, aegis.exploitability(tree, m.average_policy())))
        chunk = int(chunk * 1.5)
    return pts


def resample(pts, grid):
    """Exploitability at given times (step interpolation in log space)."""
    ts = np.array([p[0] for p in pts])
    es = np.array([p[3] for p in pts])
    out = []
    for g in grid:
        idx = np.searchsorted(ts, g, side="right") - 1
        out.append(es[idx] if idx >= 0 else np.nan)
    return np.array(out)


def run(budget_s: float = 6.0):
    results = {}
    games = {"leduc": aegis.LeducGame()}
    big = aegis.LeducConfig()
    big.num_ranks = 6
    games["leduc-6ranks"] = aegis.LeducGame(big)
    for gname, game in games.items():
        tree = aegis.GameTree.build(game)
        res = {"nodes": tree.num_nodes, "infosets": tree.num_infosets, "methods": {}}
        grid = np.geomspace(0.01, budget_s, 30)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for name, cfg in (("CFR(alt)", aegis.CFRConfig.vanilla(True)), ("CFR+", aegis.CFRConfig.cfr_plus()),
                          ("DCFR", aegis.CFRConfig.dcfr()), ("LCFR", aegis.CFRConfig.linear())):
            pts = run_deterministic(tree, cfg, budget_s)
            res["methods"][name] = {"points": pts}
            ys = resample(pts, grid)
            line_plot(ax, grid, ys, name)
            print(f"{gname} {name}: {pts[-1][3]:.2e} after {pts[-1][2]} it")
        for name, scheme, th in (("ES-MCCFR", aegis.MCCFRScheme.EXTERNAL, 1),
                                 ("OS-MCCFR", aegis.MCCFRScheme.OUTCOME, 1),
                                 ("ES-MCCFR x8", aegis.MCCFRScheme.EXTERNAL, 8)):
            runs = [run_sampled(tree, scheme, budget_s, s, th) for s in SEEDS]
            curves = np.array([resample(r, grid) for r in runs])
            m, ci = mean_ci(curves)
            res["methods"][name] = {"points": runs, "grid": grid.tolist(), "mean": m.tolist(), "ci95": ci.tolist()}
            line_plot(ax, grid, m, name, ci=ci)
            print(f"{gname} {name}: {m[-1]:.3e} ± {ci[-1]:.1e}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall-clock seconds (solver only)")
        ax.set_ylabel("exploitability (chips)")
        ax.set_title(f"{gname}: exploitability vs time ({tree.num_infosets} infosets)")
        ax.legend(ncol=2)
        save_fig(fig, NAME, f"{gname}_time.png")
        # Same data vs work (nodes touched).
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for name, d in res["methods"].items():
            pts = d["points"] if isinstance(d["points"][0], tuple) else d["points"][0]
            line_plot(ax, [p[1] for p in pts], [p[3] for p in pts], name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("nodes touched (seed 1 for sampled methods)")
        ax.set_ylabel("exploitability (chips)")
        ax.set_title(f"{gname}: exploitability vs work")
        ax.legend(ncol=2)
        save_fig(fig, NAME, f"{gname}_work.png")
        results[gname] = res
    save_json(NAME, "results.json", results)
    return results


if __name__ == "__main__":
    run()
