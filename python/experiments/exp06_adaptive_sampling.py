"""Experiment A (exp06) - Adaptive trajectory sampling for MCCFR.

Hypothesis: allocating samples by strategic sensitivity and uncertainty gives
lower exploitability per unit of compute than uniform (external) sampling.

All methods share one importance-sampled MCCFR implementation (likelihood
ratios keep every estimator unbiased; tested), and differ only in the
sampling distribution at chance/opponent nodes:
  external  - q = p (standard external sampling, λ = 0);
  random    - q = (1-λ)p + λ*uniform;
  reach     - priority prop. to estimated reach only;
  adaptive  - S(I) = (ε+ρ)^α (ε+U)^β (ε+G)^γ / (ε+C)^δ;
  ablations - adaptive with one factor removed (α, β, γ or δ = 0).

Protocol (matched budgets, equal tuning):
  1. Tune λ ∈ {0.1, 0.3, 0.6} for every non-external method on Leduc with
     tuning seeds {101, 102, 103}; pick the best mean final exploitability.
  2. Evaluate on fresh seeds 1..8 with a fixed node-touch budget on Leduc and
     on an unseen larger game (Leduc with 5 ranks) - no re-tuning.
  3. Report exploitability vs nodes touched and vs seconds (mean ± 95% CI),
     estimator efficiency (MSE x nodes per sample at a frozen strategy),
     coverage, and throughput overhead.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import color, line_plot, mean_ci, plt, save_fig, save_json  # noqa: E402

import aegis  # noqa: E402

NAME = "exp06_adaptive_sampling"
LAMBDAS = [0.1, 0.3, 0.6]
TUNE_SEEDS = [101, 102, 103]
EVAL_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]


def params(method: str, lam: float) -> tuple:
    pp = aegis.PriorityParams()
    pp.lambda_ = lam
    mode = {"external": aegis.AllocationMode.EXTERNAL, "random": aegis.AllocationMode.RANDOM,
            "reach": aegis.AllocationMode.REACH}.get(method, aegis.AllocationMode.ADAPTIVE)
    if method == "adaptive-noρ":
        pp.alpha = 0.0
    elif method == "adaptive-noU":
        pp.beta = 0.0
    elif method == "adaptive-noG":
        pp.gamma = 0.0
    elif method == "adaptive-noC":
        pp.delta = 0.0
    return mode, pp


def run_one(tree, method, lam, seed, budget, n_points=14):
    c = aegis.MCCFRConfig()
    c.scheme = aegis.MCCFRScheme.IMPORTANCE
    c.seed = seed
    m = aegis.MCCFR(tree, c)
    mode, pp = params(method, lam)
    s = aegis.AdaptiveSampler(tree, m, mode, pp)
    m.set_sampling_policy(s)
    marks = np.geomspace(budget / 200, budget, n_points)
    pts, t_used, k = [], 0.0, 0
    chunk = 100
    while k < len(marks):
        t = time.perf_counter()
        m.iterate(chunk)
        t_used += time.perf_counter() - t
        while k < len(marks) and m.nodes_touched >= marks[k]:
            cover = float((m.infoset_updates > 0).mean())
            pts.append({"nodes": float(m.nodes_touched), "seconds": t_used,
                        "exploitability": aegis.exploitability(tree, m.average_policy()), "coverage": cover})
            k += 1
        chunk = min(int(chunk * 1.3) + 1, 20000)
    return pts


def curve(runs, key="exploitability", x="nodes"):
    grid = np.array([p[x] for p in runs[0]])
    ys = np.array([[p[key] for p in r] for r in runs])
    return grid, ys


def efficiency(tree, method, lam, seed=7, warm=20000, samples=4000):
    """MSE of the sampled regret estimator x mean nodes per sample at a frozen σ."""
    c = aegis.MCCFRConfig()
    c.scheme = aegis.MCCFRScheme.IMPORTANCE
    c.seed = seed
    m = aegis.MCCFR(tree, c)
    mode, pp = params(method, lam)
    s = aegis.AdaptiveSampler(tree, m, mode, pp)
    m.set_sampling_policy(s)
    m.iterate(warm)
    exact = aegis.counterfactual_regrets(tree, m.current_policy(), 0)
    mask = np.array([tree.infoset(I).player == 0 for I in range(tree.num_infosets)
                     for _ in range(tree.infoset(I).num_actions)])
    before = m.nodes_touched
    se = 0.0
    for _ in range(samples):
        r = m.sample_regret_estimate(0)
        se += float(np.sum((r[mask] - exact[mask]) ** 2))
    nodes_per = (m.nodes_touched - before) / samples
    mse = se / samples
    return {"mse": mse, "nodes_per_sample": nodes_per, "mse_x_cost": mse * nodes_per}


def run(budget=2.0e7, tune_budget=1.0e7):
    leduc = aegis.GameTree.build(aegis.LeducGame())
    cfg5 = aegis.LeducConfig()
    cfg5.num_ranks = 5
    leduc5 = aegis.GameTree.build(aegis.LeducGame(cfg5))
    methods = ["external", "random", "reach", "adaptive", "adaptive-noρ", "adaptive-noU", "adaptive-noG", "adaptive-noC"]
    out = {"tuning": {}, "chosen_lambda": {}, "eval": {}, "efficiency": {}, "throughput": {}}
    # 1. Equal tuning.
    for method in methods:
        if method == "external":
            out["chosen_lambda"][method] = 0.0
            continue
        scores = {}
        for lam in LAMBDAS:
            finals = [run_one(leduc, method, lam, s, tune_budget, 4)[-1]["exploitability"] for s in TUNE_SEEDS]
            scores[lam] = float(np.mean(finals))
        best = min(scores, key=scores.get)
        out["tuning"][method] = scores
        out["chosen_lambda"][method] = best
        print(f"tuned {method}: {scores} -> λ={best}", flush=True)
    # 2. Evaluation on fresh seeds; Leduc and unseen Leduc-5.
    for gname, tree, b in (("leduc", leduc, budget), ("leduc-5ranks", leduc5, 2 * budget)):
        res = {}
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
        for method in methods:
            lam = out["chosen_lambda"][method]
            runs = [run_one(tree, method, lam, s, b) for s in EVAL_SEEDS]
            grid, ys = curve(runs)
            m, ci = mean_ci(ys)
            secs = np.array([[p["seconds"] for p in r] for r in runs]).mean(axis=0)
            cov = np.array([[p["coverage"] for p in r] for r in runs]).mean(axis=0)
            res[method] = {"lambda": lam, "nodes": grid.tolist(), "mean": m.tolist(), "ci95": ci.tolist(),
                           "seconds": secs.tolist(), "coverage": cov.tolist(),
                           "final_all_seeds": ys[:, -1].tolist()}
            dashed = method.startswith("adaptive-")
            ax.plot(grid, m, color=color(method), linestyle="--" if dashed else "-", marker="o", markersize=3,
                    label=f"{method} (λ={lam})")
            if not dashed:
                ax.fill_between(grid, np.maximum(m - ci, 1e-9), m + ci, color=color(method), alpha=0.12, linewidth=0)
                ax2.plot(secs, m, color=color(method), marker="o", markersize=3, label=method)
            print(f"{gname} {method}: final {m[-1]:.4f} ± {ci[-1]:.4f}", flush=True)
        # Paired comparison vs external sampling at the final budget.
        ext = np.array(res["external"]["final_all_seeds"])
        for method in methods[1:]:
            d = np.array(res[method]["final_all_seeds"]) - ext
            dm, dci = mean_ci(d[:, None])
            res[method]["paired_diff_vs_external"] = {"mean": float(dm[0]), "ci95": float(dci[0]),
                                                      "wins": int((d < 0).sum()), "seeds": len(d)}
        out["eval"][gname] = res
        for a, xl, fname in ((ax, "nodes touched", "nodes"), (ax2, "solver seconds", "seconds")):
            a.set_xscale("log")
            a.set_yscale("log")
            a.set_xlabel(xl)
            a.set_ylabel("exploitability (chips), mean of 8 seeds")
            a.set_title(f"{gname}: adaptive sampling vs baselines")
            a.legend(fontsize=7.5, ncol=2)
        save_fig(fig, NAME, f"{gname}_vs_nodes.png")
        save_fig(fig2, NAME, f"{gname}_vs_seconds.png")
    # 3. Estimator efficiency and throughput on Leduc.
    for method in ["external", "random", "reach", "adaptive"]:
        lam = out["chosen_lambda"][method]
        out["efficiency"][method] = efficiency(leduc, method, lam)
        r = out["eval"]["leduc"][method]
        out["throughput"][method] = r["nodes"][-1] / r["seconds"][-1]
        print(f"efficiency {method}: {out['efficiency'][method]}  throughput {out['throughput'][method]:.3g} nodes/s")
    save_json(NAME, "results.json", out)
    return out


if __name__ == "__main__":
    run()
