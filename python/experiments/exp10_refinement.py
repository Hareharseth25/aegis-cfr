"""Experiment B (exp10) - Dynamic action refinement.

Start from a coarse bet-size abstraction ({¾ pot} for both players), detect
where interpolating between sizes would materially change regret, add a new
legal size and re-solve. Methods choose one (player, size) per step from the
reference grid:

  hole        - regret-guided: the size whose translation hole (the bettor's
                gain from using it against the translated response, over its
                best in-abstraction action) is largest;
  largest-gap - split the largest log-ratio gap in the player's size set;
  uniform     - a fixed, evenly spread order (log-spaced), alternating players;
  random      - uniformly random candidate (3 seeds).

Every state is evaluated by EXACT exploitability in the reference game
(9 sizes + all-in) of the translated abstract solution, against the number of
abstract information sets and cumulative solve time. 8 river spots.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import color, mean_ci, plt, save_fig, save_json  # noqa: E402

import aegis  # noqa: E402

NAME = "exp10_refinement"
GRID = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0]
UNIFORM_ORDER = [0.33, 1.5, 0.5, 1.0, 0.25, 2.0, 0.66, 1.25]
SPOTS = [
    ("AsKh8c4d2s", "AA,KK,88,44,22,AK,AQ,AJ,A9s-A2s,KQ,KJs,QJs,JTs,T9s,98s,87s,76s,65s",
     "AQ,AJ,AT,A9-A2,KQ,KJ,KT,QQ-99,77-55,33,QJs,JTs,T9s", 100, 200),
    ("Jh9h6c3d2s", "JJ,99,66,AJ,KJ,QJs,JTs,T8s,87s,75s,Ah5h,Kh5h,AhQh,AA,KK", "AJ,KJ,QJ,JT,TT,88,77,QhTh,Ah4h,98s,T9s", 100, 200),
    ("QsQd7c4h2d", "QJ,AQ,KQ,77,44,AA,KK,A5s,65s,J9s,T9s,AJs", "AA-88,AK,AJ,KJ,JT,T9s,98s,87s,76s,A7s", 100, 150),
    ("Td8c5s3h2c", "TT,88,55,AT,KT,QT,JT,T9s,97s,64s,A4s,AA,KK", "AT,KT,QT,99,77,66,J9s,98s,76s,A5s,K8s", 60, 140),
    ("Ac7d6d4s3h", "A7,A6,77,66,44,33,AK,AQ,KQs,85s,98s,52s,Ad5d,Kd9d", "AJ,AT,A9,KK-88,55,Qd8d,Jd9d,T9s,65s,87s", 80, 160),
    ("KcJh5d5c2s", "KK,JJ,KJ,AK,KQ,K5s,A5s,65s,QTs,T9s,AQ", "KQ,KT,QJ,JT,AJ,99-66,44,33,A2s,76s", 100, 200),
    ("9s8s7h3d2c", "99,88,77,98,T6s,J9s,JTs,AsKs,AsQs,KsQs,AA,KK,A3s", "T9,J9,J8,QQ-TT,66,55,As5s,Ks4s,65s,A7s", 100, 250),
    ("AhKd5s4c2h", "AA,KK,55,AK,A5s,43s,Ah3h,AQ,KQ,QJs,Th9h", "AQ,AJ,KQ,KJ,QQ-JJ,66,33,63s,76s,Ah7h", 50, 150),
]


def make_refiner(spot, iterations=500):
    board, a, b, pot, stack = spot
    s = aegis.RefinementSpot()
    c = aegis.TreeConfig()
    c.board = aegis.parse_cards(board)
    c.pot, c.effective_stack = pot, stack
    ss = aegis.StreetSizes()
    ss.raise_fractions, ss.allow_all_in, ss.max_raises = [1.0], True, 2
    c.set_sizes(ss, ss)
    s.base = c
    s.r0 = aegis.Range.parse(a).remove(c.board)
    s.r1 = aegis.Range.parse(b).remove(c.board)
    s.reference_bets = GRID
    s.iterations = iterations
    return aegis.ActionRefiner(s)


def choose(method, ev, bets, rng):
    cands = [(c.player, c.fraction, c) for c in ev.candidates]
    if not cands:
        return None
    if method == "hole":
        return max(cands, key=lambda t: t[2].hole)[:2]
    if method == "random":
        return cands[rng.integers(len(cands))][:2]
    if method == "largest-gap":
        best, best_gap = None, -1.0
        for p, x, _ in cands:
            sizes = sorted(bets[p])
            lo = max([s for s in sizes if s < x], default=GRID[0] / 1.5)
            hi = min([s for s in sizes if s > x], default=GRID[-1] * 1.5)
            gap = math.log(hi / lo)
            mid_dist = abs(math.log(x) - 0.5 * (math.log(lo) + math.log(hi)))
            score = gap - 0.01 * mid_dist
            if score > best_gap:
                best_gap, best = score, (p, x)
        return best
    if method == "uniform":
        step = len(bets[0]) + len(bets[1]) - 2
        p = step % 2
        for x in UNIFORM_ORDER:
            if x not in bets[p]:
                return (p, x)
        return cands[0][:2]
    raise ValueError(method)


def trajectory(refiner, method, steps, seed=0):
    rng = np.random.default_rng(seed)
    bets = [[0.75], [0.75]]
    rows, solve_s = [], 0.0
    for k in range(steps + 1):
        ev = refiner.evaluate([sorted(bets[0]), sorted(bets[1])], True)
        solve_s += ev.solve_seconds
        rows.append({"step": k, "bets": [sorted(bets[0]), sorted(bets[1])], "ref_expl": ev.ref_exploitability_pct,
                     "infosets": ev.abs_infosets, "cum_solve_s": solve_s,
                     "max_hole_pct": max([c.hole_pct_pot for c in ev.candidates], default=0.0)})
        if k == steps:
            break
        pick = choose(method, ev, bets, rng)
        if pick is None:
            break
        bets[pick[0]].append(pick[1])
    return rows


def run(steps=5):
    methods = ["hole", "largest-gap", "uniform", "random"]
    out = {"grid": GRID, "spots": [s[0] for s in SPOTS], "trajectories": {m: [] for m in methods}}
    for spot in SPOTS:
        ref = make_refiner(spot)
        for m in methods:
            seeds = [0, 1, 2] if m == "random" else [0]
            for sd in seeds:
                out["trajectories"][m].append({"spot": spot[0], "seed": sd, "rows": trajectory(ref, m, steps, sd)})
        print(spot[0], {m: round(out["trajectories"][m][-1]["rows"][-1]["ref_expl"], 3) for m in methods}, flush=True)
    # Aggregate (normalised by each spot's starting exploitability so spots weigh equally).
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for m in methods:
        curves = np.array([[r["ref_expl"] / t["rows"][0]["ref_expl"] for r in t["rows"]] for t in out["trajectories"][m]])
        absol = np.array([[r["ref_expl"] for r in t["rows"]] for t in out["trajectories"][m]])
        info = np.array([[r["infosets"] for r in t["rows"]] for t in out["trajectories"][m]]).mean(0)
        mean, ci = mean_ci(curves)
        summary[m] = {"relative_mean": mean.tolist(), "relative_ci95": ci.tolist(),
                      "absolute_mean_pct": absol.mean(0).tolist(), "infosets_mean": info.tolist()}
        axes[0].plot(range(len(mean)), mean, color=color(m), marker="o", label=m)
        axes[0].fill_between(range(len(mean)), mean - ci, mean + ci, color=color(m), alpha=0.12, linewidth=0)
        axes[1].plot(info, absol.mean(0), color=color(m), marker="o", label=m)
    axes[0].set_xlabel("refinement steps (sizes added)")
    axes[0].set_ylabel("reference exploitability / initial")
    axes[0].set_title("Exploitability after each refinement (mean ± 95% CI)")
    axes[0].legend()
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("abstract information sets")
    axes[1].set_ylabel("reference exploitability (% pot)")
    axes[1].set_title("Exploitability vs tree size")
    axes[1].legend()
    save_fig(fig, NAME, "refinement.png")
    # Paired comparison: hole vs each baseline after all steps.
    final = {m: np.array([t["rows"][-1]["ref_expl"] / t["rows"][0]["ref_expl"] for t in out["trajectories"][m]])
             for m in methods}
    rnd = final["random"].reshape(len(SPOTS), 3).mean(1)
    paired = {}
    for m in ("largest-gap", "uniform"):
        d = final["hole"] - final[m]
        dm, dci = mean_ci(d[:, None])
        paired[m] = {"mean_diff": float(dm[0]), "ci95": float(dci[0]), "hole_better_spots": int((d < 0).sum())}
    d = final["hole"] - rnd
    dm, dci = mean_ci(d[:, None])
    paired["random"] = {"mean_diff": float(dm[0]), "ci95": float(dci[0]), "hole_better_spots": int((d < 0).sum())}
    out["summary"], out["paired_hole_minus_baseline"] = summary, paired
    print("paired (hole - baseline, relative exploitability):", paired)
    save_json(NAME, "results.json", out)
    return out


if __name__ == "__main__":
    run()
