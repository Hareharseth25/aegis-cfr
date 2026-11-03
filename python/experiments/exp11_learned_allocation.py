"""Experiment C (exp11) - Learned computation allocation.

Train a lightweight model to predict which refinement (adding bet size x for
player p) yields the largest reduction in exploitability per millisecond,
using measurements from small games where exact best responses are cheap,
then test whether it generalises to larger subgames it hasn't seen.

  * Training spots: narrow ranges (~20-60 combos), SPR ~ 1-1.5, random boards.
    At each state along refinement trajectories, every candidate is applied
    and its true effect measured: y = (Δ reference exploitability, % pot) /
    (re-solve milliseconds). Features use only information available before
    the refinement (translation hole, reach, gaps, size, SPR, ...).
  * Model: 2-layer MLP (PyTorch) on standardised features, predicting the
    rank-transformed target (robust to heavy tails).
  * Baselines, tuned EQUALLY on the training spots: hole heuristic (score =
    hole * reachᵏ / costᵐ with k, m chosen on training data), largest-gap,
    random. Test spots: wider ranges, deeper stacks, unseen boards.
  * Metrics: Spearman correlation of predicted vs actual gain on test
    candidates; reference exploitability after K steps and cumulative time.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import color, mean_ci, plt, save_fig, save_json  # noqa: E402
from experiments.exp10_refinement import GRID, choose  # noqa: E402

import aegis  # noqa: E402

NAME = "exp11_learned_allocation"
RANKS = "23456789TJQKA"


def random_spot(rng, n_classes, spr_range):
    deck = rng.permutation(52)
    board = [int(c) for c in deck[:5]]
    bm = set(board)
    vals = aegis.board_hand_values(board)
    live = [i for i in range(1326) if not (set(aegis.combo_cards()[i].tolist()) & bm)]
    order = sorted(live, key=lambda i: -int(vals[i]))
    ranges = []
    for _ in range(2):
        r = np.zeros(1326)
        # strength-correlated subset: a value region plus some bluffs/draws
        k = n_classes
        top = order[: int(len(order) * rng.uniform(0.15, 0.35))]
        bot = order[int(len(order) * 0.6):]
        pick = list(rng.choice(top, size=min(k, len(top)), replace=False)) + \
            list(rng.choice(bot, size=min(k // 3, len(bot)), replace=False))
        r[pick] = rng.uniform(0.3, 1.0, size=len(pick))
        ranges.append(r)
    pot = 100.0
    stack = float(pot * rng.uniform(*spr_range))
    return board, ranges, pot, stack


def make_refiner(board, ranges, pot, stack, iterations):
    s = aegis.RefinementSpot()
    c = aegis.TreeConfig()
    c.board = board
    c.pot, c.effective_stack = pot, stack
    ss = aegis.StreetSizes()
    ss.raise_fractions, ss.allow_all_in, ss.max_raises = [1.0], True, 2
    c.set_sizes(ss, ss)
    s.base = c
    s.r0 = aegis.Range.from_array(ranges[0])
    s.r1 = aegis.Range.from_array(ranges[1])
    s.reference_bets = GRID
    s.iterations = iterations
    return aegis.ActionRefiner(s)


def features(c, ev, bets, spr):
    return [c.hole_pct_pot, math.log(c.hole_pct_pot + 1e-4), c.reach_mass, c.gap_low, c.gap_high,
            math.log(c.fraction), float(c.player), float(len(bets[c.player])), math.log(spr),
            ev.ref_exploitability_pct, c.hole_pct_pot / max(ev.ref_exploitability_pct, 1e-6)]


FEATURE_NAMES = ["hole%", "log hole", "reach", "gap_low", "gap_high", "log x", "player", "#sizes", "log SPR",
                 "current expl%", "hole/expl"]


def measure_state(ref, bets, spr):
    """Evaluate the state, then every candidate's true effect."""
    ev = ref.evaluate([sorted(bets[0]), sorted(bets[1])], True)
    rows = []
    for c in ev.candidates:
        nb = [list(bets[0]), list(bets[1])]
        nb[c.player].append(c.fraction)
        ev2 = ref.evaluate([sorted(nb[0]), sorted(nb[1])], False)
        ms = 1e3 * ev2.solve_seconds
        gain = ev.ref_exploitability_pct - ev2.ref_exploitability_pct
        rows.append({"x": features(c, ev, bets, spr), "gain": gain, "ms": ms, "gain_per_ms": gain / max(ms, 1e-3),
                     "player": c.player, "fraction": c.fraction, "hole": c.hole_pct_pot, "reach": c.reach_mass})
    return ev, rows


def collect(n_spots, seed, n_classes, spr_range, iterations, states_per_spot=3):
    rng = np.random.default_rng(seed)
    data = []
    for k in range(n_spots):
        board, ranges, pot, stack = random_spot(rng, n_classes, spr_range)
        ref = make_refiner(board, ranges, pot, stack, iterations)
        bets = [[0.75], [0.75]]
        for s in range(states_per_spot):
            ev, rows = measure_state(ref, bets, stack / pot)
            for r in rows:
                r["spot"], r["state"] = k, s
            data += rows
            if not rows:
                break
            pick = rows[int(rng.integers(len(rows)))] if s % 2 else max(rows, key=lambda r: r["hole"])
            bets[pick["player"]].append(pick["fraction"])
    return data


class Scorer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32),
                                       torch.nn.ReLU(), torch.nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(data, epochs=400, seed=0):
    torch.manual_seed(seed)
    X = np.array([d["x"] for d in data], dtype=np.float32)
    y = np.array([d["gain_per_ms"] for d in data], dtype=np.float32)
    # Within-state rank targets (what matters for choosing is the ordering).
    groups = {}
    for i, d in enumerate(data):
        groups.setdefault((d["spot"], d["state"]), []).append(i)
    yr = np.zeros_like(y)
    for idx in groups.values():
        ranks = stats.rankdata(y[idx]) / len(idx)
        yr[idx] = ranks
    mu, sd = X.mean(0), X.std(0) + 1e-6
    model = Scorer(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(yr)
    for _ in range(epochs):
        loss = ((model(Xt) - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, mu, sd


def predict(model, mu, sd, xs):
    with torch.no_grad():
        return model(torch.tensor((np.array(xs, dtype=np.float32) - mu) / sd)).numpy()


def tune_hole(data):
    """Equal tuning for the heuristic: score = hole * reach^k / (#sizes)^m."""
    best, best_rho = (0, 0), -2
    groups = {}
    for d in data:
        groups.setdefault((d["spot"], d["state"]), []).append(d)
    for k in (0.0, 0.5, 1.0):
        for m in (0.0, 0.5, 1.0):
            hits = []
            for rows in groups.values():
                s = [r["hole"] * (r["reach"] + 1e-3) ** k / (r["x"][7] ** m) for r in rows]
                hits.append(rows[int(np.argmax(s))]["gain_per_ms"] / max(max(r["gain_per_ms"] for r in rows), 1e-9))
            rho = float(np.mean(hits))
            if rho > best_rho:
                best_rho, best = rho, (k, m)
    return best, best_rho


def run_controller(ref, spr, policy, steps, model=None, hole_params=(0, 0), seed=0):
    rng = np.random.default_rng(seed)
    bets = [[0.75], [0.75]]
    traj, t_total = [], 0.0
    for s in range(steps + 1):
        ev = ref.evaluate([sorted(bets[0]), sorted(bets[1])], True)
        t_total += ev.solve_seconds
        traj.append({"step": s, "ref_expl": ev.ref_exploitability_pct, "cum_solve_s": t_total})
        if s == steps or not ev.candidates:
            break
        if policy == "learned":
            xs = [features(c, ev, bets, spr) for c in ev.candidates]
            c = ev.candidates[int(np.argmax(predict(*model, xs)))]
            pick = (c.player, c.fraction)
        elif policy == "hole":
            k, m = hole_params
            c = max(ev.candidates, key=lambda c: c.hole * (c.reach_mass + 1e-3) ** k / (len(bets[c.player]) ** m))
            pick = (c.player, c.fraction)
        else:
            pick = choose(policy, ev, bets, rng)
        bets[pick[0]].append(pick[1])
    return traj


def run(steps=4):
    t0 = time.perf_counter()
    train = collect(40, seed=1, n_classes=18, spr_range=(0.8, 1.5), iterations=250, states_per_spot=4)
    print(f"collected {len(train)} training candidates in {time.perf_counter() - t0:.0f}s", flush=True)
    model = fit(train)
    hole_params, hole_train_score = tune_hole(train)
    # Test: larger, unseen subgames (wider ranges, deeper stacks, new boards).
    test_rng_seed = 99
    test_spots = []
    rng = np.random.default_rng(test_rng_seed)
    for _ in range(6):
        test_spots.append(random_spot(rng, 70, (1.8, 3.0)))
    out = {"train_candidates": len(train), "feature_names": FEATURE_NAMES, "hole_params": hole_params,
           "hole_train_score": hole_train_score, "test": {}}
    # Generalisation of the ranking: Spearman on the first state of each test spot.
    rhos_learned, rhos_hole = [], []
    for k, (board, ranges, pot, stack) in enumerate(test_spots):
        ref = make_refiner(board, ranges, pot, stack, 500)
        ev, rows = measure_state(ref, [[0.75], [0.75]], stack / pot)
        if len(rows) > 2:
            pred = predict(*model, [r["x"] for r in rows])
            actual = [r["gain_per_ms"] for r in rows]
            rhos_learned.append(stats.spearmanr(pred, actual).correlation)
            rhos_hole.append(stats.spearmanr([r["hole"] for r in rows], actual).correlation)
    out["spearman_test_first_state"] = {"learned": rhos_learned, "hole": rhos_hole,
                                        "learned_mean": float(np.nanmean(rhos_learned)),
                                        "hole_mean": float(np.nanmean(rhos_hole))}
    print("Spearman (test):", out["spearman_test_first_state"], flush=True)
    policies = ["learned", "hole", "largest-gap", "random"]
    finals = {p: [] for p in policies}
    for k, (board, ranges, pot, stack) in enumerate(test_spots):
        ref = make_refiner(board, ranges, pot, stack, 500)
        for p in policies:
            trajs = [run_controller(ref, stack / pot, p, steps, model, hole_params, sd)
                     for sd in ([0, 1, 2] if p == "random" else [0])]
            out["test"].setdefault(p, []).append({"spot": k, "trajectories": trajs})
            finals[p].append(np.mean([t[-1]["ref_expl"] / t[0]["ref_expl"] for t in trajs]))
        print(f"test spot {k}: " + ", ".join(f"{p}={finals[p][-1]:.3f}" for p in policies), flush=True)
    out["final_relative_exploitability"] = {p: {"per_spot": finals[p], "mean": float(np.mean(finals[p]))}
                                            for p in policies}
    paired = {}
    for p in ("hole", "largest-gap", "random"):
        d = np.array(finals["learned"]) - np.array(finals[p])
        m, ci = mean_ci(d[:, None])
        paired[p] = {"mean_diff": float(m[0]), "ci95": float(ci[0]), "learned_better_spots": int((d < 0).sum())}
    out["paired_learned_minus_baseline"] = paired
    print("paired (learned - baseline):", paired, flush=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for p in policies:
        curves = np.array([[r["ref_expl"] / s["trajectories"][0][0]["ref_expl"] for r in s["trajectories"][0]]
                           for s in out["test"][p]])
        m, ci = mean_ci(curves)
        ax.plot(range(len(m)), m, color=color(p), marker="o", label=p)
        ax.fill_between(range(len(m)), m - ci, m + ci, color=color(p), alpha=0.12, linewidth=0)
    ax.set_xlabel("refinement steps")
    ax.set_ylabel("reference exploitability / initial")
    ax.set_title("Unseen larger subgames: learned vs heuristic allocation")
    ax.legend()
    save_fig(fig, NAME, "learned_allocation.png")
    save_json(NAME, "results.json", out)
    return out


if __name__ == "__main__":
    run()
