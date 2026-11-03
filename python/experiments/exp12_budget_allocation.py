"""Experiment 12 - Where is additional computation worth spending?

Continual re-solving must answer many river subgames (one per turn line x
river card), reached with very different probabilities. Given a fixed total
work budget, we compare allocation policies (budget_allocator.h):
  uniform   - equal work per subgame;
  weighted  - work prop. to reach probability;
  adaptive  - fitted convergence curves + optimism, measurement cost charged.
Metric: reach-weighted exact exploitability Σ_k w_k * expl_k (chips), which
is the subgames' contribution to full-game exploitability, at several budgets.
Subgames come from a turn solution of the Experiment 7/8 turn spots.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import color, plt, save_fig, save_json  # noqa: E402
from experiments.exp07_safe_resolving import R0, R1, turn_config  # noqa: E402

import aegis  # noqa: E402

NAME = "exp12_budget_allocation"
SCENARIOS = {"Kd9s6h3c": (100, 150), "Jh8c5d2s": (100, 150)}


def river_subgames(board, pot, stack, max_games=60):
    cfg = turn_config(board, pot, stack, [0.5, 1.0], [0.5, 1.0])
    tree = aegis.PublicTree.build(cfg)
    r0 = aegis.Range.parse(R0).remove(cfg.board)
    r1 = aegis.Range.parse(R1).remove(cfg.board)
    ctx = aegis.SubgameContext(tree, r0, r1, 8)
    s = aegis.VectorCFR(ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    s.iterate(150)
    pol = s.average_policy()
    hs = [ctx.hands(0), ctx.hands(1)]
    games, keep = [], []
    for root in aegis.street_roots(tree, 1):
        reach = aegis.reach_at(ctx, pol, root)
        rr = []
        for p in (0, 1):
            v = np.zeros(1326)
            v[np.array(hs[p].combos)] = reach[p]
            rr.append(v)
        nd = tree.node(root)
        board5 = tree.board(nd.board_id)
        pot_r, stack_r = tree.pot(root), tree.stack_remaining(root, 0)
        c = aegis.TreeConfig()
        c.board = board5
        c.pot, c.effective_stack = pot_r, stack_r
        rs = aegis.StreetSizes()
        rs.bet_fractions, rs.raise_fractions, rs.max_raises = [0.5, 1.0], [], 1
        c.set_river_sizes(rs)
        if rr[0].sum() <= 1e-9 or rr[1].sum() <= 1e-9:
            continue
        sub_tree = aegis.PublicTree.build(c)
        try:
            sub_ctx = aegis.SubgameContext(sub_tree, aegis.Range.from_array(rr[0]), aegis.Range.from_array(rr[1]))
        except RuntimeError:
            continue
        # Probability of reaching the subgame in the full game: compatible
        # reach mass of the hand pairs times the river card's chance (1/44).
        w = sub_ctx.normalizer / (ctx.normalizer * 44.0)
        games.append((sub_ctx, w))
        keep.append(sub_tree)
    games.sort(key=lambda g: -g[1])
    games = games[:max_games]
    return games, keep, ctx


def run():
    out = {}
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(11.5, 4.0))
    for ax, (board, (pot, stack)) in zip(axes, SCENARIOS.items()):
        games, keep, ctx = river_subgames(board, pot, stack)
        weights = np.array([w for _, w in games])
        base = sum(2.0 * 13 * (g.n(0) + g.n(1)) for g, _ in games)  # ~ work of one iteration on every subgame
        res = {"num_subgames": len(games), "weight_total": float(weights.sum()),
               "weight_top5_share": float(np.sort(weights)[::-1][:5].sum() / weights.sum()), "budgets": {}}
        for mult in (20, 40, 80, 160):
            row = {}
            for name, pol in (("uniform", aegis.BudgetPolicy.UNIFORM), ("weighted", aegis.BudgetPolicy.WEIGHTED),
                              ("adaptive", aegis.BudgetPolicy.ADAPTIVE)):
                o = aegis.BudgetOptions()
                o.total_work = base * mult
                r = aegis.allocate_budget([(g, w) for g, w in games], pol, o)
                row[name] = {"weighted_exploitability": r.weighted_exploitability, "work_used": r.work_used,
                             "measurements": r.measurements, "seconds": r.seconds,
                             "iterations": r.iterations}
            res["budgets"][mult] = row
            print(board, mult, {k: f"{v['weighted_exploitability']:.4f}" for k, v in row.items()}, flush=True)
        out[board] = res
        mults = sorted(res["budgets"], key=int)
        for name in ("uniform", "weighted", "adaptive"):
            ax.plot([int(m) for m in mults], [res["budgets"][m][name]["weighted_exploitability"] for m in mults],
                    color=color(name), marker="o", label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("budget (iterations per subgame, on average)")
        ax.set_ylabel("reach-weighted exploitability (chips)")
        ax.set_title(f"{board}: {len(games)} river subgames")
        ax.legend()
    save_fig(fig, NAME, "budget_allocation.png")
    save_json(NAME, "results.json", out)
    return out


if __name__ == "__main__":
    run()
