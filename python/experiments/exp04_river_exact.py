"""Experiment 4 - Exact river solving.

Heads-up No-Limit river spots with weighted ranges and custom bet sizes:
solve for mixed strategies and measure the exact best-response
exploitability of the average strategy in the river game as built.

Outputs:
  * convergence (exploitability % pot vs iterations and seconds) for the
    demo spot and for random spots with full / medium / narrow ranges;
  * the demo spot's per-hand strategy for the hero hand and range summaries;
  * cross-validation: vector solver vs generic tabular solver on a reduced
    range (identical average strategies and NashConv).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import line_plot, log_checkpoints, plt, save_fig, save_json  # noqa: E402

import aegis  # noqa: E402
from aegis.__main__ import DEMO  # noqa: E402
from aegis.spot import solve_spot  # noqa: E402

NAME = "exp04_river_exact"


def cross_validate():
    c = aegis.TreeConfig()
    c.board = aegis.parse_cards("AsKh8c4d2s")
    c.pot, c.effective_stack = 100, 150
    s = aegis.StreetSizes()
    s.bet_fractions = [0.33, 0.75]
    s.raise_fractions = [1.0]
    s.max_raises = 2
    c.set_sizes(s, s)
    tree = aegis.PublicTree.build(c)
    r0 = aegis.Range.parse("AA,KK,88,AK,AJs,QJs,65s").remove(c.board)
    r1 = aegis.Range.parse("AQ,KQ,QQ,99,T9s,A5s").remove(c.board)
    ctx = aegis.SubgameContext(tree, r0, r1)
    efg = aegis.HoldemSubgameGame(ctx)
    gt = aegis.GameTree.build(efg)
    vec = aegis.VectorCFR(ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    tab = aegis.TabularCFR(gt, aegis.CFRConfig.dcfr())
    t = time.perf_counter()
    vec.iterate(200)
    tv = time.perf_counter() - t
    t = time.perf_counter()
    tab.iterate(200)
    tt = time.perf_counter() - t
    vp, tp = vec.average_policy(), tab.average_policy()
    diff = 0.0
    hs = [ctx.hands(0), ctx.hands(1)]
    for I, info in enumerate(gt.infosets):
        node = int(info.key[1:info.key.index("h")])
        hand = int(info.key[info.key.index("h") + 1:])
        nd = tree.node(node)
        strat = vp.get(nd.decision_index).reshape(nd.num_children, hs[nd.player].size())
        for a in range(info.num_actions):
            diff = max(diff, abs(strat[a, hand] - tp.probs[info.offset + a]))
    # What changes at scale: the explicit tree enumerates hand pairs, the vector
    # form does not. Size of the explicit tree for FULL ranges on this board:
    full_ctx = aegis.SubgameContext(tree, aegis.Range.full().remove(c.board), aegis.Range.full().remove(c.board))
    n0, n1 = full_ctx.n(0), full_ctx.n(1)
    # Disjoint pairs: with 47 live cards each card lies in 46 combos, so a hand
    # shares a card with 2*46 - 1 = 91 combos (itself included).
    full_pairs = n0 * (n1 - (2 * 46 - 1))
    efg_full = full_pairs * tree.num_nodes
    vec = aegis.VectorCFR(full_ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    t = time.perf_counter()
    vec.iterate(200)
    t_full = time.perf_counter() - t
    return {"tabular_nodes": gt.num_nodes, "tabular_infosets": gt.num_infosets, "max_strategy_diff": diff,
            "full_range_explicit_histories": efg_full, "full_range_hand_pairs": full_pairs,
            "full_range_vector_seconds_200it": t_full,
            "full_range_vector_memory_mb": vec.memory_bytes / 1e6,
            "full_range_exploitability_pct_pot": full_ctx.exploitability(vec.average_policy())["exploitability_pct_pot"],
            "vector_nash_conv": ctx.exploitability(vp)["nash_conv"], "tabular_nash_conv": aegis.nash_conv(gt, tp),
            "vector_seconds": tv, "tabular_seconds": tt}


def run():
    out = {}
    # 1. Demo spot.
    req = dict(DEMO)
    req["solver"] = {"algorithm": "dcfr", "iterations": 2000}
    spot = solve_spot(req)
    hero = spot.hero_view(1, "AdJd", ["Bet 75"])
    root = spot.node_strategy(0)
    facing = spot.node_strategy(spot.node_by_path(["Bet 75"]))
    out["demo"] = {"convergence": spot.convergence, "exploitability": spot.exploitability,
                   "hero": {k: hero[k] for k in ("path", "actions", "strategy", "action_ev_from_here", "ev_from_here",
                                                 "opponent_range_blocked_fraction")},
                   "root_frequency": dict(zip(root["actions"], root["aggregate_frequency"])),
                   "facing_bet75_frequency": dict(zip(facing["actions"], facing["aggregate_frequency"])),
                   "hands": [spot.ctx.n(0), spot.ctx.n(1)], "nodes": spot.tree.num_nodes}
    # How many hands actually mix when facing the bet.
    strat = np.array(facing["strategy"])
    reach = np.array(facing["reach"])
    mixing = (strat.max(axis=1) < 0.95) & (reach > 0)
    out["demo"]["facing_bet75_mixing_hands"] = int(mixing.sum())
    out["demo"]["facing_bet75_mixing_examples"] = [
        {"hand": facing["combos"][i], "strategy": dict(zip(facing["actions"], strat[i].round(3).tolist()))}
        for i in np.where(mixing)[0][:12]]

    # 2. Convergence on spots of different range widths.
    spots = {"full ranges": ("random", "random"),
             "medium ranges": ("22+,A2+,K9+,QT+,JT,T9s,98s,87s,76s,65s", "22+,A2+,K7+,Q9+,J9+,T9,98s,87s,76s"),
             "narrow ranges": ("AA,KK,AK,AQs,88,44", "QQ-99,AQ,AJs,KQ")}
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2))
    colors = {"full ranges": "CFR", "medium ranges": "CFR+", "narrow ranges": "DCFR"}
    out["ranges"] = {}
    for label, (a, b) in spots.items():
        r = dict(DEMO)
        r.update({"range_p0": a, "range_p1": b, "solver": {"algorithm": "dcfr", "iterations": 1500}})
        r.pop("hero", None)
        sp = solve_spot(r)
        xs = [p["seconds"] for p in sp.convergence]
        ys = [p["exploitability_pct_pot"] for p in sp.convergence]
        out["ranges"][label] = {"convergence": sp.convergence, "hands": [sp.ctx.n(0), sp.ctx.n(1)]}
        from experiments.common import color
        ax.plot(xs, ys, color=color(colors[label]), marker="o", markersize=4, label=f"{label} ({sp.ctx.n(0)}x{sp.ctx.n(1)} hands)")
        print(label, sp.ctx.n(0), sp.ctx.n(1), f"{ys[-1]:.4f}% pot after {xs[-1]:.2f}s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("solver seconds")
    ax.set_ylabel("exploitability (% pot)")
    ax.set_title("River DCFR: convergence by range width")
    ax.legend()
    save_fig(fig, NAME, "convergence_by_range.png")

    # 3. Cross-validation against the generic tabular pipeline.
    out["cross_validation"] = cross_validate()
    print("cross-validation:", out["cross_validation"])
    print("demo:", out["demo"]["exploitability"], "hero:", out["demo"]["hero"])
    save_json(NAME, "results.json", out)
    return out


if __name__ == "__main__":
    run()
