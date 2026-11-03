"""Experiment 5 - Action abstraction: exploitability vs tree size.

How much do you lose by using fewer bet sizes? A fine reference river
game (9 bet sizes + all-in) is the ground truth. Each abstraction (a subset of
the sizes) is solved, translated into the reference game (pseudo-harmonic or
nearest mapping for the opponent's off-tree sizes) and its exact
exploitability in the reference game is measured. Reported against the number
of abstract information sets (Σ decision nodes x hands).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import color, plt, save_fig, save_json  # noqa: E402

import aegis  # noqa: E402

NAME = "exp05_action_abstraction"
GRID = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0]
ABSTRACTIONS = [[0.75], [0.33, 1.0], [0.33, 0.75, 1.5], [0.25, 0.5, 1.0, 2.0], [0.25, 0.33, 0.5, 0.75, 1.0, 1.5],
                [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0], GRID]
SPOTS = [
    ("AsKh8c4d2s", "AA,KK,88,44,22,AK,AQ,AJ,A9s-A2s,KQ,KJs,QJs,JTs,T9s,98s,87s,76s,65s",
     "AQ,AJ,AT,A9-A2,KQ,KJ,KT,QQ-99,77-55,33,QJs,JTs,T9s"),
    ("Jh9h6c3d2s", "JJ,99,66,AJ,KJ,QJs,JTs,T8s,87s,75s,Ah5h,Kh5h,AhQh,AA,KK", "AJ,KJ,QJ,JT,TT,88,77,QhTh,Ah4h,98s,T9s"),
    ("QsQd7c4h2d", "QJ,AQ,KQ,77,44,AA,KK,A5s,65s,J9s,T9s,AJs", "AA-88,AK,AJ,KJ,JT,T9s,98s,87s,76s,A7s"),
]


def spot_base(board, pot=100, stack=200):
    c = aegis.TreeConfig()
    c.board = aegis.parse_cards(board)
    c.pot, c.effective_stack = pot, stack
    s = aegis.StreetSizes()
    s.raise_fractions = [1.0]
    s.allow_all_in = True
    s.max_raises = 2
    c.set_sizes(s, s)
    return c


def run(iterations=600):
    out = {"grid": GRID, "spots": []}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    agg = {"pseudo-harmonic": [], "nearest": [], "abstract": []}
    for board, a, b in SPOTS:
        res = {"board": board, "abstractions": []}
        for method in (aegis.TranslationMethod.PSEUDO_HARMONIC, aegis.TranslationMethod.NEAREST):
            spot = aegis.RefinementSpot()
            spot.base = spot_base(board)
            spot.r0 = aegis.Range.parse(a).remove(spot.base.board)
            spot.r1 = aegis.Range.parse(b).remove(spot.base.board)
            spot.reference_bets = GRID
            spot.iterations = iterations
            spot.translation = method
            ref = aegis.ActionRefiner(spot)
            for abst in ABSTRACTIONS:
                ev = ref.evaluate([abst, abst], False)
                res["abstractions"].append({"method": str(method).split(".")[-1], "sizes": abst,
                                            "infosets": ev.abs_infosets, "nodes": ev.abs_nodes,
                                            "ref_exploitability_pct": ev.ref_exploitability_pct,
                                            "abs_exploitability_pct": ev.abs_exploitability_pct,
                                            "solve_seconds": ev.solve_seconds})
        out["spots"].append(res)
        print(board, [(r["method"][:6], len(r["sizes"]), round(r["ref_exploitability_pct"], 3)) for r in res["abstractions"]])
    # Aggregate: mean over spots per abstraction.
    for key, label in (("PSEUDO_HARMONIC", "pseudo-harmonic"), ("NEAREST", "nearest")):
        xs, ys, ya = [], [], []
        for k, abst in enumerate(ABSTRACTIONS):
            rows = [[r for r in s["abstractions"] if r["method"] == key and r["sizes"] == abst][0] for s in out["spots"]]
            xs.append(np.mean([r["infosets"] for r in rows]))
            ys.append(np.mean([r["ref_exploitability_pct"] for r in rows]))
            ya.append(np.mean([r["abs_exploitability_pct"] for r in rows]))
        agg[label] = list(zip(xs, ys))
        ax.plot(xs, ys, color=color(label), marker="o", label=f"reference-game exploitability ({label} translation)")
        if key == "PSEUDO_HARMONIC":
            agg["abstract"] = list(zip(xs, ya))
            ax.plot(xs, ya, color=color("abstract"), marker="s", linestyle="--",
                    label="exploitability inside the abstraction")
    for (x, y), abst in zip(agg["pseudo-harmonic"], ABSTRACTIONS):
        ax.annotate(f"{len(abst)} sizes", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("abstract information sets (decision nodes x hands)")
    ax.set_ylabel("exploitability (% pot), mean of 3 spots")
    ax.set_title("Strategic cost of bet-size compression")
    ax.legend(fontsize=8)
    save_fig(fig, NAME, "exploitability_vs_tree_size.png")
    out["aggregate"] = agg
    save_json(NAME, "results.json", out)
    return out


if __name__ == "__main__":
    run()
