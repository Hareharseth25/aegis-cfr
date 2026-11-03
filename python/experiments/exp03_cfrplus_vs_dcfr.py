"""Experiment 3 - CFR+ vs DCFR (and LCFR, CFR) on identical games and budgets.

The point is to see where each one wins, not to crown one. Games span:
  * small/medium tabular games (Kuhn, Leduc, Leduc with 6 ranks);
  * No-Limit river subgames (vector form) with deep stacks/all-in options,
    where a few very costly early mistakes (huge negative regrets) occur -
    the regime DCFR's discounting targets;
  * a river subgame with polarised ranges and a narrow (single-size) tree.
For each game every algorithm gets the same iteration budget; we report the
exploitability at every checkpoint and the winner at each budget.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import line_plot, log_checkpoints, plt, save_fig, save_json  # noqa: E402

import aegis  # noqa: E402

NAME = "exp03_cfrplus_vs_dcfr"
ALGOS = {"CFR(alt)": lambda: aegis.CFRConfig.vanilla(True), "CFR+": aegis.CFRConfig.cfr_plus,
         "DCFR": aegis.CFRConfig.dcfr, "LCFR": aegis.CFRConfig.linear}


def tabular_curve(tree, cfg, total):
    cfr = aegis.TabularCFR(tree, cfg)
    xs, ys, done = [], [], 0
    for cp in log_checkpoints(total, factor=1.6):
        cfr.iterate(cp - done)
        done = cp
        xs.append(cp)
        ys.append(aegis.exploitability(tree, cfr.average_policy()))
    return xs, ys


def river_spot(board, r0, r1, pot, stack, bets, raises=(1.0,), all_in=True, max_raises=3):
    c = aegis.TreeConfig()
    c.board = aegis.parse_cards(board)
    c.pot = pot
    c.effective_stack = stack
    s = aegis.StreetSizes()
    s.bet_fractions = list(bets)
    s.raise_fractions = list(raises)
    s.allow_all_in = all_in
    s.max_raises = max_raises
    c.set_sizes(s, s)
    tree = aegis.PublicTree.build(c)
    r0 = aegis.Range.parse(r0).remove(c.board)
    r1 = aegis.Range.parse(r1).remove(c.board)
    return tree, aegis.SubgameContext(tree, r0, r1)


def vector_curve(ctx, cfg, total):
    s = aegis.VectorCFR(ctx, aegis.VectorCFRConfig(cfg))
    xs, ys, done = [], [], 0
    for cp in log_checkpoints(total, factor=1.6):
        s.iterate(cp - done)
        done = cp
        xs.append(cp)
        ys.append(ctx.exploitability(s.average_policy())["exploitability_pct_pot"])
    return xs, ys


def run():
    games = {}
    games["kuhn"] = ("tab", aegis.GameTree.build(aegis.KuhnGame()), 20000, "exploitability (chips)")
    games["leduc"] = ("tab", aegis.GameTree.build(aegis.LeducGame()), 3000, "exploitability (chips)")
    big = aegis.LeducConfig()
    big.num_ranks = 6
    games["leduc-6ranks"] = ("tab", aegis.GameTree.build(aegis.LeducGame(big)), 1500, "exploitability (chips)")
    spots = {
        "river-deep-allin": dict(board="AsKh8c4d2s", r0="random", r1="random", pot=20, stack=200,
                                 bets=(0.33, 0.75, 1.5)),
        "river-demo": dict(board="AsKh8c4d2s",
                           r0="AA,KK,88,44,22,AK,AQ,AJ,AT,A9s-A2s,KQ,KJs,QJs,JTs,T9s,98s,87s,76s,65s,54s",
                           r1="AQ,AJ,AT,A9-A2,KQ,KJ,KT,K9s,QQ-99,77-55,33,QJs,JTs,T9s,98s,87s",
                           pot=100, stack=150, bets=(0.33, 0.75, 1.5)),
        "river-polarised-1size": dict(board="Qs9h7d4c2c", r0="AA,QQ,99,77,AQs,KJs,T8s,65s,A5s,K5s",
                                      r1="AQ,KQ,QJ,JJ-TT,A9s,K9s,98s", pot=100, stack=100, bets=(1.0,),
                                      raises=(), max_raises=1),
    }
    for name, kw in spots.items():
        tree, ctx = river_spot(**kw)
        games[name] = ("vec", ctx, 2000, "exploitability (% pot)")
        games[name + "_tree"] = tree  # keep alive
    results = {}
    for gname, g in games.items():
        if gname.endswith("_tree"):
            continue
        kind, obj, total, ylabel = g
        res = {}
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for aname, mk in ALGOS.items():
            xs, ys = tabular_curve(obj, mk(), total) if kind == "tab" else vector_curve(obj, mk(), total)
            res[aname] = {"iterations": xs, "exploitability": ys}
            line_plot(ax, xs, ys, aname)
        winners = {}
        for i, it in enumerate(res["CFR+"]["iterations"]):
            best = min(ALGOS, key=lambda a: res[a]["exploitability"][i])
            winners[it] = best
        res["winner_by_budget"] = winners
        res["dcfr_over_cfrplus_final"] = res["DCFR"]["exploitability"][-1] / max(res["CFR+"]["exploitability"][-1], 1e-300)
        results[gname] = res
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iterations")
        ax.set_ylabel(ylabel)
        ax.set_title(gname)
        ax.legend()
        save_fig(fig, NAME, f"{gname}.png")
        print(f"{gname}: final " + ", ".join(f"{a}={res[a]['exploitability'][-1]:.2e}" for a in ALGOS)
              + f" | DCFR/CFR+ = {res['dcfr_over_cfrplus_final']:.2f}")
    save_json(NAME, "results.json", results)
    return results


if __name__ == "__main__":
    run()
