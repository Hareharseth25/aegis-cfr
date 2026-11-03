"""Experiment 7 - Safe vs naive re-solving: full-game exploitability.

A subgame equilibrium need not be compatible with an equilibrium of the full
game. We measure exploitability of the composite strategy (blueprint trunk +
re-solved subgames) in the full game, where it can be computed exactly:

  A. Coin Toss (Brown & Sandholm 2017): the canonical unsafe example.
  B. Leduc: every round-2 subgame re-solved, blueprints of varying quality.
  C. No-Limit turn->river continual re-solving: every river subgame
     re-solved from (i) an unconverged blueprint and (ii) a coarse-abstraction
     blueprint translated into the full game.
  D. Nested re-solving after an off-tree opponent bet: translation vs naive
     re-solving vs safe re-solving (best in-abstraction alternative as the
     constraint) vs safe + nested river re-solving, compared with solving the
     reference game directly.

Exploitability of player p's strategy = (opponent best-response value) - (the
opponent's game value); for poker games we report the opponent's BR value
directly (lower is better; the game value is a constant across methods).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import color, plt, save_fig, save_json  # noqa: E402

import aegis  # noqa: E402
from evaluation import sequence_form  # noqa: E402

NAME = "exp07_safe_resolving"
METHODS = {"naive": aegis.ResolveMethod.NAIVE, "resolve": aegis.ResolveMethod.RESOLVE,
           "maxmargin": aegis.ResolveMethod.MAXMARGIN}


# ----------------------------------------------------------------- A. coin toss
def coin_toss():
    t = aegis.GameTree.build(aegis.CoinTossGame())
    out = {}
    for label, iters in (("converged blueprint", 20000), ("unconverged blueprint", 15)):
        cfr = aegis.TabularCFR(t, aegis.CFRConfig.cfr_plus())
        cfr.iterate(iters)
        bp = cfr.average_policy()
        roots = aegis.nodes_with_public_state(t, aegis.public_state_id(t, "pub:P"))
        I = t.infoset_index(1, "*:P")
        row = {"blueprint": {"p1_exploitability": aegis.best_response(t, bp, 0).value, "guess_heads": bp.at(t, I)[0]}}
        for m, mm in METHODS.items():
            o = aegis.TabularResolveOptions()
            o.us, o.iterations, o.method = 1, 5000, mm
            r = aegis.resolve_subgame(t, bp, roots, o)
            row[m] = {"p1_exploitability": aegis.best_response(t, r.policy, 0).value,
                      "guess_heads": r.policy.at(t, I)[0]}
        out[label] = row
        print("coin toss", label, {k: round(v["p1_exploitability"], 4) for k, v in row.items()})
    return out


# ----------------------------------------------------------------------- B. Leduc
def leduc():
    t = aegis.GameTree.build(aegis.LeducGame())
    v0 = sequence_form.solve(t)["value_p0"]
    subgames = [ps for ps, key in enumerate(t.public_keys) if "B-" not in key and key.endswith("/")]
    out = {"game_value_p0": v0, "num_subgames": len(subgames), "blueprints": {}}
    for iters in (5, 20, 100, 1000):
        cfr = aegis.TabularCFR(t, aegis.CFRConfig.cfr_plus())
        cfr.iterate(iters)
        bp = cfr.average_policy()
        expl = lambda pol, us: (aegis.best_response(t, pol, 1 - us).value - (-v0 if us == 0 else v0))  # noqa: E731
        row = {"blueprint": {"p0": expl(bp, 0), "p1": expl(bp, 1), "nash_conv": aegis.nash_conv(t, bp)}}
        for m, mm in METHODS.items():
            composite = {0: bp, 1: bp}
            for us in (0, 1):
                pol = bp
                for ps in subgames:
                    o = aegis.TabularResolveOptions()
                    o.us, o.iterations, o.method = us, 400, mm
                    pol = aegis.resolve_subgame(t, pol, aegis.nodes_with_public_state(t, ps), o).policy
                composite[us] = pol
            both = aegis.TabularPolicy(t)
            probs = composite[0].probs.copy()
            p1 = composite[1].probs
            for I, info in enumerate(t.infosets):
                if info.player == 1:
                    probs[info.offset:info.offset + info.num_actions] = p1[info.offset:info.offset + info.num_actions]
            both.probs = probs
            row[m] = {"p0": expl(composite[0], 0), "p1": expl(composite[1], 1), "nash_conv": aegis.nash_conv(t, both)}
        out["blueprints"][iters] = row
        print("leduc blueprint", iters, {k: round(v["nash_conv"], 4) for k, v in row.items()})
    return out


# ---------------------------------------------------------- poker helpers
def turn_config(board, pot, stack, turn_bets, river_bets, p1_turn_bets=None):
    c = aegis.TreeConfig()
    c.board = aegis.parse_cards(board)
    c.pot, c.effective_stack = pot, stack
    ts, rs = aegis.StreetSizes(), aegis.StreetSizes()
    ts.bet_fractions, ts.raise_fractions, ts.max_raises = list(turn_bets), [], 1
    rs.bet_fractions, rs.raise_fractions, rs.max_raises = list(river_bets), [], 1
    c.set_sizes(ts, rs)
    if p1_turn_bets is not None:
        s = c.get_street_sizes(0, 1)
        s.bet_fractions = list(p1_turn_bets)
        c.set_street_sizes(0, 1, s)
    return c


R0 = "AA,KK,99,66,AK,AQ,AJ,KQ,KJs,QJs,JTs,T9s,98s,87s,76s,A5s,A4s,K9s,Q9s"
R1 = "AQ,AJ,AT,KQ,KJ,KT,QJ,QQ-77,55,44,A9s,K9s,T9s,98s,J9s,T8s,65s"


def resolve_all_rivers(ctx, tree, base, method, us, iters=150):
    pol = base
    t0 = time.perf_counter()
    for root in aegis.street_roots(tree, 1):
        o = aegis.VectorResolveOptions()
        o.method, o.us, o.iterations = method, us, iters
        pol = aegis.resolve_subtree(ctx, pol, root, o).policy
    return pol, time.perf_counter() - t0


def continual_turn_river():
    board = "Kd9s6h3c"
    cfg = turn_config(board, 100, 150, [0.5, 1.0], [0.5, 1.0])
    tree = aegis.PublicTree.build(cfg)
    r0 = aegis.Range.parse(R0).remove(cfg.board)
    r1 = aegis.Range.parse(R1).remove(cfg.board)
    ctx = aegis.SubgameContext(tree, r0, r1, 1)
    out = {"nodes": tree.num_nodes, "river_subgames": len(aegis.street_roots(tree, 1)), "hands": [ctx.n(0), ctx.n(1)]}
    # Reference: near-exact solve of the full game.
    full = aegis.VectorCFR(ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    full.iterate(400)
    out["full_solve"] = ctx.exploitability(full.average_policy())
    blueprints = {}
    un = aegis.VectorCFR(ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    un.iterate(12)
    blueprints["unconverged (12 it)"] = un.average_policy()
    # Coarse abstraction: one bet size per street, translated into the full game.
    ccfg = turn_config(board, 100, 150, [0.75], [0.75])
    ctree = aegis.PublicTree.build(ccfg)
    cctx = aegis.SubgameContext(ctree, r0, r1, 1)
    cs = aegis.VectorCFR(cctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    cs.iterate(400)
    blueprints["coarse abstraction + translation"] = aegis.translate_policy(ctx, cctx, cs.average_policy())
    out["blueprints"] = {}
    for bname, bp in blueprints.items():
        row = {"blueprint": {"br_vs_p0": ctx.best_response(bp, 1)[0], "br_vs_p1": ctx.best_response(bp, 0)[0],
                             **{k: v for k, v in ctx.exploitability(bp).items() if k in ("nash_conv", "exploitability_pct_pot")}}}
        for m, mm in METHODS.items():
            p0pol, t0 = resolve_all_rivers(ctx, tree, bp, mm, 0)
            p1pol, t1 = resolve_all_rivers(ctx, tree, bp, mm, 1)
            both = p0pol.copy()
            for n in range(tree.num_nodes):
                nd = tree.node(n)
                if nd.type == aegis.PNodeType.DECISION and nd.player == 1:
                    both.set(nd.decision_index, p1pol.get(nd.decision_index))
            e = ctx.exploitability(both)
            row[m] = {"br_vs_p0": ctx.best_response(p0pol, 1)[0], "br_vs_p1": ctx.best_response(p1pol, 0)[0],
                      "nash_conv": e["nash_conv"], "exploitability_pct_pot": e["exploitability_pct_pot"],
                      "seconds": t0 + t1}
        out["blueprints"][bname] = row
        print("turn->river", bname, {k: round(v["exploitability_pct_pot"], 3) for k, v in row.items()})
    return out


def nested_offtree():
    """Opponent (P1) uses turn bet sizes the abstraction never considered."""
    board = "Jh8c5d2s"
    abs_cfg = turn_config(board, 100, 150, [0.75], [0.75])
    ref_cfg = turn_config(board, 100, 150, [0.75], [0.75], p1_turn_bets=[0.3, 0.75, 1.5])
    atree, rtree = aegis.PublicTree.build(abs_cfg), aegis.PublicTree.build(ref_cfg)
    r0 = aegis.Range.parse(R0).remove(abs_cfg.board)
    r1 = aegis.Range.parse(R1).remove(abs_cfg.board)
    actx, rctx = aegis.SubgameContext(atree, r0, r1, 1), aegis.SubgameContext(rtree, r0, r1, 1)
    bs = aegis.VectorCFR(actx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    bs.iterate(400)
    translated = aegis.translate_policy(rctx, actx, bs.average_policy())
    d = rtree.child_by_label(0, "Check")  # P1 to act after P0 checks
    kids = list(range(rtree.node(d).first_child, rtree.node(d).first_child + rtree.node(d).num_children))
    labels = [rtree.node(k).action.to_string() for k in kids]
    sizes = aegis.StreetSizes()
    sizes.bet_fractions, sizes.allow_all_in = [0.75], True
    on = [k for k in kids if aegis.action_in_abstraction(rtree.node(k).action, sizes, False)]
    off = [k for k in kids if k not in on]
    out = {"p1_bets_reference": labels, "offtree": [rtree.node(k).action.to_string() for k in off]}
    results = {"translation": translated}
    # Naive: opponent range = range with which P1 makes the translated (in-abstraction) bet.
    naive = translated.copy()
    for k in off:
        a = rtree.node(k).action
        # Neighbouring in-abstraction sizes and pseudo-harmonic weights.
        sized = sorted([(rtree.node(c).action.fraction, c) for c in on if rtree.node(c).action.kind != aegis.ActionKind.CHECK])
        lo = max([s for s in sized if s[0] <= a.fraction], default=sized[0])
        hi = min([s for s in sized if s[0] >= a.fraction], default=sized[-1])
        f = 1.0 if lo == hi else aegis.pseudo_harmonic_low_prob(lo[0], hi[0], a.fraction)
        r_lo = aegis.reach_at(rctx, translated, lo[1])[1]
        r_hi = aegis.reach_at(rctx, translated, hi[1])[1]
        opp_range = f * r_lo + (1 - f) * r_hi
        ours = aegis.reach_at(rctx, translated, k)[0]
        c = aegis.VectorCFRConfig(aegis.CFRConfig.dcfr())
        c.root = k
        c.set_root_reach(ours.tolist(), opp_range.tolist())
        s = aegis.VectorCFR(rctx, c)
        s.iterate(300)
        naive = aegis.splice_policy(rtree, naive, s.average_policy(), k, 0)
    results["naive"] = naive
    # Safe nested: constraint = best in-abstraction alternative at d.
    w = aegis.best_alternative_constraint(rctx, translated, d, on, 1)
    safe = translated.copy()
    margins = {}
    for k in off:
        o = aegis.VectorResolveOptions()
        o.method, o.us, o.iterations, o.constraint = aegis.ResolveMethod.RESOLVE, 0, 300, w
        r = aegis.resolve_subtree(rctx, safe, k, o)
        safe = r.policy
        margins[rtree.node(k).action.to_string()] = {"worst_violation": r.worst_violation,
                                                      "total_violation": r.total_violation}
    results["resolve"] = safe
    # Safe + nested river re-solving below the off-tree bets.
    nested = safe
    river_roots = aegis.street_roots(rtree, 1)

    def under(node, anc):
        while node > 0:
            if node == anc:
                return True
            node = rtree.node(node).parent
        return False
    for k in off:
        for root in river_roots:
            if under(root, k):
                o = aegis.VectorResolveOptions()
                o.method, o.us, o.iterations = aegis.ResolveMethod.RESOLVE, 0, 150
                nested = aegis.resolve_subtree(rctx, nested, root, o).policy
    results["nested"] = nested
    # Gold standard: solve the reference game directly.
    gold = aegis.VectorCFR(rctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    gold.iterate(400)
    gpol = gold.average_policy()
    out["br_vs_p0"] = {name: rctx.best_response(pol, 1)[0] for name, pol in results.items()}
    out["br_vs_p0"]["reference-solve"] = rctx.best_response(gpol, 1)[0]
    out["game_value_p1_estimate"] = rctx.evaluate(gpol)[0][1]
    out["constraint_margins"] = margins
    print("nested off-tree: P1 BR value vs P0 strategy:", {k: round(v, 3) for k, v in out["br_vs_p0"].items()})
    return out


def plot(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    # Coin toss
    ax = axes[0]
    ct = results["coin_toss"]["converged blueprint"]
    names = ["blueprint", "naive", "resolve", "maxmargin"]
    ax.bar(names, [ct[n]["p1_exploitability"] for n in names], color=[color(n) for n in names], width=0.6)
    ax.set_title("Coin Toss: P2 exploitability")
    ax.set_ylabel("chips")
    # Leduc
    ax = axes[1]
    bl = results["leduc"]["blueprints"]
    its = sorted(bl, key=int)
    for n in names:
        ax.plot([int(i) for i in its], [bl[i][n]["nash_conv"] for i in its], color=color(n), marker="o", label=n)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("blueprint CFR+ iterations")
    ax.set_ylabel("full-game NashConv")
    ax.set_title("Leduc: re-solving all round-2 subgames")
    ax.legend()
    # Nested off-tree
    ax = axes[2]
    nv = results["nested_offtree"]["br_vs_p0"]
    order = ["translation", "naive", "resolve", "nested", "reference-solve"]
    base = min(nv.values())
    ax.bar(order, [nv[k] - base for k in order], color=[color(k) for k in order], width=0.6)
    ax.set_title("Off-tree turn bets: opponent BR gain")
    ax.set_ylabel("BR value - best method (chips)")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, NAME, "safe_vs_naive.png")


def run():
    res = {"coin_toss": coin_toss(), "leduc": leduc(), "turn_river": continual_turn_river(),
           "nested_offtree": nested_offtree()}
    save_json(NAME, "results.json", res)
    plot(res)
    return res


if __name__ == "__main__":
    run()
