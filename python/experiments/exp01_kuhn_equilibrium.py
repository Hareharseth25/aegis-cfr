"""Experiment 1 - Exact equilibria (Kuhn & Leduc).

NashConv vs iterations. This is the basic check that CFR is right:
  * every CFR variant drives NashConv -> 0 on Kuhn;
  * the average strategies match the closed-form Kuhn equilibrium family
    (player 1 unique; player 0 parameterised by α ∈ [0, 1/3]);
  * the game values agree with an exact sequence-form LP solution
    (-1/18 for Kuhn; Leduc's value), whose own NashConv is ~0.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import line_plot, log_checkpoints, plt, save_fig, save_json  # noqa: E402

import aegis  # noqa: E402
from evaluation import sequence_form  # noqa: E402

NAME = "exp01_kuhn_equilibrium"
VARIANTS = {
    "CFR(alt)": lambda: aegis.CFRConfig.vanilla(True),
    "CFR(sim)": lambda: aegis.CFRConfig.vanilla(False),
    "CFR+": aegis.CFRConfig.cfr_plus,
    "DCFR": aegis.CFRConfig.dcfr,
    "LCFR": aegis.CFRConfig.linear,
}


def kuhn_check(tree, pol) -> dict:
    p = lambda player, key, a: pol.at(tree, tree.infoset_index(player, key))[a]  # noqa: E731
    alpha = p(0, "0", 1)
    return {
        "alpha (J bet)": alpha,
        "K bet / (3 alpha)": p(0, "2", 1) / (3 * alpha) if alpha > 1e-9 else None,
        "Q call - (alpha + 1/3)": p(0, "1pb", 1) - (alpha + 1 / 3),
        "P1 Q call vs bet (1/3)": p(1, "1b", 1),
        "P1 J bet vs check (1/3)": p(1, "0p", 1),
        "P1 K call (1)": p(1, "2b", 1),
        "P1 J call (0)": p(1, "0b", 1),
    }


def run(total: int = 100_000):
    out = {"kuhn": {}, "leduc_lp": {}}
    kuhn = aegis.GameTree.build(aegis.KuhnGame())
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, mk in VARIANTS.items():
        cfr = aegis.TabularCFR(kuhn, mk())
        xs, ys, done = [], [], 0
        for cp in log_checkpoints(total):
            cfr.iterate(cp - done)
            done = cp
            xs.append(cp)
            ys.append(aegis.nash_conv(kuhn, cfr.average_policy()))
        avg = cfr.average_policy()
        out["kuhn"][name] = {"iterations": xs, "nash_conv": ys, "value_p0": aegis.expected_values(kuhn, avg)[0],
                             "closed_form_check": kuhn_check(kuhn, avg)}
        line_plot(ax, xs, ys, name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iterations")
    ax.set_ylabel("NashConv (chips)")
    ax.set_title("Kuhn poker: NashConv vs iterations")
    ax.legend()
    save_fig(fig, NAME, "kuhn_nashconv.png")

    for gname, game in (("kuhn", aegis.KuhnGame()), ("leduc", aegis.LeducGame())):
        tree = aegis.GameTree.build(game)
        lp = sequence_form.solve(tree)
        rep = aegis.exploitability_report(tree, lp["policy"])
        cfr = aegis.TabularCFR(tree, aegis.CFRConfig.cfr_plus())
        cfr.iterate(5000 if gname == "leduc" else 20000)
        out["leduc_lp" if gname == "leduc" else "kuhn_lp"] = {
            "lp_value_p0": lp["value_p0"], "lp_value_p1_side": -lp["value_p1"],
            "lp_policy_nash_conv": rep.nash_conv, "sequences": lp["num_sequences"],
            "cfr_plus_value_p0": aegis.expected_values(tree, cfr.average_policy())[0],
            "cfr_plus_nash_conv": aegis.nash_conv(tree, cfr.average_policy()),
        }
    out["kuhn_closed_form_value"] = -1 / 18
    save_json(NAME, "results.json", out)
    k = out["kuhn"]
    print("Kuhn final NashConv:", {n: f"{v['nash_conv'][-1]:.2e}" for n, v in k.items()})
    print("Kuhn LP value", out["kuhn_lp"]["lp_value_p0"], "closed form", -1 / 18)
    print("Leduc LP value", out["leduc_lp"]["lp_value_p0"], "LP NashConv", out["leduc_lp"]["lp_policy_nash_conv"],
          "CFR+ value", out["leduc_lp"]["cfr_plus_value_p0"])
    return out


if __name__ == "__main__":
    run()
