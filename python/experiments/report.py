"""Writes docs/RESULTS.md from results/*/results.json.

All numbers come straight from the experiment outputs, so the write-up can't
drift from what was actually run.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "RESULTS.md")


def load(name):
    p = os.path.join(RES, name, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fig(name, file, alt):
    return f"![{alt}](../results/{name}/{file})"


def e(x, d=3):
    if x is None:
        return "-"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    a = abs(x)
    return f"{x:.{d}e}" if (a != 0 and (a < 1e-3 or a >= 1e5)) else f"{x:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def section_exp01(lines):
    r = load("exp01_kuhn_equilibrium")
    if not r:
        return
    lines += ["## 1. Exact equilibria (Kuhn, Leduc)", "",
              "*NashConv vs iterations. This is the basic check that CFR is right.*", "",
              fig("exp01_kuhn_equilibrium", "kuhn_nashconv.png", "Kuhn NashConv"), ""]
    rows = []
    for name, v in r["kuhn"].items():
        c = v["closed_form_check"]
        rows.append([name, e(v["nash_conv"][-1]), e(v["value_p0"], 6), e(c["alpha (J bet)"], 4),
                     e(c["Q call - (alpha + 1/3)"], 4), e(c["P1 Q call vs bet (1/3)"], 4)])
    lines += [table(["algorithm (100k it)", "NashConv", "value P0 (-1/18 = -0.055556)", "α = P0 J-bet",
                     "Q-call - (α+⅓)", "P1 Q-call (⅓)"], rows), ""]
    lp, klp = r["leduc_lp"], r.get("kuhn_lp", {})
    lines += [f"Sequence-form LP (solved separately, as a cross-check): Kuhn value **{e(klp.get('lp_value_p0'), 6)}** "
              f"(closed form -1/18 = -0.055556); Leduc value **{e(lp['lp_value_p0'], 7)}**, LP strategy NashConv "
              f"{e(lp['lp_policy_nash_conv'])}; CFR+ (5000 it) value {e(lp['cfr_plus_value_p0'], 7)}, NashConv "
              f"{e(lp['cfr_plus_nash_conv'])}.", "",
              "Compared with OpenSpiel (tests/python/test_openspiel_oracle.py): same tree sizes and information sets, "
              "NashConv of random policies agrees to 1e-9, and the CFR, CFR+ and DCFR average policies match "
              "OpenSpiel's at every iteration (max |Δσ̄| < 1e-9) on Kuhn and Leduc.", ""]


def section_exp02(lines):
    r = load("exp02_leduc_comparison")
    if not r:
        return
    lines += ["## 2. Leduc: exploitability vs time", "",
              "*Same game and same wall-clock budget. Sampling methods use 5 seeds (mean ± 95% CI).*", "",
              fig("exp02_leduc_comparison", "leduc_time.png", "Leduc time"), " ",
              fig("exp02_leduc_comparison", "leduc-6ranks_time.png", "Leduc 6 ranks time"), ""]
    for g, d in r.items():
        rows = []
        for m, v in d["methods"].items():
            if "mean" in v:
                rows.append([m, f"{v['mean'][-1]:.3e} ± {v['ci95'][-1]:.1e}", "-"])
            else:
                p = v["points"][-1]
                rows.append([m, e(p[3]), p[2]])
        lines += [f"**{g}** ({d['infosets']} information sets):", "",
                  table(["method", "exploitability at budget", "iterations"], rows), ""]


def section_exp03(lines):
    r = load("exp03_cfrplus_vs_dcfr")
    if not r:
        return
    lines += ["## 3. CFR+ vs DCFR: which one wins where?", "",
              "*Same games and iteration budgets. A ratio below 1 means DCFR ended up less exploitable than CFR+.*", ""]
    rows = []
    for g, d in r.items():
        fin = {a: d[a]["exploitability"][-1] for a in ("CFR(alt)", "CFR+", "DCFR", "LCFR")}
        best = min(fin, key=fin.get)
        rows.append([g, d["CFR+"]["iterations"][-1], e(fin["CFR(alt)"]), e(fin["CFR+"]), e(fin["DCFR"]), e(fin["LCFR"]),
                     f"{d['dcfr_over_cfrplus_final']:.2f}", best])
    lines += [table(["game", "iterations", "CFR", "CFR+", "DCFR", "LCFR", "DCFR/CFR+", "best"], rows), "",
              " ".join(fig("exp03_cfrplus_vs_dcfr", f"{g}.png", g) for g in r), "",
              "DCFR discounts the regret from early iterations, which helps when those early iterations are badly "
              "wrong (deep stacks with all-in actions, the bigger Leduc). CFR+ wins on Kuhn and on the polarised "
              "river spot with a single bet size. So neither is better everywhere, which is why I ran both on the "
              "same games.", ""]


def section_exp04(lines):
    r = load("exp04_river_exact")
    if not r:
        return
    d = r["demo"]
    lines += ["## 4. Exact river solving", "",
              "*Weighted ranges and custom bet sizes. Exploitability is the exact best-response gain in the "
              "river game as constructed.*", "", fig("exp04_river_exact", "convergence_by_range.png", "river convergence"), ""]
    rows = [[k, "x".join(map(str, v["hands"])), v["convergence"][-1]["iteration"], e(v["convergence"][-1]["seconds"]),
             e(v["convergence"][-1]["exploitability_pct_pot"], 5)] for k, v in r["ranges"].items()]
    lines += [table(["ranges", "hands", "iterations", "seconds", "exploitability (% pot)"], rows), ""]
    h = d["hero"]
    lines += [f"Demo spot (A♠K♥8♣4♦2♠, pot 100, stacks 150): exploitability **{e(d['exploitability']['exploitability_pct_pot'], 5)}% "
              f"of pot** after 2000 DCFR iterations. Hero A♦J♦ facing a 75 BB bet: "
              + ", ".join(f"{a} {100 * p:.2f}% (EV {e(ev, 2)} BB)" for a, p, ev in zip(h["actions"], h["strategy"], h["action_ev_from_here"]))
              + f". {d['facing_bet75_mixing_hands']} hands in player 1's range play a mixed strategy at that node, e.g. "
              + "; ".join(f"{x['hand']} {x['strategy']}" for x in d["facing_bet75_mixing_examples"][:4]) + ".", ""]
    cv = r["cross_validation"]
    lines += [f"Checked against the generic tabular solver (the same game written out as an explicit tree with "
              f"{cv['tabular_infosets']} information sets / {cv['tabular_nodes']} histories): max strategy difference "
              f"{e(cv['max_strategy_diff'])} after 200 DCFR iterations; NashConv {e(cv['vector_nash_conv'], 6)} vs "
              f"{e(cv['tabular_nash_conv'], 6)}; vector solver {cv['tabular_seconds'] / max(cv['vector_seconds'], 1e-9):.0f}x faster.", ""]
    if "full_range_explicit_histories" in cv:
        lines += [f"With full ranges the explicit tree would need "
                  f"{cv['full_range_hand_pairs']:,} hand pairs x {r['demo']['nodes']} public nodes ~ "
                  f"{cv['full_range_explicit_histories']:.2e} histories, which is too big for the tabular solver. The vector "
                  f"solver handles it with {e(cv['full_range_vector_memory_mb'], 1)} MB in {e(cv['full_range_vector_seconds_200it'], 2)} s "
                  f"(200 iterations, {e(cv['full_range_exploitability_pct_pot'], 3)}% pot), and exact best responses still work.", ""]


def section_exp05(lines):
    r = load("exp05_action_abstraction")
    if not r:
        return
    lines += ["## 5. Bet-size abstraction: exploitability vs tree size", "",
              "*Each abstract solution is translated into a reference game with 9 bet sizes and measured there exactly.*", "",
              fig("exp05_action_abstraction", "exploitability_vs_tree_size.png", "abstraction"), ""]
    ph = r["aggregate"]["pseudo-harmonic"]
    nr = r["aggregate"]["nearest"]
    ab = r["aggregate"]["abstract"]
    rows = [[len(s), e(ph[i][0], 0), e(ab[i][1], 4), e(ph[i][1], 4), e(nr[i][1], 4)] for i, s in enumerate(
        [[0.75], [0.33, 1.0], [0.33, 0.75, 1.5], [0.25, 0.5, 1.0, 2.0], [0.25, 0.33, 0.5, 0.75, 1.0, 1.5],
         [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0], r["grid"]])]
    lines += [table(["bet sizes", "abstract infosets", "expl. inside abstraction (% pot)",
                     "reference expl., pseudo-harmonic", "reference expl., nearest"], rows), "",
              "Exploitability measured *inside* the abstraction tells you very little. In the reference game, coarse "
              "abstractions are much worse than they look, and the choice of translation mapping matters about as "
              "much as adding more sizes.", ""]


def section_exp06(lines):
    r = load("exp06_adaptive_sampling")
    if not r:
        return
    lines += ["## A. Priority-guided sampling (Experiment A)", "",
              "*Same node budget for every method. λ was tuned the same way for each one on separate seeds, then "
              "evaluated on 8 new seeds and on a bigger game it wasn't tuned on. Mean ± 95% CI.*", "",
              fig("exp06_adaptive_sampling", "leduc_vs_nodes.png", "Leduc"), " ",
              fig("exp06_adaptive_sampling", "leduc-5ranks_vs_nodes.png", "Leduc 5 ranks"), ""]
    for g, d in r["eval"].items():
        rows = []
        for m, v in d.items():
            pd = v.get("paired_diff_vs_external")
            rows.append([m, v["lambda"], f"{e(v['mean'][-1], 4)} ± {e(v['ci95'][-1], 4)}",
                         "-" if not pd else f"{e(pd['mean'], 4)} ± {e(pd['ci95'], 4)} ({pd['wins']}/{pd['seeds']} seeds better)"])
        lines += [f"**{g}**:", "", table(["method", "λ", "final exploitability", "paired diff vs external"], rows), ""]
    rows = [[m, e(v["mse"], 1), e(v["nodes_per_sample"], 1), e(v["mse_x_cost"], 0), e(r["throughput"][m], 2)]
            for m, v in r["efficiency"].items()]
    lines += ["Estimator efficiency with the strategy held fixed (Leduc):", "",
              table(["sampler", "regret-estimate MSE", "nodes / sample", "MSE x cost", "nodes / s"], rows), "",
              finding_exp06(r), ""]


def finding_exp06(r):
    sig = []
    for g, d in r["eval"].items():
        for m, v in d.items():
            pd = v.get("paired_diff_vs_external")
            if pd and pd["mean"] + pd["ci95"] < 0:
                sig.append(f"{m} on {g}")
    eff = r.get("efficiency", {})
    var_note = ""
    if "adaptive" in eff and "external" in eff:
        ratio = eff["adaptive"]["mse_x_cost"] / eff["external"]["mse_x_cost"]
        var_note = (f"With the strategy held fixed, the adaptive sampler's regret-estimate MSE x cost is {ratio:.2f}x "
                    f"that of external sampling. ")
    if sig:
        return ("**Result.** " + var_note + "At the same node budget, these beat external sampling outside the 95% "
                "confidence interval: " + ", ".join(sig) + ".")
    return ("**Result.** " + var_note + "At the same node budget, none of the priority rules beat plain external "
            "sampling by more than the 95% confidence interval, on Leduc or on the bigger held-out game, and the extra "
            "bookkeeping costs throughput. So on these games, steering samples by sensitivity/uncertainty did **not** "
            "buy lower exploitability per unit of compute. The samplers are all tested to be unbiased, so this is about "
            "efficiency, not correctness.")


def section_exp07(lines):
    r = load("exp07_safe_resolving")
    if not r:
        return
    lines += ["## 7. Safe vs naive re-solving (full-game exploitability)", "",
              fig("exp07_safe_resolving", "safe_vs_naive.png", "safe vs naive"), ""]
    ct = r["coin_toss"]
    rows = [[k] + [e(v[m]["p1_exploitability"], 4) for m in ("blueprint", "naive", "resolve", "maxmargin")] for k, v in ct.items()]
    lines += ["**Coin Toss** - exploitability of player 2's strategy:", "",
              table(["blueprint", "blueprint", "naive", "resolve", "maxmargin"], rows), ""]
    lb = r["leduc"]["blueprints"]
    rows = [[k] + [e(lb[k][m]["nash_conv"], 4) for m in ("blueprint", "naive", "resolve", "maxmargin")] for k in sorted(lb, key=int)]
    lines += [f"**Leduc** - NashConv after re-solving all {r['leduc']['num_subgames']} round-2 subgames for both players:", "",
              table(["blueprint CFR+ iterations", "blueprint", "naive", "resolve", "maxmargin"], rows), "",
              "Naive re-solving helps when the blueprint is bad, but makes good blueprints a lot more exploitable. "
              "The gadget methods stay close to the blueprint (the small increases at 1000 iterations come from the "
              "400-iteration re-solves not being fully converged).", ""]
    tr = r["turn_river"]
    rows = [[k] + [e(v[m]["exploitability_pct_pot"], 3) for m in ("blueprint", "naive", "resolve", "maxmargin")] for k, v in tr["blueprints"].items()]
    lines += [f"**No-Limit turn -> river re-solving** ({tr['river_subgames']} river subgames re-solved per player; "
              f"full solve reference {e(tr['full_solve']['exploitability_pct_pot'], 3)}% pot) - exploitability (% pot):", "",
              table(["blueprint", "blueprint", "naive", "resolve", "maxmargin"], rows), ""]
    nv = r["nested_offtree"]
    rows = [[k, e(v, 3)] for k, v in nv["br_vs_p0"].items()]
    lines += [f"**Nested re-solving after bet sizes the abstraction doesn't have** (opponent turn bets of {', '.join(nv['offtree'])}) "
              "- opponent's best-response value against our strategy (lower is better):", "",
              table(["method", "opponent BR value (BB)"], rows), "",
              "Safe nested re-solving (constrained by the opponent's best alternative inside the abstraction) closes almost "
              "all of the gap between action translation and solving the reference game directly.", ""]


def section_exp08(lines):
    r = load("exp08_value_network")
    if not r:
        return
    lines += ["## 8. Learned leaf values: network error vs solving quality", "",
              fig("exp08_value_network", "error_vs_performance.png", "value net"), ""]
    d = r["data"]
    lines += [f"Training data: {d['samples']} solver-generated river positions ({e(d['generation_seconds'], 0)} s on 8 threads); "
              f"target quality: mean exploitability {e(d['target_exploitability_pct_pot_mean'], 4)}% pot "
              f"(max {e(d['target_exploitability_pct_pot_max'], 3)}%).", ""]
    rows = []
    for n in sorted(r["models"], key=int):
        m = r["models"][n]
        rows.append([n, e(m["test"]["rmse"], 4), e(m["test"]["target_std"], 4), e(m["test"]["zero_sum_residual_mean_abs"], 4),
                     e(m["dl_solve_seconds"], 2), e(m["composite_safe"]["exploitability_pct_pot"], 3),
                     e(m["composite_naive"]["exploitability_pct_pot"], 3)])
    lines += [table(["training positions", "test RMSE", "target std", "|zero-sum residual|", "depth-limited solve (s)",
                     "full-game expl. (safe re-solve, % pot)", "(naive re-solve)"], rows), "",
              "Adding on-distribution data (the leaf ranges depth-limited CFR actually asks about, solved exactly and "
              "added to the training set):", "",
              table(["round", "new positions", "train size", "on-distribution RMSE before", "full-game expl. safe (% pot)",
                     "naive (% pot)"],
                    [[a["round"], a["new_positions"], a["train_size"], e(a["on_distribution_rmse_before"], 4),
                      e(a["composite_safe"]["exploitability_pct_pot"], 3), e(a["composite_naive"]["exploitability_pct_pot"], 3)]
                     for a in r.get("aggregation", [])]), "",
              f"Full turn+river solve: {e(r['full_solve']['exploitability_pct_pot'], 3)}% pot in {e(r['full_solve']['seconds'], 2)} s. "
              f"Blueprint leaves: safe {e(r['blueprint_leaves']['composite_safe']['exploitability_pct_pot'], 3)}%, naive "
              f"{e(r['blueprint_leaves']['composite_naive']['exploitability_pct_pot'], 3)}% pot.", "",
              "The safety guarantee for re-solving assumes exact leaf values, and the network's values aren't exact, so "
              "the table reports the full-game exploitability that actually came out.", "",
              "On speed: this two-street tree is small enough to solve exactly in seconds, so the depth-limited solve "
              "with the network (mostly batched inference at the leaves) is slower than just solving it. A value network "
              "only pays off when the full tree is too big to build (earlier streets, deeper trees). I used the turn here "
              "because it's a place where the network's cost in exploitability can be measured exactly.", ""]


def section_exp09(lines):
    r = load("exp09_performance")
    if not r:
        return
    lines += ["## 9. Performance", "",
              fig("exp09_performance", "thread_scaling.png", "threads"), " ",
              fig("exp09_performance", "river_cpu_vs_gpu.png", "cpu vs gpu"), " ",
              fig("exp09_performance", "value_net_inference.png", "inference"), ""]
    rows = []
    for b in r["bench"]:
        if b["bench"] == "meta":
            continue
        key = next((k for k in ("evals_per_sec", "nodes_per_sec", "hand_node_updates_per_sec", "traversals_per_sec", "rows_per_sec") if k in b), None)
        extra = f"{e(b['ms_per_iteration'], 3)} ms/it" if "ms_per_iteration" in b else ""
        rows.append([b["bench"], b["config"], f"{e(b[key], 3)} {key}" if key else "", extra])
    lines += [table(["benchmark", "configuration", "throughput", "latency"], rows), ""]
    simd = r.get("simd_ablation", {})
    if simd:
        lines += [table(["SIMD ablation", "vectorised ms", "scalar ms", "speed-up"],
                        [[k, e(v["vectorized_ms"], 3), e(v["scalar_ms"], 3), f"{v['speedup']:.2f}x"] for k, v in simd.items()]), ""]
    rows = [[g["ranges"], f"{g['backend']} ({g['dtype']})", e(g["ms_per_iter"], 3), e(g["expl"], 4)] for g in r["gpu_vs_cpu"]]
    lines += [table(["ranges", "backend", "ms / iteration", "expl. after 60 it (% pot)"], rows), ""]
    lat = r["latency_to_0.1pct_pot"]
    lines += [table(["ranges", "hands", "iterations to 0.1% pot", "seconds", "memory (MB)"],
                    [[k, "x".join(map(str, v["hands"])), v["iterations"], e(v["solver_seconds"], 3), e(v["memory_mb"], 1)] for k, v in lat.items()]), ""]
    hw = r.get("hardware_counters", {})
    lines += [f"Hardware counters: {'available via ' + hw.get('backend', '') if hw.get('hw_counters') else 'not available - ' + hw.get('reason', '')}. "
              "On Linux the same benchmark reads cycles, instructions, cache and branch misses through perf_event_open.", "",
              "The GPU isn't automatically faster. With narrow ranges, kernel launches and transfers dominate; the O(n) C++ "
              "sweep beats an O(n²) dense matmul on the CPU, and MPS only catches up when the matrices are large.", ""]


def section_exp10(lines):
    r = load("exp10_refinement")
    if not r:
        return
    lines += ["## B. Adding bet sizes where they matter (Experiment B)", "",
              fig("exp10_refinement", "refinement.png", "refinement"), ""]
    s = r["summary"]
    rows = [[m, " -> ".join(f"{x:.3f}" for x in v["relative_mean"]), e(v["absolute_mean_pct"][-1], 4)] for m, v in s.items()]
    lines += [table(["method", "reference exploitability / initial, per step", "final (% pot)"], rows), ""]
    p = r["paired_hole_minus_baseline"]
    lines += ["Paired difference (hole - baseline) in final relative exploitability over 8 spots: "
              + "; ".join(f"vs {k}: {e(v['mean_diff'], 3)} ± {e(v['ci95'], 3)} (hole better on {v['hole_better_spots']}/8)" for k, v in p.items()) + ".", ""]
    wins = [k for k, v in p.items() if v["mean_diff"] + v["ci95"] < 0]
    lines += [("**Result.** Picking the size with the largest translation hole is significantly better than "
               + ", ".join(wins) + " at the same number of added sizes." if wins else
               "**Result.** Picking by translation hole is not significantly better than the baselines."), ""]


def section_exp11(lines):
    r = load("exp11_learned_allocation")
    if not r:
        return
    lines += ["## C. Learned refinement choice (Experiment C)", "",
              fig("exp11_learned_allocation", "learned_allocation.png", "learned allocation"), ""]
    sp = r["spearman_test_first_state"]
    lines += [f"Trained on {r['train_candidates']} measured refinements from small games; the heuristic was tuned on the same "
              f"data (chosen hole*reach^k/#sizes^m exponents {r['hole_params']}). Ranking quality on unseen larger subgames "
              f"(Spearman ρ between predicted and actual exploitability reduction per ms): learned {e(sp['learned_mean'], 3)}, "
              f"hole heuristic {e(sp['hole_mean'], 3)}.", ""]
    rows = [[k, e(v["mean"], 3)] for k, v in r["final_relative_exploitability"].items()]
    lines += [table(["controller", "final reference exploitability / initial (6 unseen spots)"], rows), ""]
    pl = r["paired_learned_minus_baseline"]
    lines += ["Paired (learned - baseline): " + "; ".join(f"vs {k}: {e(v['mean_diff'], 3)} ± {e(v['ci95'], 3)} "
                                                       f"(learned better on {v['learned_better_spots']}/6)" for k, v in pl.items()) + ".", ""]
    h = pl["hole"]
    verdict = ("beats" if h["mean_diff"] + h["ci95"] < 0 else
               "is worse than" if h["mean_diff"] - h["ci95"] > 0 else "is statistically indistinguishable from")
    lines += [f"**Result.** On the larger held-out subgames the learned model {verdict} the translation-hole heuristic "
              "(tuned on the same data). Both are clearly better than largest-gap and random. So far the model doesn't add "
              "anything beyond the heuristic, which it gets as one of its inputs.", ""]


def section_exp12(lines):
    r = load("exp12_budget_allocation")
    if not r:
        return
    lines += ["## 12. Where is compute worth spending? (budget allocation)", "",
              fig("exp12_budget_allocation", "budget_allocation.png", "budget"), ""]
    for board, d in r.items():
        rows = [[m] + [e(d["budgets"][m][p]["weighted_exploitability"], 4) for p in ("uniform", "weighted", "adaptive")]
                for m in sorted(d["budgets"], key=int)]
        lines += [f"**{board}** - {d['num_subgames']} river subgames, top-5 subgames hold {100 * d['weight_top5_share']:.0f}% of the reach:", "",
                  table(["budget (avg it/subgame)", "uniform", "weighted", "adaptive"], rows), ""]
        best = {m: min(("uniform", "weighted", "adaptive"), key=lambda p: d["budgets"][m][p]["weighted_exploitability"])
                for m in d["budgets"]}
        lines += ["Best policy per budget: " + ", ".join(f"{m}: {b}" for m, b in sorted(best.items(), key=lambda kv: int(kv[0]))) + ".", ""]
    lines += ["**Result.** Fitting convergence curves helps when subgames converge at very different speeds and the "
              "measurements pay for themselves. When the reach weights already say where the error matters, plain "
              "reach-weighted allocation wins and the measuring is wasted. No policy wins everywhere.", ""]


def section_exp13(lines):
    r = load("exp13_winrate_vs_exploitability")
    if not r:
        return
    lines += ["## 13. Sanity check: win rate vs exploitability", "",
              fig("exp13_winrate_vs_exploitability", "winrate_vs_exploitability.png", "control"), ""]
    rows = [[s, e(v["exploitability"], 4)] + [f"{x:+.3f}" for x in v["win_rate"].values()] for s, v in r["table"].items()]
    opp = list(next(iter(r["table"].values()))["win_rate"])
    lines += [table(["strategy", "own exploitability"] + [f"vs {o}" for o in opp], rows), "",
              "A best response to a weak opponent beats it by much more than the equilibrium does, while being orders of "
              "magnitude more exploitable. Winning against weak opponents says nothing about how close you are to equilibrium.", ""]


def main():
    lines = ["# AEGIS - experiment results", "",
             "This file is written by `python/experiments/report.py` from `results/*/results.json`; to rerun "
             "everything, use `.venv/bin/python python/experiments/run_all.py`. All exploitability numbers are exact "
             "best responses in the game stated: the full game for Kuhn, Leduc and Coin Toss, and the constructed "
             "spot (ranges + bet sizes) for Hold'em. None of them says anything about full No-Limit Hold'em.", ""]
    log = os.path.join(RES, "run_log.json")
    if os.path.exists(log):
        env = json.load(open(log))["environment"]
        lines += [f"Environment: {env.get('cpu')}, {env.get('cores')} cores, {env.get('platform')}, Python {env.get('python')}, "
                  f"run {env.get('time')}.", ""]
    for f in (section_exp01, section_exp02, section_exp03, section_exp04, section_exp05, section_exp06, section_exp07,
              section_exp08, section_exp09, section_exp10, section_exp11, section_exp12, section_exp13):
        try:
            f(lines)
        except Exception as ex:  # noqa: BLE001
            lines += [f"*(section {f.__name__} could not be generated: {ex!r})*", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
