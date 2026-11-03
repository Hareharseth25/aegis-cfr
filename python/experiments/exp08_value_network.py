"""Experiment 8 - Learned continuation values: error vs solving performance.

Turn spot (board Kd9s6h3c, pot 100, stacks 150, turn sizes {½, 1} pot, river
sizes {½, 1} pot + all-in). Exact machinery first, approximation second:

  1. Solver-generated data: river subgames for every river card with random
     strength-correlated ranges, solved by DCFR; targets are per-hand
     counterfactual values of both players (pot/mass normalised).
  2. Value networks trained on growing data sizes; held-out RMSE/MAE and
     zero-sum residual before/after projection.
  3. Depth-limited turn solving with NEURAL leaves (batched C++ inference),
     followed by continual river re-solving (safe gadget with the network's
     opponent values as constraints, or naive), evaluated by EXACT
     exploitability in the full turn+river game. Baselines: the full-tree
     solve (ground truth) and depth-limited solving with blueprint leaves.

Approximation error means safety is NOT guaranteed by the re-solving theory
alone; this experiment measures the realised exploitability.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import color, outdir, plt, save_fig, save_json  # noqa: E402
from experiments.exp07_safe_resolving import R0, R1, turn_config  # noqa: E402

import aegis  # noqa: E402
from training.train_value_net import evaluate, split, train  # noqa: E402

NAME = "exp08_value_network"
BOARD = "Kd9s6h3c"
POT, STACK = 100.0, 150.0
TURN_BETS, RIVER_BETS = [0.5, 1.0], [0.5, 1.0]
COMBO_CARDS = aegis.combo_cards()


def games():
    cfg = turn_config(BOARD, POT, STACK, TURN_BETS, RIVER_BETS)
    full = aegis.PublicTree.build(cfg)
    cfg.depth_limited = True
    dl = aegis.PublicTree.build(cfg)
    r0 = aegis.Range.parse(R0).remove(cfg.board)
    r1 = aegis.Range.parse(R1).remove(cfg.board)
    return cfg, full, dl, r0, r1


def leaf_pot_stacks(dl):
    out = set()
    for n in range(dl.num_nodes):
        if dl.node(n).type == aegis.PNodeType.LEAF:
            out.add((dl.pot(n), dl.stack_remaining(n, 0)))
    return sorted(out)


def make_data(dl, n, seed, threads=8):
    path = os.path.join(outdir(NAME), f"data_{n}_{seed}.npz")
    if os.path.exists(path):
        d = dict(np.load(path))
        if d["features"].shape[1] != aegis.NN_INPUT_SIZE:
            d["features"] = refeaturize(d)
            np.savez_compressed(path, **d)
        return d
    c = aegis.RiverSampleConfig()
    c.turn_board = aegis.parse_cards(BOARD)
    c.pot_stack = leaf_pot_stacks(dl)
    rs = aegis.StreetSizes()
    rs.bet_fractions, rs.raise_fractions, rs.max_raises = RIVER_BETS, [], 1
    c.river_sizes = rs
    c.iterations = 250
    c.seed = seed
    c.threads = threads
    t = time.perf_counter()
    d = aegis.generate_river_samples(c, n)
    d["generation_seconds"] = np.array(time.perf_counter() - t)
    np.savez_compressed(path, **d)
    return d


def refeaturize(d):
    """Rebuild input features (e.g. after adding feature channels) from the
    stored boards, normalised ranges, pots and stacks - no re-solving needed."""
    old = d["features"]
    feats = np.zeros((old.shape[0], aegis.NN_INPUT_SIZE), dtype=np.float32)
    for k in range(old.shape[0]):
        r0 = old[k, 52:52 + 1326].astype(float).tolist()
        r1 = old[k, 52 + 1326:52 + 2652].astype(float).tolist()
        feats[k] = aegis.build_features([int(c) for c in d["boards"][k]], r0, r1, float(d["pot"][k]), float(d["stack"][k]))
    return feats


def leaf_positions(dl, dl_ctx, policies):
    """River positions (board, ranges, pot, stack) that depth-limited CFR queries
    at its leaves under the given policies (one per river card)."""
    hs = [dl_ctx.hands(0), dl_ctx.hands(1)]
    cards = aegis.combo_cards()
    out = []
    for pol in policies:
        for n in range(dl.num_nodes):
            if dl.node(n).type != aegis.PNodeType.LEAF:
                continue
            r = aegis.reach_at(dl_ctx, pol, n)
            base = [to_combo(hs[0], r[0]), to_combo(hs[1], r[1])]
            board = list(dl.board(dl.node(n).board_id))
            for c in range(52):
                if c in board:
                    continue
                keep = ~((cards[:, 0] == c) | (cards[:, 1] == c))
                rr = [b * keep for b in base]
                if rr[0].sum() <= 1e-9 or rr[1].sum() <= 1e-9:
                    continue
                out.append((board + [c], rr[0], rr[1], dl.pot(n), dl.stack_remaining(n, 0)))
    return out


def solve_positions(positions, iterations=250, threads=8):
    """Exact (DCFR-solved) training samples for given river positions."""
    from concurrent.futures import ThreadPoolExecutor
    rs = aegis.StreetSizes()
    rs.bet_fractions, rs.raise_fractions, rs.max_raises = RIVER_BETS, [], 1
    cfg = aegis.CFRConfig.dcfr()

    def one(pos):
        b5, r0, r1, pot, stack = pos
        c0, c1, ex = aegis.solve_river_values(b5, r0.tolist(), r1.tolist(), pot, stack, rs, iterations, cfg)
        f = aegis.build_features(b5, r0.tolist(), r1.tolist(), pot, stack)
        live = np.array([not ({int(x) for x in cc} & set(b5)) for cc in aegis.combo_cards()])
        m0, m1 = r0[live].sum(), r1[live].sum()
        t = np.concatenate([c0 / (pot * m1), c1 / (pot * m0)]).astype(np.float32)
        mask = np.concatenate([live, live]).astype(np.float32)
        return f, t * mask, mask, ex

    with ThreadPoolExecutor(threads) as ex:
        rows = list(ex.map(one, positions))
    return {"features": np.stack([r[0] for r in rows]), "targets": np.stack([r[1] for r in rows]),
            "mask": np.stack([r[2] for r in rows]), "exploitability_pct_pot": np.array([r[3] for r in rows])}


def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in ("features", "targets", "mask")}


# ------------------------------------------------------------- composites
def path_labels(tree, n):
    labels = []
    while n > 0:
        labels.append(tree.node(n).action.to_string())
        n = tree.node(n).parent
    return labels[::-1]


def follow(tree, labels):
    n = 0
    for lab in labels:
        n = tree.child_by_label(n, lab)
        assert n >= 0, labels
    return n


def turn_policy_into_full(full_ctx, full, dl, dl_policy):
    pol = full_ctx.uniform_policy()
    for d in range(dl.num_nodes):
        nd = dl.node(d)
        if nd.type == aegis.PNodeType.DECISION and nd.street == 0:
            f = follow(full, path_labels(dl, d))
            pol.set(full.node(f).decision_index, dl_policy.get(nd.decision_index))
    return pol


def to_combo(hs, local):
    v = np.zeros(1326)
    v[np.array(hs.combos)] = local
    return v


def nn_constraint(net, full_ctx, full, pol, root, opp):
    r = aegis.reach_at(full_ctx, pol, root)
    hs = [full_ctx.hands(0), full_ctx.hands(1)]
    c0, c1 = to_combo(hs[0], r[0]), to_combo(hs[1], r[1])
    board = full.board(full.node(root).board_id)
    pot = full.pot(root)
    stack = full.stack_remaining(root, 0)
    x = aegis.build_features(board, c0.tolist(), c1.tolist(), pot, stack)[None, :]
    y = net.forward(x)[0].astype(float)
    r0n, r1n = x[0, 52:52 + 1326].astype(float), x[0, 52 + 1326:52 + 2652].astype(float)  # normalised ranges
    e = (r0n * y[:1326]).sum() + (r1n * y[1326:]).sum()
    y[:1326] -= 0.5 * e / max(r0n.sum(), 1e-12)
    y[1326:] -= 0.5 * e / max(r1n.sum(), 1e-12)
    us = 1 - opp
    mass_us = (c0, c1)[us].sum()
    v_opp = y[opp * 1326:(opp + 1) * 1326]
    return (v_opp[np.array(hs[opp].combos)] * pot * mass_us).tolist()


def composite(full_ctx, full, base_turn_policy, mode, constraint_fn=None, iters=150):
    """Re-solve every river subgame for each player; combine; exact NashConv."""
    per_player = {}
    t0 = time.perf_counter()
    for us in (0, 1):
        pol = base_turn_policy
        for root in aegis.street_roots(full, 1):
            o = aegis.VectorResolveOptions()
            o.us, o.iterations = us, iters
            if mode == "naive":
                o.method = aegis.ResolveMethod.NAIVE
            else:
                o.method = aegis.ResolveMethod.RESOLVE
                o.constraint = constraint_fn(pol, root, 1 - us)
            pol = aegis.resolve_subtree(full_ctx, pol, root, o).policy
        per_player[us] = pol
    both = per_player[0].copy()
    for n in range(full.num_nodes):
        nd = full.node(n)
        if nd.type == aegis.PNodeType.DECISION and nd.player == 1:
            both.set(nd.decision_index, per_player[1].get(nd.decision_index))
    e = full_ctx.exploitability(both)
    return {"nash_conv": e["nash_conv"], "exploitability_pct_pot": e["exploitability_pct_pot"],
            "resolve_seconds": time.perf_counter() - t0}


def run(n_total=20000, iterations=400):
    cfg, full, dl, r0, r1 = games()
    full_ctx = aegis.SubgameContext(full, r0, r1, 8)
    dl_ctx = aegis.SubgameContext(dl, r0, r1, 1)
    out = {"leaf_pot_stacks": leaf_pot_stacks(dl), "full_nodes": full.num_nodes, "dl_nodes": dl.num_nodes,
           "river_subgames": len(aegis.street_roots(full, 1)), "hands": [full_ctx.n(0), full_ctx.n(1)]}
    # Ground truth.
    t = time.perf_counter()
    fs = aegis.VectorCFR(full_ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    fs.iterate(iterations)
    out["full_solve"] = {"seconds": time.perf_counter() - t, **full_ctx.exploitability(fs.average_policy())}
    print("full solve:", out["full_solve"], flush=True)

    # 1. Data.
    data = make_data(dl, n_total, seed=11)
    out["data"] = {"samples": int(data["features"].shape[0]), "generation_seconds": float(data["generation_seconds"]),
                   "target_exploitability_pct_pot_mean": float(np.mean(data["exploitability_pct_pot"])),
                   "target_exploitability_pct_pot_max": float(np.max(data["exploitability_pct_pot"]))}
    print("data:", out["data"], flush=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_total)
    test_idx, val_idx, train_pool = perm[:2000], perm[2000:4000], perm[4000:]
    test, val = split(data, test_idx), split(data, val_idx)

    # 2-3. Learning curve: error vs solving performance.
    out["models"] = {}
    for n_train in (1000, 4000, 8000, len(train_pool)):
        tr = split(data, train_pool[:n_train])
        t = time.perf_counter()
        model, hist = train(tr, val, epochs=40, verbose=False, seed=1)
        train_s = time.perf_counter() - t
        err = evaluate(model, test)
        err_proj = evaluate(model, test, project=True)
        net = model.to_aegis()
        net.save(os.path.join(outdir(NAME), f"value_net_{n_train}.bin"))
        leaf = aegis.NeuralLeafProvider(net, True)
        t = time.perf_counter()
        ds = aegis.VectorCFR(dl_ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()), leaf)
        ds.iterate(iterations)
        dl_s = time.perf_counter() - t
        turn_pol = turn_policy_into_full(full_ctx, full, dl, ds.average_policy())
        safe = composite(full_ctx, full, turn_pol, "safe",
                         lambda pol, root, opp: nn_constraint(net, full_ctx, full, pol, root, opp))
        naive = composite(full_ctx, full, turn_pol, "naive")
        out["models"][n_train] = {"train_seconds": train_s, "test": err, "test_projected": err_proj,
                                  "history": hist, "dl_solve_seconds": dl_s,
                                  "leaf_stats": {"calls": leaf.stats.calls, "rows": leaf.stats.rows,
                                                 "seconds": leaf.stats.seconds},
                                  "composite_safe": safe, "composite_naive": naive}
        print(f"n_train={n_train}: test rmse {err['rmse']:.4f} (std {err['target_std']:.4f}), zero-sum residual "
              f"{err['zero_sum_residual_mean_abs']:.4f}; DL {dl_s:.1f}s; composite safe "
              f"{safe['exploitability_pct_pot']:.3f}% / naive {naive['exploitability_pct_pot']:.3f}% pot", flush=True)

    # On-distribution data aggregation (DAgger-style): collect the leaf ranges
    # depth-limited CFR actually queries, solve them exactly, measure the
    # network's error there, add them to the training set and retrain.
    out["aggregation"] = []
    train_set = split(data, train_pool)
    net = model.to_aegis()
    rng_pos = np.random.default_rng(5)
    for rnd in range(2):
        leaf = aegis.NeuralLeafProvider(net, True)
        ds = aegis.VectorCFR(dl_ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()), leaf)
        pols = []
        for chunk in (25, 75, 150, 150):
            ds.iterate(chunk)
            pols += [ds.current_policy(), ds.average_policy()]
        positions = leaf_positions(dl, dl_ctx, pols)
        pick = rng_pos.choice(len(positions), size=min(3000, len(positions)), replace=False)
        on_dist = solve_positions([positions[i] for i in pick])
        err_before = evaluate(model, on_dist)
        train_set = concat(train_set, on_dist)
        t = time.perf_counter()
        model, hist = train(train_set, val, epochs=40, verbose=False, seed=2 + rnd)
        train_s = time.perf_counter() - t
        net = model.to_aegis()
        net.save(os.path.join(outdir(NAME), f"value_net_aggregated_{rnd + 1}.bin"))
        leaf = aegis.NeuralLeafProvider(net, True)
        t = time.perf_counter()
        ds = aegis.VectorCFR(dl_ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()), leaf)
        ds.iterate(iterations)
        dl_s = time.perf_counter() - t
        turn_pol = turn_policy_into_full(full_ctx, full, dl, ds.average_policy())
        safe = composite(full_ctx, full, turn_pol, "safe",
                         lambda pol, root, opp: nn_constraint(net, full_ctx, full, pol, root, opp))
        naive = composite(full_ctx, full, turn_pol, "naive")
        row = {"round": rnd + 1, "new_positions": len(pick), "train_size": int(train_set["features"].shape[0]),
               "on_distribution_rmse_before": err_before["rmse"], "on_distribution_target_std": err_before["target_std"],
               "random_range_test_rmse_after": evaluate(model, test)["rmse"], "train_seconds": train_s,
               "dl_solve_seconds": dl_s, "composite_safe": safe, "composite_naive": naive}
        out["aggregation"].append(row)
        print(f"aggregation round {rnd + 1}: on-distribution RMSE before {err_before['rmse']:.4f} "
              f"(std {err_before['target_std']:.4f}); composite safe {safe['exploitability_pct_pot']:.3f}% / "
              f"naive {naive['exploitability_pct_pot']:.3f}% pot", flush=True)

    # Blueprint leaves baseline (coarse river abstraction, same turn tree).
    bcfg = turn_config(BOARD, POT, STACK, TURN_BETS, [0.75])
    btree = aegis.PublicTree.build(bcfg)
    bctx = aegis.SubgameContext(btree, r0, r1, 8)
    bsol = aegis.VectorCFR(bctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
    bsol.iterate(300)
    bp = bsol.average_policy()
    leaf = aegis.BlueprintLeafProvider(bctx, bp)
    t = time.perf_counter()
    ds = aegis.VectorCFR(dl_ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()), leaf)
    ds.iterate(iterations)
    dl_s = time.perf_counter() - t
    turn_pol = turn_policy_into_full(full_ctx, full, dl, ds.average_policy())

    def bp_constraint(pol, root, opp):
        r = aegis.reach_at(full_ctx, pol, root)
        broot = follow(btree, path_labels(full, root))
        return bctx.subtree_values(bp, broot, r[0].tolist(), r[1].tolist(), -1)[opp].tolist()

    out["blueprint_leaves"] = {"dl_solve_seconds": dl_s, "composite_safe": composite(full_ctx, full, turn_pol, "safe", bp_constraint),
                               "composite_naive": composite(full_ctx, full, turn_pol, "naive")}
    print("blueprint leaves:", out["blueprint_leaves"], flush=True)
    save_json(NAME, "results.json", out)
    plot(out)
    return out


def plot(out):
    ns = sorted(out["models"], key=int)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    ax.plot([int(n) for n in ns], [out["models"][n]["test"]["rmse"] for n in ns], color=color("CFR"), marker="o",
            label="held-out RMSE (pot-normalised cfv)")
    ax.set_xscale("log")
    ax.set_xlabel("training positions")
    ax.set_ylabel("RMSE")
    ax.set_title("Value-network error vs data")
    ax.legend()
    ax = axes[1]
    ax.plot([int(n) for n in ns], [out["models"][n]["composite_safe"]["exploitability_pct_pot"] for n in ns],
            color=color("resolve"), marker="o", label="neural leaves + safe river re-solve")
    ax.plot([int(n) for n in ns], [out["models"][n]["composite_naive"]["exploitability_pct_pot"] for n in ns],
            color=color("naive"), marker="s", label="neural leaves + naive river re-solve")
    if out.get("aggregation"):
        n_last = int(ns[-1])
        xs = [n_last + sum(r["new_positions"] for r in out["aggregation"][:k + 1]) for k in range(len(out["aggregation"]))]
        ax.plot(xs, [r["composite_safe"]["exploitability_pct_pot"] for r in out["aggregation"]], color=color("nested"),
                marker="D", label="+ on-distribution aggregation (safe)")
    ax.axhline(out["full_solve"]["exploitability_pct_pot"], color=color("reference-solve"), linestyle="--",
               label="full turn+river solve")
    ax.axhline(out["blueprint_leaves"]["composite_safe"]["exploitability_pct_pot"], color=color("blueprint"),
               linestyle=":", label="blueprint leaves + safe re-solve")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training positions")
    ax.set_ylabel("full-game exploitability (% pot)")
    ax.set_title("Error vs solving performance")
    ax.legend(fontsize=8)
    save_fig(fig, NAME, "error_vs_performance.png")


if __name__ == "__main__":
    run()
