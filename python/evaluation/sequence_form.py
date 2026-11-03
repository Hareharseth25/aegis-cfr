"""Exact Nash equilibrium of a two-player zero-sum game via the sequence-form
linear program (Koller, Megiddo & von Stengel 1994).

With realisation plans x (player 0) and y (player 1), E x = e, F y = f,
x, y >= 0, and sequence payoff matrix A (to player 0), player 0 solves

    max_{x, q}  fᵀq   s.t.   Fᵀq - Aᵀx <= 0,   E x = e,   x >= 0.

The optimum is the game value. Behaviour strategies are recovered as
σ(I, a) = x(I, a) / x(parent(I)). This is an independent exact oracle for
the CFR family (which only approaches the equilibrium in the limit).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import aegis


def _chance_reach(tree) -> np.ndarray:
    n = tree.num_nodes
    reach = np.ones(n)
    nt, cb, cc, ch, cp = tree.node_type, tree.child_begin, tree.child_count, tree.children, tree.chance_prob
    for h in range(n):
        if nt[h] == int(aegis.NodeType.TERMINAL):
            continue
        for k in range(cc[h]):
            c = ch[cb[h] + k]
            reach[c] = reach[h] * (cp[cb[h] + k] if nt[h] == int(aegis.NodeType.CHANCE) else 1.0)
    return reach


def _constraints(tree, player):
    """E matrix (rows: root + one per infoset of `player`) over sequences
    (index 0 = empty, slot s of the player's infoset -> s+1 in global slot space)."""
    isets = [I for I, info in enumerate(tree.infosets) if info.player == player]
    seq_of_slot = {}
    seq = 1
    for I in isets:
        info = tree.infoset(I)
        for a in range(info.num_actions):
            seq_of_slot[info.offset + a] = seq
            seq += 1
    rows, cols, vals = [0], [0], [1.0]
    for r, I in enumerate(isets, start=1):
        info = tree.infoset(I)
        for a in range(info.num_actions):
            rows.append(r)
            cols.append(seq_of_slot[info.offset + a])
            vals.append(1.0)
        parent = 0 if info.parent_sequence < 0 else seq_of_slot[info.parent_sequence]
        rows.append(r)
        cols.append(parent)
        vals.append(-1.0)
    E = sp.csr_matrix((vals, (rows, cols)), shape=(len(isets) + 1, seq))
    e = np.zeros(len(isets) + 1)
    e[0] = 1.0
    return E, e, seq_of_slot, isets


def solve(tree) -> dict:
    """Solve the game exactly. Returns value (to player 0) and a TabularPolicy."""
    reach_c = _chance_reach(tree)
    E, e, s0, isets0 = _constraints(tree, 0)
    F, f, s1, isets1 = _constraints(tree, 1)
    seq0, seq1 = tree.sequence(0), tree.sequence(1)
    u0 = tree.utility(0)
    nt = tree.node_type
    rows, cols, vals = [], [], []
    for z in range(tree.num_nodes):
        if nt[z] != int(aegis.NodeType.TERMINAL):
            continue
        a = 0 if seq0[z] < 0 else s0[seq0[z]]
        b = 0 if seq1[z] < 0 else s1[seq1[z]]
        rows.append(a)
        cols.append(b)
        vals.append(reach_c[z] * u0[z])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(E.shape[1], F.shape[1]))

    def side(A_, E_, e_, F_, f_):
        nx, nq = E_.shape[1], F_.shape[0]
        # variables [x (nx), q (nq)]; maximise fᵀq -> minimise -fᵀq
        c = np.concatenate([np.zeros(nx), -f_])
        A_ub = sp.hstack([-A_.T, F_.T]).tocsr()
        b_ub = np.zeros(F_.shape[1])
        A_eq = sp.hstack([E_, sp.csr_matrix((E_.shape[0], nq))]).tocsr()
        bounds = [(0, None)] * nx + [(None, None)] * nq
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=e_, bounds=bounds, method="highs")
        assert res.status == 0, res.message
        return -res.fun, res.x[:nx]

    v0, x = side(A, E, e, F, f)
    v1, y = side(-A.T, F, f, E, e)
    pol = aegis.TabularPolicy(tree)
    probs = pol.probs
    for player, plan, smap in ((0, x, s0), (1, y, s1)):
        for I, info in enumerate(tree.infosets):
            if info.player != player:
                continue
            parent = 1.0 if info.parent_sequence < 0 else plan[smap[info.parent_sequence]]
            for a in range(info.num_actions):
                probs[info.offset + a] = plan[smap[info.offset + a]] / parent if parent > 1e-12 else 1.0 / info.num_actions
            tot = sum(probs[info.offset + a] for a in range(info.num_actions))
            for a in range(info.num_actions):
                probs[info.offset + a] /= tot
    pol.probs = probs
    return {"value_p0": v0, "value_p1": v1, "policy": pol, "num_sequences": (E.shape[1], F.shape[1])}
