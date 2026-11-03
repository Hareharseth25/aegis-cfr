"""PyTorch implementation of vector-form river DCFR (the "GPU alternative").

Same algorithm as the C++ VectorCFR (alternating DCFR(1.5, 0, 2), regret
matching, per-hand regrets), but terminal values use dense matrices instead of
the O(n) sorted sweep:

    showdown:  cfv_p = C_node * (W_p @ r_o),   W_p[i, j] = sign(s_i - s_j)*[i ∩ j = empty]
    fold:      cfv_p = u_node * (K_p @ r_o),   K_p[i, j] = [i ∩ j = empty]

which is O(n²) work but maps onto a GPU matmul. All terminal nodes of a kind
are batched into one matmul per pass. Used only for the CPU-vs-GPU benchmark;
correctness is checked against the C++ solver's exploitability.
"""
from __future__ import annotations

import numpy as np
import torch

import aegis


class TorchRiverCFR:
    def __init__(self, tree: "aegis.PublicTree", ctx: "aegis.SubgameContext", device="cpu", dtype=torch.float64):
        self.tree, self.ctx, self.dev, self.dt = tree, ctx, torch.device(device), dtype
        nodes = tree.nodes
        assert len(tree.boards) == 1 and len(tree.boards[0]) == 5, "river trees only"
        board = tree.boards[0]
        hs = [ctx.hands(0), ctx.hands(1)]
        self.n = [h.size() for h in hs]
        cards = [h.cards for h in hs]
        vals = aegis.board_hand_values(board)
        s = [np.array([vals[c] for c in h.combos], dtype=np.int64) for h in hs]
        # Compatibility and win/lose matrices for player 0 (player 1 uses transposes).
        c0, c1 = cards
        compat = np.ones((self.n[0], self.n[1]))
        for a in range(2):
            for b in range(2):
                compat *= (c0[:, a][:, None] != c1[:, b][None, :])
        win = np.sign(s[0][:, None] - s[1][None, :]) * compat
        t = lambda x: torch.tensor(x, device=self.dev, dtype=self.dt)  # noqa: E731
        self.K = [t(compat), t(compat.T)]
        self.W = [t(win), t(-win.T)]
        self.prior = [t(h.root_weight) for h in hs]
        self.Z = float(ctx.normalizer)
        self.nodes = nodes
        self.dec = [i for i in range(tree.num_nodes) if nodes[i].type == aegis.PNodeType.DECISION]
        self.R = {i: torch.zeros((nodes[i].num_children, self.n[nodes[i].player]), device=self.dev, dtype=self.dt)
                  for i in self.dec}
        self.S = {i: torch.zeros_like(self.R[i]) for i in self.dec}
        self.order = list(range(tree.num_nodes))
        self.show = [i for i in range(tree.num_nodes) if nodes[i].type == aegis.PNodeType.SHOWDOWN]
        self.fold = [i for i in range(tree.num_nodes) if nodes[i].type == aegis.PNodeType.FOLD]
        self.t = 0

    def _sigma(self, i):
        R = self.R[i].clamp_min(0)
        tot = R.sum(0, keepdim=True)
        A = R.shape[0]
        return torch.where(tot > 0, R / tot.clamp_min(1e-300), torch.full_like(R, 1.0 / A))

    def iterate(self, n=1, alpha=1.5, beta=0.0, gamma=2.0):
        nodes = self.nodes
        for _ in range(n):
            self.t += 1
            t = self.t
            dp, dn = t ** alpha / (t ** alpha + 1), t ** beta / (t ** beta + 1)
            w = float(t) ** gamma
            for p in (0, 1):
                o = 1 - p
                sig = {i: self._sigma(i) for i in self.dec}
                reach = [[None] * len(nodes), [None] * len(nodes)]
                reach[0][0], reach[1][0] = self.prior[0], self.prior[1]
                for i in self.order:  # parents precede children
                    nd = nodes[i]
                    for k in range(nd.num_children):
                        c = nd.first_child + k
                        for q in (0, 1):
                            reach[q][c] = reach[q][i] * sig[i][k] if (nd.type == aegis.PNodeType.DECISION and nd.player == q) else reach[q][i]
                cfv = [None] * len(nodes)
                if self.show:
                    Ro = torch.stack([reach[o][i] for i in self.show], 1)
                    V = self.W[p] @ Ro
                    for j, i in enumerate(self.show):
                        cfv[i] = nodes[i].commit[0] * V[:, j]
                if self.fold:
                    Ro = torch.stack([reach[o][i] for i in self.fold], 1)
                    V = self.K[p] @ Ro
                    for j, i in enumerate(self.fold):
                        f = nodes[i].player
                        u = -nodes[i].commit[p] if f == p else nodes[i].commit[f]
                        cfv[i] = u * V[:, j]
                for i in reversed(self.order):
                    nd = nodes[i]
                    if nd.type != aegis.PNodeType.DECISION:
                        continue
                    kids = torch.stack([cfv[nd.first_child + k] for k in range(nd.num_children)])
                    if nd.player != p:
                        cfv[i] = kids.sum(0)
                        continue
                    s = sig[i]
                    v = (s * kids).sum(0)
                    cfv[i] = v
                    R = self.R[i] + (kids - v[None, :])
                    self.R[i] = torch.where(R >= 0, R * dp, R * dn)
                    self.S[i] += w * reach[p][i][None, :] * s

    def average_policy(self) -> "aegis.VectorPolicy":
        pol = self.ctx.uniform_policy()
        for i in self.dec:
            S = self.S[i].cpu().double().numpy()
            tot = S.sum(0, keepdims=True)
            avg = np.where(tot > 0, S / np.maximum(tot, 1e-300), 1.0 / S.shape[0])
            pol.set(self.nodes[i].decision_index, avg.reshape(-1))
        return pol

    def sync(self):
        if self.dev.type == "mps":
            torch.mps.synchronize()
