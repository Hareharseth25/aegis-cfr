"""High-level API: solve a heads-up turn/river spot from a JSON-style request.

Request (all amounts in big blinds):

    {
      "board": ["As", "Kh", "8c", "4d", "2s"],
      "pot": 100,
      "effective_stack": 150,
      "player_to_act": 0,
      "actions": {"bet_pot_fractions": [0.33, 0.75, 1.5], "raise_pot_fractions": [1.0],
                  "allow_all_in": true, "max_raises": 3},
      "range_p0": "<range notation> | {combo: weight} | [1326 weights]",
      "range_p1": ...,
      "solver": {"algorithm": "dcfr", "iterations": 1000, "threads": 1},
      "hero": {"player": 1, "hand": "AdJd", "path": ["Bet 75"]}   # path optional
    }

Ranges are real numerical weights over the 1,326 combinations (never verbal
descriptions); combos that intersect the board are removed.

The measured exploitability belongs to the constructed subgame (these ranges,
this betting abstraction), not to unrestricted No-Limit Hold'em.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import aegis_core as ac

ALGORITHMS = {
    "cfr": lambda: ac.CFRConfig.vanilla(True),
    "cfr+": ac.CFRConfig.cfr_plus,
    "cfr_plus": ac.CFRConfig.cfr_plus,
    "dcfr": ac.CFRConfig.dcfr,
    "lcfr": ac.CFRConfig.linear,
    "linear": ac.CFRConfig.linear,
}

_COMBO_CARDS = ac.combo_cards()
_COMBO_CLASS = ac.combo_hand_class()
_COMBO_NAMES = [ac.combo_to_string(i) for i in range(1326)]


def combo_name(i: int) -> str:
    return _COMBO_NAMES[i]


def _int_in(value, lo: int, hi: int, name: str) -> int:
    """Parse an integer and check lo <= value <= hi (ValueError otherwise)."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if isinstance(value, float) and not float(value).is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {v}")
    return v


def _positive_number(value, name: str, allow_zero: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'>= 0' if allow_zero else '> 0'}, got {value!r}")
    return v


MAX_ITERATIONS = 200_000


def parse_range(spec: Any, board: list[int]) -> "ac.Range":
    """Range from notation string, {combo: weight} mapping or 1326-vector."""
    if isinstance(spec, ac.Range):
        r = spec
    elif isinstance(spec, str):
        r = ac.Range.parse(spec)
    elif isinstance(spec, dict):
        r = ac.Range()
        for k, w in spec.items():
            w = float(w)
            if w < 0 or not math.isfinite(w):
                raise ValueError(f"invalid weight for {k}: {w}")
            r[ac.parse_combo(k)] = w
    else:
        arr = np.asarray(spec, dtype=float)
        if arr.shape != (1326,):
            raise ValueError("a numeric range must have exactly 1326 weights")
        r = ac.Range.from_array(arr)
    r.remove(board)
    if r.total() <= 0:
        raise ValueError("range is empty after removing board cards")
    return r


def street_sizes(spec: dict | None) -> "ac.StreetSizes":
    spec = spec or {}
    s = ac.StreetSizes()
    s.bet_fractions = [_positive_number(x, "bet size") for x in spec.get("bet_pot_fractions", [0.33, 0.75, 1.5])]
    s.raise_fractions = [_positive_number(x, "raise size") for x in spec.get("raise_pot_fractions", [1.0])]
    s.allow_all_in = bool(spec.get("allow_all_in", True))
    s.max_raises = _int_in(spec.get("max_raises", 3), 0, 6, "max_raises")
    return s


def tree_config(req: dict) -> "ac.TreeConfig":
    board = [ac.parse_card(c) for c in req["board"]] if isinstance(req["board"], list) else ac.parse_cards(req["board"])
    if len(board) not in (4, 5):
        raise ValueError("board must have 4 (turn) or 5 (river) cards")
    c = ac.TreeConfig()
    c.board = board
    c.pot = _positive_number(req.get("pot", 100), "pot")
    c.effective_stack = _positive_number(req.get("effective_stack", 150), "effective_stack", allow_zero=True)
    c.first_player = _int_in(req.get("player_to_act", 0), 0, 1, "player_to_act")
    river = street_sizes(req.get("actions"))
    turn = street_sizes(req.get("turn_actions", req.get("actions")))
    c.set_sizes(turn, river)
    # Optional per-player overrides: "actions_p0"/"actions_p1" (river).
    for p in (0, 1):
        key = f"actions_p{p}"
        if key in req:
            c.set_street_sizes(1, p, street_sizes(req[key]))
    return c


@dataclass
class SolvedSpot:
    request: dict
    tree: "ac.PublicTree"
    ctx: "ac.SubgameContext"
    solver: "ac.VectorCFR"
    policy: "ac.VectorPolicy"
    convergence: list[dict] = field(default_factory=list)
    regret_trace: list[dict] = field(default_factory=list)
    solve_seconds: float = 0.0
    exploitability: dict = field(default_factory=dict)
    total_seconds: float = 0.0

    # ----------------------------------------------------------- navigation
    def node_by_path(self, path: list[str]) -> int:
        node = 0
        for label in path or []:
            nxt = self.tree.child_by_label(node, label)
            if nxt < 0:
                options = [self.tree.node(self.tree.node(node).first_child + k).action.to_string()
                           for k in range(self.tree.node(node).num_children)]
                raise ValueError(f"no action '{label}' at {self.tree.path_string(node)}; options: {options}")
            node = nxt
        return node

    def check_node(self, node) -> int:
        return _int_in(node, 0, self.tree.num_nodes - 1, "node")

    def children(self, node: int) -> list[int]:
        nd = self.tree.node(self.check_node(node))
        return list(range(nd.first_child, nd.first_child + nd.num_children))

    # -------------------------------------------------------------- queries
    def node_strategy(self, node: int) -> dict:
        """Per-hand action probabilities of the acting player at `node`."""
        nd = self.tree.node(self.check_node(node))
        if nd.type != ac.PNodeType.DECISION:
            raise ValueError("not a decision node")
        p = nd.player
        hs = self.ctx.hands(p)
        n = hs.size()
        strat = self.policy.get(nd.decision_index).reshape(nd.num_children, n)
        reach = ac.reach_at(self.ctx, self.policy, node)[p]
        actions = [self.tree.node(c).action.to_string() for c in self.children(node)]
        total = reach.sum()
        freq = (strat * reach[None, :]).sum(axis=1) / total if total > 0 else strat.mean(axis=1)
        return {
            "node": node,
            "player": p,
            "path": self.tree.path_string(node),
            "actions": actions,
            "combos": [combo_name(c) for c in hs.combos],
            "strategy": strat.T.tolist(),  # [hand][action]
            "reach": reach.tolist(),
            "aggregate_frequency": freq.tolist(),
            "class_grid": self._class_grid(p, strat, reach),
        }

    def _class_grid(self, p: int, strat: np.ndarray, reach: np.ndarray) -> list:
        """13x13 hand-class grid: reach-weighted action mix per class."""
        hs = self.ctx.hands(p)
        A = strat.shape[0]
        mix = np.zeros((169, A))
        mass = np.zeros(169)
        for i, c in enumerate(hs.combos):
            k = _COMBO_CLASS[c]
            mix[k] += reach[i] * strat[:, i]
            mass[k] += reach[i]
        grid = []
        for k in range(169):
            grid.append({"name": ac.hand_class_name(k), "mass": float(mass[k]),
                         "mix": (mix[k] / mass[k]).tolist() if mass[k] > 0 else None})
        return grid

    def hand_values(self, node: int, player: int, best_response: bool = False) -> dict:
        """Per-hand EV (chips) of `player` at `node`: cfv / compatible opponent mass."""
        node = self.check_node(node)
        player = _int_in(player, 0, 1, "player")
        r0, r1 = ac.reach_at(self.ctx, self.policy, node)
        cfv = self.ctx.subtree_values(self.policy, node, r0, r1, player if best_response else -1)[player]
        opp = (r1, r0)[player]
        mass = self._compatible_mass(player, opp, node)
        ev = np.where(mass > 1e-12, cfv / np.maximum(mass, 1e-300), np.nan)
        return {"combos": [combo_name(c) for c in self.ctx.hands(player).combos], "ev": ev, "cfv": cfv, "mass": mass}

    def _compatible_mass(self, player: int, opp_reach: np.ndarray, node: int) -> np.ndarray:
        hs, ho = self.ctx.hands(player), self.ctx.hands(1 - player)
        oc = ho.cards
        total = opp_reach.sum()
        card_sum = np.zeros(52)
        np.add.at(card_sum, oc[:, 0], opp_reach)
        np.add.at(card_sum, oc[:, 1], opp_reach)
        pc = hs.cards
        same = np.zeros(hs.size())
        index = {c: j for j, c in enumerate(ho.combos)}
        for i, c in enumerate(hs.combos):
            j = index.get(c)
            if j is not None:
                same[i] = opp_reach[j]
        board_mask = set(self.tree.board(self.tree.node(node).board_id))
        mass = total - card_sum[pc[:, 0]] - card_sum[pc[:, 1]] + same
        dead = np.array([(a in board_mask) or (b in board_mask) for a, b in pc])
        mass[dead] = 0.0
        return np.maximum(mass, 0.0)

    def action_values(self, node: int) -> dict:
        """EV of each action per hand for the acting player (range-by-range EV)."""
        nd = self.tree.node(self.check_node(node))
        if nd.type != ac.PNodeType.DECISION:
            raise ValueError("not a decision node")
        p = nd.player
        r = ac.reach_at(self.ctx, self.policy, node)
        mass = self._compatible_mass(p, r[1 - p], node)
        out = []
        for c in self.children(node):
            rc = ac.reach_at(self.ctx, self.policy, c)
            # Counterfactual values do not depend on p's own reach: use p's reach at
            # the parent so hands that never take the action still get a value.
            reach = [rc[0], rc[1]]
            reach[p] = r[p]
            cfv = self.ctx.subtree_values(self.policy, c, reach[0], reach[1], -1)[p]
            out.append(np.where(mass > 1e-12, cfv / np.maximum(mass, 1e-300), np.nan))
        return {"actions": [self.tree.node(c).action.to_string() for c in self.children(node)],
                "combos": [combo_name(c) for c in self.ctx.hands(p).combos],
                "ev": np.array(out).T}

    def hero_view(self, player: int, hand: str, path: list[str]) -> dict:
        player = _int_in(player, 0, 1, "player")
        node = self.node_by_path(path)
        nd = self.tree.node(node)
        combo = ac.parse_combo(hand)
        hand = combo_name(combo)  # canonical spelling (high card first), e.g. "JdAd" -> "AdJd"
        hs = self.ctx.hands(player)
        if combo not in hs.combos:
            raise ValueError(f"{hand} is not in player {player}'s range on this board")
        i = hs.combos.index(combo)
        # EVs are reported "from here": chips collected from the pot minus chips
        # still to be put in. Chips already committed are sunk, so folding is
        # worth 0 at every node. (Solver utilities are zero-sum relative to the
        # root, so we add back what the player has committed at this node.)
        sunk = float(nd.commit[player])
        view: dict[str, Any] = {"node": node, "path": self.tree.path_string(node), "hand": hand, "player": player,
                                "pot": self.tree.pot(node), "to_act": nd.player, "committed": sunk}
        if nd.type == ac.PNodeType.DECISION and nd.player == player:
            st = self.node_strategy(node)
            av = self.action_values(node)
            view["actions"] = st["actions"]
            view["strategy"] = st["strategy"][i]
            view["action_ev"] = [None if not np.isfinite(x) else float(x) for x in av["ev"][i]]
            view["action_ev_from_here"] = [None if x is None else x + sunk for x in view["action_ev"]]
            view["range_frequency"] = st["aggregate_frequency"]
        hv = self.hand_values(node, player)
        view["ev"] = None if not np.isfinite(hv["ev"][i]) else float(hv["ev"][i])
        view["ev_from_here"] = None if view["ev"] is None else view["ev"] + sunk
        # Blockers: how the hero hand reshapes the opponent's range at this node.
        r = ac.reach_at(self.ctx, self.policy, node)[1 - player]
        ho = self.ctx.hands(1 - player)
        hero_cards = set(_COMBO_CARDS[combo].tolist())
        blocked = sum(r[j] for j, c in enumerate(ho.combos) if set(_COMBO_CARDS[c].tolist()) & hero_cards)
        view["opponent_range_blocked_fraction"] = float(blocked / r.sum()) if r.sum() > 0 else 0.0
        # Regret trajectory at this node for the hero hand (from the solve).
        view["regret_trace"] = [
            {"iteration": t["iteration"], "avg_positive_regret": t["per_hand"].get(hand)}
            for t in self.regret_trace if "per_hand" in t and t.get("node") == node
        ]
        return view

    def best_response_audit(self, player: int, top: int = 15) -> dict:
        """Where does the opponent's best response gain the most against `player`?"""
        player = _int_in(player, 0, 1, "player")
        top = _int_in(top, 1, 1326, "top")
        opp = 1 - player
        br_value, br_cfv = self.ctx.best_response(self.policy, opp)
        ev, cfv0, cfv1 = self.ctx.evaluate(self.policy)
        cfv = (cfv0, cfv1)[opp]
        hs = self.ctx.hands(opp)
        gain = (br_cfv - cfv) * hs.root_weight / self.ctx.normalizer
        order = np.argsort(-gain)[:top]
        return {"exploiter": opp, "br_value": br_value, "policy_value": ev[opp],
                "gain_total": float(gain.sum()),
                "top_hands": [{"combo": combo_name(hs.combos[i]), "gain": float(gain[i])} for i in order]}

    # -------------------------------------------------------------- export
    def summary(self, include_tree: bool = True) -> dict:
        out: dict[str, Any] = {
            "board": [ac.card_to_string(c) for c in self.tree.board(0)],
            "pot": self.tree.config().pot,
            "effective_stack": self.tree.config().effective_stack,
            "exploitability": self.exploitability,
            "convergence": self.convergence,
            "solve_seconds": self.solve_seconds,      # CFR iterations only
            "total_seconds": self.total_seconds,      # tree build + solve + exploitability checks
            "num_nodes": self.tree.num_nodes,
            "hands": [self.ctx.n(0), self.ctx.n(1)],
            "memory_mb": self.solver.memory_bytes / 1e6,
            "note": "Exploitability is measured exactly in the constructed subgame (these ranges and bet sizes), "
                    "not in unrestricted No-Limit Hold'em.",
        }
        if include_tree:
            ids = initial_tree_ids(self.tree)
            out["tree"] = tree_json(self.tree, ids)
            out["tree_complete"] = len(ids) == self.tree.num_nodes
        return out


TREE_INLINE_LIMIT = 6000   # trees up to this size are sent whole
TREE_BFS_PREFIX = 2000     # larger trees: this many nodes in breadth-first order


def initial_tree_ids(tree: "ac.PublicTree") -> list[int]:
    """Node ids sent with a solve. Small trees are sent whole; large ones as a
    breadth-first prefix (every node's ancestors are included). Any child id
    not in the set can be fetched later with tree_json(tree, [ids])."""
    n = tree.num_nodes
    if n <= TREE_INLINE_LIMIT:
        return list(range(n))
    nodes = tree.nodes  # one bulk copy for the whole walk
    out, queue, i = [0], [0], 0
    while i < len(queue) and len(out) < TREE_BFS_PREFIX:
        nd = nodes[queue[i]]
        i += 1
        for c in range(nd.first_child, nd.first_child + nd.num_children):
            if len(out) >= TREE_BFS_PREFIX:
                break
            out.append(c)
            queue.append(c)
    return out


def tree_json(tree: "ac.PublicTree", ids=None) -> list[dict]:
    """Serialise public-tree nodes. `children` always lists every child id;
    ids that were not sent can be requested separately (lazy loading)."""
    n = tree.num_nodes
    if ids is None:
        ids = range(n)
    ids = [_int_in(i, 0, n - 1, "node") for i in ids]
    nodes = tree.nodes if len(ids) > 64 else None  # bulk copy once for big batches
    boards: dict[int, list[str]] = {}
    out = []
    for i in ids:
        nd = nodes[i] if nodes is not None else tree.node(i)
        if nd.board_id not in boards:
            boards[nd.board_id] = [ac.card_to_string(c) for c in tree.board(nd.board_id)]
        out.append({
            "id": i, "type": str(nd.type).split(".")[-1], "player": nd.player, "parent": nd.parent,
            "children": list(range(nd.first_child, nd.first_child + nd.num_children)) if nd.num_children else [],
            "action": nd.action.to_string() if i > 0 else "root", "pot": nd.commit[0] + nd.commit[1],
            "commit": list(nd.commit), "street_bet": list(nd.street_bet),
            "street": nd.street, "board": boards[nd.board_id],
        })
    return out


def default_hero_path(tree: "ac.PublicTree", player: int) -> list[str]:
    """The hero's first decision: the root if the hero acts first, otherwise
    facing the smallest opening bet (or after a check if nobody bets)."""
    root = tree.node(0)
    if root.player == player:
        return []
    kids = [tree.node(c) for c in range(root.first_child, root.first_child + root.num_children)]
    for want in (ac.ActionKind.BET, ac.ActionKind.ALLIN, ac.ActionKind.CHECK):
        for k in kids:
            if k.action.kind == want and k.type == ac.PNodeType.DECISION and k.player == player:
                return [k.action.to_string()]
    return []


def _checkpoints(total: int) -> list[int]:
    pts, x = [], 1
    while x < total:
        pts.append(x)
        x = max(x + 1, int(x * 1.6))
    pts.append(total)
    return sorted(set(pts))


def solve_spot(req: dict, progress=None) -> SolvedSpot:
    t_start = time.perf_counter()
    cfg = tree_config(req)
    tree = ac.PublicTree.build(cfg)
    board = list(cfg.board)
    r0 = parse_range(req.get("range_p0", "random"), board)
    r1 = parse_range(req.get("range_p1", "random"), board)
    sol = req.get("solver", {})
    algo = sol.get("algorithm", "dcfr").lower()
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm '{algo}' (choose from {sorted(ALGORITHMS)})")
    iterations = _int_in(sol.get("iterations", 1000), 1, MAX_ITERATIONS, "iterations")
    threads = _int_in(sol.get("threads", 1), 1, 64, "threads")
    ctx = ac.SubgameContext(tree, r0, r1, threads)
    solver = ac.VectorCFR(ctx, ac.VectorCFRConfig(ALGORITHMS[algo]()))
    convergence, regret_trace = [], []
    hero = req.get("hero")
    hero_node = None
    if hero:
        hero_player = _int_in(hero.get("player", 1), 0, 1, "hero player")
        if "path" not in hero:
            hero["path"] = default_hero_path(tree, hero_player)
        try:
            tmp = SolvedSpot(req, tree, ctx, solver, ctx.uniform_policy())
            hero_node = tmp.node_by_path(hero.get("path", []))
        except ValueError:
            hero_node = None
    t0 = time.perf_counter()
    solve_time = 0.0
    done = 0
    for cp in _checkpoints(iterations):
        ts = time.perf_counter()
        solver.iterate(cp - done)
        solve_time += time.perf_counter() - ts
        done = cp
        e = ctx.exploitability(solver.average_policy())
        convergence.append({"iteration": cp, "seconds": solve_time, "exploitability_pct_pot": e["exploitability_pct_pot"],
                            "nash_conv": e["nash_conv"]})
        if hero_node is not None and tree.node(hero_node).type == ac.PNodeType.DECISION:
            nd = tree.node(hero_node)
            n = ctx.n(nd.player)
            R = solver.regrets(hero_node).reshape(nd.num_children, n)
            avg_pos = np.maximum(R, 0).sum(axis=0) / max(cp, 1)
            per_hand = {combo_name(c): float(avg_pos[i]) for i, c in enumerate(ctx.hands(nd.player).combos)}
            regret_trace.append({"iteration": cp, "node": hero_node, "per_hand": per_hand,
                                 "mean": float(avg_pos.mean())})
        if progress:
            progress(cp, iterations, convergence[-1])
    policy = solver.average_policy()
    spot = SolvedSpot(req, tree, ctx, solver, policy, convergence, regret_trace, time.perf_counter() - t0)
    spot.exploitability = ctx.exploitability(policy)
    spot.solve_seconds = solve_time
    spot.total_seconds = time.perf_counter() - t_start
    return spot


def solve_request(req: dict) -> dict:
    """Solve and return a JSON-serialisable result (used by the CLI)."""
    spot = solve_spot(req)
    out = spot.summary(include_tree=True)
    out["root_strategy"] = _json_safe(spot.node_strategy(0))
    if "hero" in req:
        h = req["hero"]
        out["hero"] = _json_safe(spot.hero_view(h.get("player", 1), h["hand"], h.get("path", [])))
    out["best_response_audit"] = [spot.best_response_audit(0), spot.best_response_audit(1)]
    return _json_safe(out)


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x
