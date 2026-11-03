"""AEGIS Solver Lab server: JSON API over the C++ engine + static frontend.

Run:  python -m aegis.server [--port 8765]
Then open http://localhost:8765 (frontend built into frontend/dist).

Only the Python standard library is used for HTTP. Solves run in the request
thread (the C++ core releases the GIL), solved spots are cached by id.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import aegis_core as ac
from .__main__ import DEMO
from .spot import _int_in, _json_safe, solve_spot, tree_json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIST = os.path.join(ROOT, "frontend", "dist")
RESULTS = os.path.join(ROOT, "results")

_SPOTS: dict[str, object] = {}
_LOCK = threading.Lock()
_TABULAR: dict[str, dict] = {}


# ------------------------------------------------------------------ helpers
TABULAR_GAMES = ("kuhn", "leduc", "leduc4", "leduc5", "leduc6", "coin_toss")


def _tabular_game(name: str):
    if name not in TABULAR_GAMES:
        raise ValueError(f"unknown game '{name}' (choose from {', '.join(TABULAR_GAMES)})")
    if name == "kuhn":
        return ac.KuhnGame()
    if name == "leduc":
        return ac.LeducGame()
    if name == "coin_toss":
        return ac.CoinTossGame()
    c = ac.LeducConfig()
    c.num_ranks = int(name.replace("leduc", ""))
    return ac.LeducGame(c)


def _tabular(name: str) -> dict:
    """Solved tabular game cache used by the microscope."""
    with _LOCK:
        if name in _TABULAR:
            return _TABULAR[name]
    game = _tabular_game(name)
    tree = ac.GameTree.build(game)
    cfr = ac.TabularCFR(tree, ac.CFRConfig.cfr_plus())
    cfr.iterate(2000 if tree.num_nodes < 20000 else 300)
    entry = {"game": game, "tree": tree, "policy": cfr.average_policy()}
    with _LOCK:
        _TABULAR[name] = entry
    return entry


_ALGOS = {"cfr": lambda: ac.CFRConfig.vanilla(True), "cfr_sim": lambda: ac.CFRConfig.vanilla(False),
          "cfr+": ac.CFRConfig.cfr_plus, "dcfr": ac.CFRConfig.dcfr, "lcfr": ac.CFRConfig.linear}
# The vector river solver uses alternating updates and full traversals only,
# so it supports a subset of what the tabular games support.
RIVER_ALGOS = ("cfr", "cfr+", "dcfr", "lcfr")
TABULAR_ALGOS = ("cfr", "cfr_sim", "cfr+", "dcfr", "lcfr", "es", "os")


def _algo(name: str):
    if name not in _ALGOS:
        raise ValueError(f"unknown algorithm '{name}'")
    return _ALGOS[name]()


def _checkpoints(total: int):
    pts, x = [], 1
    while x < total:
        pts.append(x)
        x = max(x + 1, int(x * 1.5))
    pts.append(total)
    return sorted(set(pts))


# ------------------------------------------------------------------ handlers
def api_solve(body: dict) -> dict:
    req = dict(DEMO)
    req.update(body or {})
    t = time.perf_counter()
    spot = solve_spot(req)
    sid = uuid.uuid4().hex[:10]
    with _LOCK:
        _SPOTS[sid] = spot
        while len(_SPOTS) > 12:
            _SPOTS.pop(next(iter(_SPOTS)))
    out = spot.summary(include_tree=True)
    out["id"] = sid
    out["root"] = spot.node_strategy(0)
    out["wall_seconds"] = time.perf_counter() - t
    out["regret_trace"] = [{"iteration": r["iteration"], "mean": r["mean"], "node": r["node"]} for r in spot.regret_trace]
    if req.get("hero"):
        h = req["hero"]
        try:
            out["hero"] = spot.hero_view(h.get("player", 1), h["hand"], h.get("path", []))
        except ValueError as e:
            out["hero_error"] = str(e)
    return out


def _spot(sid: str):
    with _LOCK:
        if sid not in _SPOTS:
            raise KeyError("unknown or expired spot id; solve again")
        return _SPOTS[sid]


def api_node(sid: str, node) -> dict:
    spot = _spot(sid)
    node = spot.check_node(node)
    nd = spot.tree.node(node)
    out = {"node": node, "path": spot.tree.path_string(node), "type": str(nd.type).split(".")[-1],
           "pot": spot.tree.pot(node), "children": spot.children(node),
           "child_actions": [spot.tree.node(c).action.to_string() for c in spot.children(node)]}
    if nd.type == ac.PNodeType.DECISION:
        out.update(spot.node_strategy(node))
        av = spot.action_values(node)
        out["action_ev"] = av["ev"]
        # EV from this decision: chips already committed are sunk, folding = 0.
        out["action_ev_from_here"] = av["ev"] + float(nd.commit[nd.player])
    return out


MAX_TREE_IDS = 500


def api_tree(sid: str, q: dict) -> dict:
    """Lazy tree loading: /api/spot/<id>/tree?ids=1,2,3 returns those nodes
    (and only those). Without ids, the root and its children."""
    spot = _spot(sid)
    raw = q.get("ids", [""])[0]
    if raw:
        parts = [x for x in raw.split(",") if x.strip()]
        if len(parts) > MAX_TREE_IDS:
            raise ValueError(f"at most {MAX_TREE_IDS} node ids per request")
        ids = [spot.check_node(x) for x in parts]
    else:
        ids = [0] + spot.children(0)
    return {"nodes": tree_json(spot.tree, ids), "num_nodes": spot.tree.num_nodes}


def api_hero(sid: str, q: dict) -> dict:
    spot = _spot(sid)
    path = [p for p in q.get("path", [""])[0].split("|") if p]
    if "hand" not in q:
        raise ValueError("missing 'hand' parameter")
    return spot.hero_view(q.get("player", ["1"])[0], q["hand"][0], path)


def api_audit(sid: str, q: dict) -> dict:
    spot = _spot(sid)
    return spot.best_response_audit(q.get("player", ["1"])[0], top=20)


def api_range(body: dict) -> dict:
    board = ac.parse_cards(body.get("board", "")) if body.get("board") else []
    r = ac.Range.parse(body.get("range", "random")).remove(board)
    out = {"class_frequency": r.class_frequency(board), "class_names": [ac.hand_class_name(k) for k in range(169)],
           "combos": int(r.num_nonzero()), "text": r.to_string()}
    hero = body.get("hero")
    if hero:
        known = ac.parse_cards(hero) + board
        out["blocked_fraction"] = r.blocked_fraction(ac.parse_cards(hero))
        cond = r.conditioned_on(known)
        out["conditioned_class_frequency"] = cond.class_frequency(known)
        if len(board) >= 3:
            eq = ac.equity_vs_range_exact(ac.parse_combo(hero), r, board)
            out["hero_equity"] = {"win": eq.win, "tie": eq.tie, "equity": eq.equity()}
    if body.get("villain"):
        # Blocker effect of every combo in the range on the opposing range.
        v = ac.Range.parse(body["villain"]).remove(board)
        vw = v.weights
        cards = ac.combo_cards()
        tot = vw.sum()
        card_mass = np.zeros(52)
        np.add.at(card_mass, cards[:, 0], vw)
        np.add.at(card_mass, cards[:, 1], vw)
        w = r.weights
        out["combo_blocked"] = {ac.combo_to_string(i): float((card_mass[cards[i, 0]] + card_mass[cards[i, 1]] - vw[i]) / tot)
                                for i in range(1326) if w[i] > 0 and tot > 0}
    if body.get("villain") and len(board) >= 3:
        v = ac.Range.parse(body["villain"]).remove(board)
        overall, combo_eq, _ = ac.equity_range_vs_range_exact(r, v, board)
        cls = ac.combo_hand_class()
        per_class = np.zeros(169)
        cnt = np.zeros(169)
        w = r.weights
        for i in range(1326):
            if w[i] > 0 and np.isfinite(combo_eq[i]):
                per_class[cls[i]] += w[i] * combo_eq[i]
                cnt[cls[i]] += w[i]
        out["range_equity"] = overall.equity()
        out["class_equity"] = np.where(cnt > 0, per_class / np.maximum(cnt, 1e-12), np.nan)
    return out


def api_range_strategy(body: dict) -> dict:
    """Equilibrium strategy of `range` (player 0, acts first) against `villain`
    on a 5-card board: per-class and per-combo frequencies at the root and
    when facing the first bet, with each combo's blocker effect."""
    board = ac.parse_cards(body.get("board", ""))
    if len(board) != 5:
        raise ValueError("strategy view needs a 5-card river board")
    req = {"board": [ac.card_to_string(c) for c in board], "pot": body.get("pot", 100),
           "effective_stack": body.get("stack", 150), "player_to_act": 0,
           "actions": {"bet_pot_fractions": body.get("bets", [0.5, 1.0]), "raise_pot_fractions": [1.0],
                       "allow_all_in": True, "max_raises": 2},
           "range_p0": body.get("range", "random"), "range_p1": body.get("villain", "random"),
           "solver": {"algorithm": "dcfr", "iterations": _int_in(body.get("iterations", 600), 1, 5000, "iterations")}}
    spot = solve_spot(req)
    out = {"exploitability_pct_pot": spot.exploitability["exploitability_pct_pot"], "nodes": {}}
    root = spot.node_strategy(0)
    out["nodes"]["root"] = root
    # Player 0 facing a bet after checking (the bluff-catching decision).
    check = spot.tree.child_by_label(0, "Check")
    if check >= 0:
        kids = spot.children(check)
        bet = next((c for c in kids if spot.tree.node(c).action.to_string().startswith(("Bet", "AllIn"))), None)
        if bet is not None and spot.tree.node(bet).type == ac.PNodeType.DECISION:
            out["nodes"]["check_then_facing_bet"] = spot.node_strategy(bet)
    return out


def api_convergence(body: dict) -> dict:
    game = body.get("game", "leduc")
    algos = body.get("algorithms", ["cfr", "cfr+", "dcfr", "lcfr"])
    if not isinstance(algos, list) or not algos or not all(isinstance(a, str) for a in algos):
        raise ValueError("algorithms must be a non-empty list of names")
    supported = RIVER_ALGOS if game == "river" else TABULAR_ALGOS
    bad = [a for a in algos if a not in supported]
    if bad:
        raise ValueError(f"{', '.join(bad)} not supported for {game} (supported: {', '.join(supported)})")
    if game != "river":
        _tabular_game(game)  # validates the name
    iters = _int_in(body.get("iterations", 500), 1, 5000 if game == "river" else 20000, "iterations")
    out = {"game": game, "series": [], "work_unit": "hand-node updates" if game == "river" else "nodes touched"}
    if game == "river":
        req = dict(DEMO)
        req.pop("hero", None)
        from .spot import parse_range, tree_config
        cfg = tree_config(req)
        tree = ac.PublicTree.build(cfg)
        ctx = ac.SubgameContext(tree, parse_range(req["range_p0"], list(cfg.board)), parse_range(req["range_p1"], list(cfg.board)))
        for a in algos:
            s = ac.VectorCFR(ctx, ac.VectorCFRConfig(_algo(a)))
            pts, done, secs = [], 0, 0.0
            for cp in _checkpoints(iters):
                t = time.perf_counter()
                s.iterate(cp - done)
                secs += time.perf_counter() - t
                done = cp
                e = ctx.exploitability(s.average_policy())
                pts.append({"iteration": cp, "seconds": secs, "nodes": s.hand_updates,
                            "exploitability": e["exploitability_pct_pot"]})
            out["series"].append({"algorithm": a, "points": pts, "unit": "% pot"})
        return out
    tree = ac.GameTree.build(_tabular_game(game))
    for a in algos:
        if a in ("es", "os"):
            c = ac.MCCFRConfig()
            c.scheme = ac.MCCFRScheme.EXTERNAL if a == "es" else ac.MCCFRScheme.OUTCOME
            m = ac.MCCFR(tree, c)
            pts, done, secs = [], 0, 0.0
            for cp in _checkpoints(iters * 50):
                t = time.perf_counter()
                m.iterate(cp - done)
                secs += time.perf_counter() - t
                done = cp
                pts.append({"iteration": cp, "seconds": secs, "nodes": m.nodes_touched,
                            "exploitability": ac.exploitability(tree, m.average_policy())})
            out["series"].append({"algorithm": a, "points": pts, "unit": "chips"})
            continue
        cfr = ac.TabularCFR(tree, _algo(a))
        pts, done, secs = [], 0, 0.0
        for cp in _checkpoints(iters):
            t = time.perf_counter()
            cfr.iterate(cp - done)
            secs += time.perf_counter() - t
            done = cp
            R = cfr.cumulative_regret
            pts.append({"iteration": cp, "seconds": secs, "nodes": cfr.nodes_touched,
                        "exploitability": ac.exploitability(tree, cfr.average_policy()),
                        "avg_positive_regret": float(np.maximum(R, 0).sum() / cp)})
        out["series"].append({"algorithm": a, "points": pts, "unit": "chips"})
    return out


def api_microscope(q: dict) -> dict:
    name = q.get("game", ["kuhn"])[0]
    e = _tabular(name)
    tree, pol = e["tree"], e["policy"]
    br = [ac.best_response(tree, pol, p) for p in (0, 1)]
    public = tree.public_state
    infos = []
    for I, info in enumerate(tree.infosets):
        p = info.player
        cav = br[p].cf_action_value
        infos.append({"id": I, "key": info.key, "player": p, "actions": list(info.action_names),
                      "public_state": tree.public_keys[int(public[info.nodes[0]])],
                      "strategy": pol.at(tree, I), "depth": info.depth, "num_histories": len(info.nodes),
                      "br_action_values": [float(cav[info.offset + a]) for a in range(info.num_actions)],
                      "br_best": int(br[p].best_action[I])})
    rep = ac.exploitability_report(tree, pol)
    return {"game": name, "nodes": tree.num_nodes, "infosets": infos, "nash_conv": rep.nash_conv,
            "values": rep.policy_value}


def api_whatif(body: dict) -> dict:
    """Change one information set's strategy and report the consequences."""
    name = body.get("game", "kuhn")
    e = _tabular(name)
    tree, pol = e["tree"], e["policy"]
    I = _int_in(body.get("infoset"), 0, tree.num_infosets - 1, "infoset")
    info = tree.infoset(I)
    try:
        probs = np.array(body.get("strategy"), dtype=float)
    except (TypeError, ValueError):
        raise ValueError("strategy must be a list of numbers") from None
    if probs.shape != (info.num_actions,):
        raise ValueError(f"strategy needs {info.num_actions} entries for this information set")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("strategy weights must be finite, non-negative and not all zero")
    probs = probs / probs.sum()
    new = ac.TabularPolicy(tree)
    p = pol.probs.copy()
    p[info.offset:info.offset + info.num_actions] = probs
    new.probs = p
    before = ac.exploitability_report(tree, pol)
    after = ac.exploitability_report(tree, new)
    opp = 1 - info.player
    br_before = ac.best_response(tree, pol, opp)
    br_after = ac.best_response(tree, new, opp)
    changed = [{"key": tree.infoset(J).key, "before": tree.infoset(J).action_names[br_before.best_action[J]],
                "after": tree.infoset(J).action_names[br_after.best_action[J]]}
               for J in range(tree.num_infosets) if tree.infoset(J).player == opp
               and br_before.best_action[J] != br_after.best_action[J]]
    return {"before": {"nash_conv": before.nash_conv, "value_p0": before.policy_value[0], "br_opp": br_before.value},
            "after": {"nash_conv": after.nash_conv, "value_p0": after.policy_value[0], "br_opp": br_after.value},
            "opponent_br_changes": changed}


def api_adaptive(body: dict) -> dict:
    name = body.get("game", "leduc")
    iters = _int_in(body.get("iterations", 20000), 1, 500_000, "iterations")
    lam = float(body.get("lambda", 0.3))
    if not 0.0 <= lam < 1.0:
        raise ValueError("lambda must be in [0, 1)")
    seed = _int_in(body.get("seed", 1), 0, 2**31 - 1, "seed")
    tree = ac.GameTree.build(_tabular_game(name))
    out = {"game": name, "methods": {}}
    for method, mode in (("external", ac.AllocationMode.EXTERNAL), ("adaptive", ac.AllocationMode.ADAPTIVE)):
        c = ac.MCCFRConfig()
        c.scheme = ac.MCCFRScheme.IMPORTANCE
        c.seed = seed
        m = ac.MCCFR(tree, c)
        pp = ac.PriorityParams()
        pp.lambda_ = lam
        s = ac.AdaptiveSampler(tree, m, mode, pp)
        m.set_sampling_policy(s)
        curve, done = [], 0
        for cp in _checkpoints(iters):
            m.iterate(cp - done)
            done = cp
            curve.append({"iteration": cp, "nodes": m.nodes_touched,
                          "exploitability": ac.exploitability(tree, m.average_policy())})
        visits = s.visits
        infos = []
        for I, info in enumerate(tree.infosets):
            infos.append({"key": info.key, "player": info.player, "depth": info.depth, "visits": int(visits[I]),
                          "score": s.score(I), "reach": s.reach_estimate(I), "uncertainty": s.uncertainty(I),
                          "sensitivity": s.sensitivity(I), "cost": s.cost(I)})
        out["methods"][method] = {"curve": curve, "infosets": infos}
    return out


def api_profiler() -> dict:
    path = os.path.join(RESULTS, "exp09_performance", "results.json")
    out = {"available": os.path.exists(path)}
    if out["available"]:
        out["results"] = json.load(open(path))
    hw = ac.HwCounters()
    out["hw_counters"] = {"available": hw.available, "backend": hw.backend, "reason": hw.reason}
    # Live micro-benchmark: river iteration speed on this machine right now.
    from .spot import tree_config
    cfg = tree_config(dict(DEMO))
    tree = ac.PublicTree.build(cfg)
    ctx = ac.SubgameContext(tree, ac.Range.full().remove(list(cfg.board)), ac.Range.full().remove(list(cfg.board)))
    s = ac.VectorCFR(ctx, ac.VectorCFRConfig(ac.CFRConfig.dcfr()))
    t = time.perf_counter()
    s.iterate(50)
    out["live_river_ms_per_iteration"] = 1e3 * (time.perf_counter() - t) / 50
    out["threads"] = ac.hardware_threads()
    return out


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_SAFE_FIGURE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]*\.png$")


class NotFound(Exception):
    pass


def _inside(base: str, path: str) -> bool:
    """True if `path` resolves to a location inside `base` (symlinks resolved)."""
    base_r, path_r = os.path.realpath(base), os.path.realpath(path)
    return os.path.commonpath([base_r, path_r]) == base_r


def _experiment_file(experiment: str, filename: str) -> str:
    """Path to a file of one experiment; only plain names, no traversal."""
    if not _SAFE_NAME.match(experiment or ""):
        raise NotFound("unknown experiment")
    folder = os.path.join(RESULTS, experiment)
    path = os.path.join(folder, filename)
    if not os.path.isdir(folder) or not _inside(folder, path) or not os.path.isfile(path):
        raise NotFound(f"{experiment}/{filename} not found")
    return path


def api_results(name: str) -> dict:
    try:
        path = _experiment_file(name, "results.json")
    except NotFound:
        raise NotFound(f"no results for {name}; run python/experiments/{name}.py") from None
    return json.load(open(path))


def api_figure(experiment: str, filename: str) -> bytes:
    if not _SAFE_FIGURE.match(filename or ""):
        raise NotFound("figure not found")
    return open(_experiment_file(experiment, filename), "rb").read()


def api_results_index() -> dict:
    items = []
    if os.path.isdir(RESULTS):
        for d in sorted(os.listdir(RESULTS)):
            p = os.path.join(RESULTS, d)
            if os.path.isdir(p):
                items.append({"name": d, "has_results": os.path.exists(os.path.join(p, "results.json")),
                              "figures": sorted(f for f in os.listdir(p) if f.endswith(".png"))})
    return {"experiments": items}


# --------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    server_version = "AEGIS/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        pass

    def _send(self, code: int, payload, ctype="application/json"):
        data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(_json_safe(payload)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _dispatch(self, method: str):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]
        try:
            if parts[:1] == ["api"]:
                r = parts[1:]
                if method == "POST" and r == ["solve"]:
                    return self._send(200, api_solve(self._body()))
                if method == "GET" and len(r) == 4 and r[0] == "spot" and r[2] == "node":
                    return self._send(200, api_node(r[1], r[3]))
                if method == "GET" and len(r) == 3 and r[0] == "spot" and r[2] == "hero":
                    return self._send(200, api_hero(r[1], q))
                if method == "GET" and len(r) == 3 and r[0] == "spot" and r[2] == "audit":
                    return self._send(200, api_audit(r[1], q))
                if method == "GET" and len(r) == 3 and r[0] == "spot" and r[2] == "tree":
                    return self._send(200, api_tree(r[1], q))
                if method == "POST" and r == ["range"]:
                    return self._send(200, api_range(self._body()))
                if method == "POST" and r == ["range", "strategy"]:
                    return self._send(200, api_range_strategy(self._body()))
                if method == "POST" and r == ["convergence"]:
                    return self._send(200, api_convergence(self._body()))
                if method == "GET" and r == ["microscope"]:
                    return self._send(200, api_microscope(q))
                if method == "POST" and r == ["microscope", "whatif"]:
                    return self._send(200, api_whatif(self._body()))
                if method == "POST" and r == ["adaptive"]:
                    return self._send(200, api_adaptive(self._body()))
                if method == "GET" and r == ["profiler"]:
                    return self._send(200, api_profiler())
                if method == "GET" and r == ["results"]:
                    return self._send(200, api_results_index())
                if method == "GET" and len(r) == 2 and r[0] == "results":
                    return self._send(200, api_results(r[1]))
                if method == "GET" and len(r) == 3 and r[0] == "figure":
                    return self._send(200, api_figure(r[1], r[2]), "image/png")
                if method == "GET" and r == ["health"]:
                    return self._send(200, {"ok": True})
                return self._send(404, {"error": f"no route {method} {url.path}"})
            return self._static(url.path)
        except NotFound as e:
            return self._send(404, {"error": str(e)})
        except (KeyError, ValueError, RuntimeError) as e:
            return self._send(400, {"error": str(e).strip("'\"")})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _static(self, path: str):
        if not os.path.isdir(DIST):
            return self._send(200, b"<h1>AEGIS server running</h1><p>Build the frontend: cd frontend && npm install "
                                   b"&& npm run build</p>", "text/html")
        rel = path.lstrip("/") or "index.html"
        full = os.path.join(DIST, rel)
        if ".." in rel.split("/") or not _inside(DIST, full) or not os.path.isfile(full):
            full = os.path.join(DIST, "index.html")
        ctype = {".html": "text/html", ".js": "application/javascript", ".css": "text/css", ".svg": "image/svg+xml",
                 ".png": "image/png", ".json": "application/json"}.get(os.path.splitext(full)[1], "application/octet-stream")
        return self._send(200, open(full, "rb").read(), ctype)

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")


def main(argv=None):
    ap = argparse.ArgumentParser(description="AEGIS Solver Lab server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AEGIS Solver Lab on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

