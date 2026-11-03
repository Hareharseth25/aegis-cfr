"""End-to-end tests of the spot API, CLI and HTTP server."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import aegis
from aegis.__main__ import DEMO, main as cli_main
from aegis.server import Handler
from aegis.spot import parse_range, solve_request, solve_spot


def small_request(**kw):
    r = dict(DEMO)
    r.update({"solver": {"algorithm": "dcfr", "iterations": 300}})
    r.update(kw)
    return r


def test_solve_request_is_json_safe_and_converged():
    res = solve_request(small_request())
    json.dumps(res)  # serialisable
    assert res["exploitability"]["exploitability_pct_pot"] < 0.1
    h = res["hero"]
    assert abs(sum(h["strategy"]) - 1) < 1e-9
    assert len(h["actions"]) == len(h["action_ev"])
    assert res["convergence"][-1]["exploitability_pct_pot"] <= res["convergence"][0]["exploitability_pct_pot"]


def test_range_formats_are_equivalent():
    board = aegis.parse_cards("AsKh8c4d2s")
    a = parse_range("QQ,AKs", board).weights
    d = {aegis.combo_to_string(i): 1.0 for i in range(1326) if a[i] > 0}
    b = parse_range(d, board).weights
    c = parse_range(a.tolist(), board).weights
    assert np.allclose(a, b) and np.allclose(a, c)
    with pytest.raises(ValueError):
        parse_range([1.0] * 10, board)


def test_hero_view_action_values_consistent_with_strategy():
    spot = solve_spot(small_request(solver={"algorithm": "dcfr", "iterations": 1500}))
    hv = spot.hero_view(1, "AdJd", ["Bet 75"])
    evs = np.array([x for x in hv["action_ev"]])
    strat = np.array(hv["strategy"])
    # Actions played with real probability are (near) EV-maximising at equilibrium.
    best = evs.max()
    for p, ev in zip(strat, evs):
        if p > 0.05:
            assert ev > best - 0.02 * spot.tree.config().pot
    # EV of the hand equals the strategy-weighted action EV.
    assert abs(hv["ev"] - float(strat @ evs)) < 1e-6 * spot.tree.config().pot + 1e-9


def test_turn_spot_solves():
    req = small_request(board=["As", "Kh", "8c", "4d"], range_p0="AA,KK,AK,AQ,88,QJs", range_p1="AQ,KQ,QQ,99,T9s",
                        actions={"bet_pot_fractions": [0.75], "raise_pot_fractions": [], "allow_all_in": True, "max_raises": 1},
                        solver={"algorithm": "dcfr", "iterations": 60})
    req.pop("hero")
    res = solve_request(req)
    assert res["num_nodes"] > 48
    assert res["exploitability"]["exploitability_pct_pot"] < 2.0


def test_cli_demo(capsys):
    assert cli_main(["evaluate", "As", "Kh", "8c", "4d", "2s", "Ad", "Jd"]) == 0
    assert "Pair, Aces" in capsys.readouterr().out


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_server_endpoints(server):
    assert call(server, "/api/health")["ok"]
    res = call(server, "/api/solve", {"solver": {"algorithm": "dcfr", "iterations": 200}})
    sid = res["id"]
    node = call(server, f"/api/spot/{sid}/node/0")
    assert abs(sum(node["aggregate_frequency"]) - 1) < 1e-6
    assert len(node["class_grid"]) == 169
    audit = call(server, f"/api/spot/{sid}/audit?player=1")
    assert audit["gain_total"] >= -1e-9
    hero = call(server, f"/api/spot/{sid}/hero?player=1&hand=AdJd&path=Bet%2075")
    assert len(hero["strategy"]) == 3
    rng = call(server, "/api/range", {"range": "TT+,AK", "villain": "QQ+", "board": "AhKd7c", "hero": "AsQs"})
    assert 0 < rng["range_equity"] < 1
    conv = call(server, "/api/convergence", {"game": "kuhn", "algorithms": ["cfr+"], "iterations": 50})
    assert conv["series"][0]["points"][-1]["exploitability"] < 0.05
    micro = call(server, "/api/microscope?game=kuhn")
    assert len(micro["infosets"]) == 12
    w = call(server, "/api/microscope/whatif", {"game": "kuhn", "infoset": 0, "strategy": [0.0, 1.0]})
    assert w["after"]["nash_conv"] > w["before"]["nash_conv"]
    ad = call(server, "/api/adaptive", {"game": "kuhn", "iterations": 500})
    assert set(ad["methods"]) == {"external", "adaptive"}


def test_range_strategy_and_blockers(server):
    body = {"range": "AA,KK,AK,AQs,KQs,QJs,JTs,T9s,A5s", "villain": "QQ+,AK,AQ,KQ", "board": "AhKd7c4s2h",
            "iterations": 300}
    st = call(server, "/api/range/strategy", body)
    root = st["nodes"]["root"]
    assert abs(sum(root["aggregate_frequency"]) - 1) < 1e-6
    assert "check_then_facing_bet" in st["nodes"]
    rng = call(server, "/api/range", {"range": body["range"], "villain": body["villain"], "board": body["board"]})
    # AsKs blocks villain's AA/KK/AK/AQ/KQ combos more than JsTs does.
    assert rng["combo_blocked"]["AsKs"] > rng["combo_blocked"]["JsTs"]
