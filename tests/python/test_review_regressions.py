"""Regression tests for issues found in a code review (input validation,
EV baseline, large-tree export, path containment, unsupported options)."""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

import aegis
from aegis.__main__ import DEMO
from aegis.server import Handler
from aegis.spot import solve_spot, tree_json


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def raw(port, method, path, body=None):
    """Send the path exactly as given (no client-side normalisation)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    payload = r.read()
    conn.close()
    return r.status, payload


@pytest.fixture(scope="module")
def spot_id(server):
    status, payload = raw(server, "POST", "/api/solve", {"solver": {"algorithm": "dcfr", "iterations": 200}})
    assert status == 200
    return json.loads(payload)["id"]


@pytest.fixture(scope="module")
def demo_spot():
    req = dict(DEMO)
    req["solver"] = {"algorithm": "dcfr", "iterations": 300}
    return solve_spot(req)


# 1. Invalid inputs never reach unchecked native code.
def test_native_accessors_reject_bad_indices(demo_spot):
    with pytest.raises(RuntimeError):
        demo_spot.ctx.hands(2)
    with pytest.raises(RuntimeError):
        demo_spot.tree.node(-1)
    with pytest.raises(RuntimeError):
        demo_spot.tree.pot(10**6)
    with pytest.raises(ValueError):
        demo_spot.hero_view(2, "AdJd", [])
    with pytest.raises(ValueError):
        demo_spot.node_strategy(-1)


@pytest.mark.parametrize("path", ["/node/-1", "/node/99999", "/node/abc", "/hero?player=2&hand=AdJd",
                                  "/hero?player=1", "/audit?player=-1", "/tree?ids=-3"])
def test_bad_spot_queries_are_client_errors(server, spot_id, path):
    status, payload = raw(server, "GET", f"/api/spot/{spot_id}{path}")
    assert status == 400, payload


@pytest.mark.parametrize("body", [
    {"game": "kuhn", "infoset": 0, "strategy": [-1, 2]},
    {"game": "kuhn", "infoset": 0, "strategy": [1, 2, 3]},
    {"game": "kuhn", "infoset": 0, "strategy": [0, 0]},
    {"game": "kuhn", "infoset": 999, "strategy": [1, 1]},
    {"game": "nope", "infoset": 0, "strategy": [1, 1]},
])
def test_whatif_rejects_invalid_policies(server, body):
    status, _ = raw(server, "POST", "/api/microscope/whatif", body)
    assert status == 400


@pytest.mark.parametrize("solver", [{"iterations": -5}, {"iterations": 0}, {"iterations": "x"},
                                    {"iterations": 10, "threads": 0}, {"algorithm": "nope"}])
def test_solve_rejects_bad_solver_options(server, solver):
    status, _ = raw(server, "POST", "/api/solve", {"solver": solver})
    assert status == 400


def test_solve_rejects_bad_spot_parameters(server):
    for body in ({"pot": -10}, {"effective_stack": -1}, {"player_to_act": 2},
                 {"actions": {"bet_pot_fractions": [-0.5]}}):
        status, _ = raw(server, "POST", "/api/solve", dict(body, solver={"iterations": 10}))
        assert status == 400, body


# 3. EV is measured from the current decision, so folding is always worth 0.
def test_fold_is_zero_after_earlier_investment(demo_spot):
    view = demo_spot.hero_view(0, "AcAd", ["Bet 75", "AllIn 150"])
    fold = view["actions"].index("Fold")
    assert abs(view["action_ev_from_here"][fold]) < 1e-9
    assert view["committed"] == pytest.approx(50 + 75)


def test_node_endpoint_ev_baseline(server, spot_id):
    status, payload = raw(server, "GET", f"/api/spot/{spot_id}/node/0")
    node = json.loads(payload)
    # Root: nothing invested beyond the starting pot share, so a fold-free
    # action set; check the "from here" EVs are offset by exactly pot/2.
    for ev, ev_here in zip(node["action_ev"], node["action_ev_from_here"]):
        for a, b in zip(ev, ev_here):
            if a is not None:
                assert b == pytest.approx(a + 50.0)


# 4. Large trees: the initial export has no dangling children; the rest loads lazily.
def test_large_turn_tree_export_is_navigable(server):
    body = {"board": ["As", "Kh", "8c", "4d"], "effective_stack": 300, "range_p0": "AA,KK,AK", "range_p1": "QQ,JJ,AQ",
            "solver": {"algorithm": "dcfr", "iterations": 2}}
    status, payload = raw(server, "POST", "/api/solve", body)
    assert status == 200
    res = json.loads(payload)
    assert res["num_nodes"] > 6000 and res["tree_complete"] is False
    sent = {n["id"]: n for n in res["tree"]}
    # Every sent node's parent is also sent (navigable prefix).
    for n in sent.values():
        assert n["parent"] == -1 or n["parent"] in sent
    missing = sorted({c for n in sent.values() for c in n["children"] if c not in sent})
    assert missing, "expected a partial tree"
    status, payload = raw(server, "GET", f"/api/spot/{res['id']}/tree?ids={','.join(map(str, missing[:300]))}")
    assert status == 200
    got = json.loads(payload)["nodes"]
    assert [n["id"] for n in got] == missing[:300]


def test_tree_json_rejects_out_of_range(demo_spot):
    with pytest.raises(ValueError):
        tree_json(demo_spot.tree, [demo_spot.tree.num_nodes])


# 5. Figures and static files cannot escape their folders.
@pytest.mark.parametrize("path", ["/api/figure/../README.md", "/api/figure/..%2F..%2FREADME.md",
                                  "/api/figure/exp01_kuhn_equilibrium/..%2Fresults.json",
                                  "/api/figure/exp01_kuhn_equilibrium/results.json",
                                  "/api/figure/nope/x.png", "/api/results/..", "/api/results/.."])
def test_figure_and_results_routes_stay_inside_results(server, path):
    status, payload = raw(server, "GET", path)
    assert status == 404, payload
    assert b"AEGIS" not in payload  # not the README


def test_valid_figure_is_served(server):
    status, payload = raw(server, "GET", "/api/figure/exp01_kuhn_equilibrium/kuhn_nashconv.png")
    assert status == 200 and payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_static_route_does_not_leave_dist(server):
    status, payload = raw(server, "GET", "/../../README.md")
    assert status == 200
    assert b"# AEGIS" not in payload


# 6. Per-node access does not copy the whole native vector.
def test_single_node_accessor_matches_bulk(demo_spot):
    nodes = demo_spot.tree.nodes
    for i in (0, 1, len(nodes) - 1):
        assert demo_spot.tree.node(i).parent == nodes[i].parent
    assert demo_spot.tree.board(0) == demo_spot.tree.boards[0]


# 7. Unsupported algorithm/game combinations are rejected, not silently run.
@pytest.mark.parametrize("algo", ["es", "os", "cfr_sim"])
def test_river_convergence_rejects_unsupported_algorithms(server, algo):
    status, payload = raw(server, "POST", "/api/convergence", {"game": "river", "algorithms": [algo], "iterations": 5})
    assert status == 400 and b"not supported" in payload


def test_river_convergence_reports_work(server):
    status, payload = raw(server, "POST", "/api/convergence", {"game": "river", "algorithms": ["dcfr"], "iterations": 5})
    res = json.loads(payload)
    assert status == 200 and res["work_unit"] == "hand-node updates"
    assert all("nodes" in p for p in res["series"][0]["points"])


def test_default_hero_node_follows_bet_sizes(server):
    body = {"actions": {"bet_pot_fractions": [0.5], "raise_pot_fractions": [], "allow_all_in": True, "max_raises": 1},
            "solver": {"iterations": 20}, "hero": {"player": 1, "hand": "AdJd"}}
    status, payload = raw(server, "POST", "/api/solve", body)
    res = json.loads(payload)
    assert status == 200 and "hero_error" not in res
    assert res["hero"]["path"] == "Bet 50"
