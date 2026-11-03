"""Check AEGIS against OpenSpiel.

OpenSpiel is only used here, as an outside reference: AEGIS's games, CFR
code and best response are written separately. The tests check that
 * the game trees are identical (sizes, information sets, chance, payoffs);
 * exploitability / NashConv of arbitrary policies agree;
 * CFR, CFR+ and DCFR produce the same average policies iterate-for-iterate.
"""
import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
from open_spiel.python.algorithms import cfr as os_cfr  # noqa: E402
from open_spiel.python.algorithms import discounted_cfr as os_dcfr  # noqa: E402
from open_spiel.python.algorithms import exploitability as os_expl  # noqa: E402

import aegis  # noqa: E402
from aegis.openspiel_bridge import from_openspiel_policy, infoset_mapping, to_openspiel_policy  # noqa: E402

GAMES = [("kuhn_poker", lambda: aegis.KuhnGame()), ("leduc_poker", lambda: aegis.LeducGame())]


@pytest.fixture(scope="module", params=GAMES, ids=[g[0] for g in GAMES])
def setup(request):
    name, make = request.param
    os_game = pyspiel.load_game(name)
    game = make()
    tree = aegis.GameTree.build(game)
    mapping = infoset_mapping(tree, os_game)
    return name, os_game, game, tree, mapping


def test_tree_bijection(setup):
    name, os_game, game, tree, mapping = setup
    assert len(mapping) == tree.num_infosets
    assert len(set(mapping.values())) == tree.num_infosets
    expected = {"kuhn_poker": (58, 12), "leduc_poker": (9457, 936)}[name]
    assert (tree.num_nodes, tree.num_infosets) == expected


def test_terminal_payoffs_match(setup):
    name, os_game, game, tree, mapping = setup
    # Expected values under random policies agree -> payoffs and chance agree.
    for seed in range(3):
        pol = aegis.TabularPolicy.random(tree, seed)
        os_pol = to_openspiel_policy(tree, pol, os_game, mapping)
        from open_spiel.python.algorithms import expected_game_score
        os_val = expected_game_score.policy_value(os_game.new_initial_state(), [os_pol, os_pol])
        ours = aegis.expected_values(tree, pol)
        assert abs(os_val[0] - ours[0]) < 1e-10
        assert abs(os_val[1] - ours[1]) < 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_exploitability_of_random_policies_matches(setup, seed):
    name, os_game, game, tree, mapping = setup
    pol = aegis.TabularPolicy.random(tree, seed, 0.5)
    os_pol = to_openspiel_policy(tree, pol, os_game, mapping)
    ours = aegis.exploitability_report(tree, pol)
    theirs = os_expl.nash_conv(os_game, os_pol)
    assert abs(ours.nash_conv - theirs) < 1e-9
    assert abs(ours.exploitability - os_expl.exploitability(os_game, os_pol)) < 1e-9


def test_uniform_policy_exploitability(setup):
    name, os_game, game, tree, mapping = setup
    pol = aegis.TabularPolicy.uniform(tree)
    os_pol = to_openspiel_policy(tree, pol, os_game, mapping)
    assert abs(aegis.nash_conv(tree, pol) - os_expl.nash_conv(os_game, os_pol)) < 1e-12
    if name == "kuhn_poker":
        assert abs(aegis.nash_conv(tree, pol) - 0.9166666666666666) < 1e-12


def _compare_solver(tree, os_game, mapping, ours_cfg, os_solver, iterations):
    ours = aegis.TabularCFR(tree, ours_cfg)
    for it in range(1, iterations + 1):
        ours.iterate(1)
        os_solver.evaluate_and_update_policy()
        if it in (1, 2, 5, iterations):
            os_avg = os_solver.average_policy()
            converted = from_openspiel_policy(tree, os_avg, os_game, mapping)
            diff = np.max(np.abs(ours.average_policy().probs - converted.probs))
            assert diff < 1e-9, f"iteration {it}: max |Δσ̄| = {diff}"
    return ours


def test_vanilla_cfr_iterates_match(setup):
    name, os_game, game, tree, mapping = setup
    cfg = aegis.CFRConfig.vanilla(True)
    cfg.tie_tolerance = 0.0  # bit-for-bit arithmetic comparison
    _compare_solver(tree, os_game, mapping, cfg, os_cfr.CFRSolver(os_game), 50 if name == "leduc_poker" else 200)


def test_cfr_plus_iterates_match(setup):
    name, os_game, game, tree, mapping = setup
    cfg = aegis.CFRConfig.cfr_plus()
    cfg.tie_tolerance = 0.0
    _compare_solver(tree, os_game, mapping, cfg, os_cfr.CFRPlusSolver(os_game), 50 if name == "leduc_poker" else 200)


def test_dcfr_iterates_match(setup):
    name, os_game, game, tree, mapping = setup
    cfg = aegis.CFRConfig.dcfr(1.5, 0.0, 2.0)
    cfg.tie_tolerance = 0.0
    _compare_solver(tree, os_game, mapping, cfg, os_dcfr.DCFRSolver(os_game, alpha=1.5, beta=0, gamma=2),
                    30 if name == "leduc_poker" else 100)


def test_cfr_plus_converges_to_openspiel_equilibrium_value(setup):
    name, os_game, game, tree, mapping = setup
    ours = aegis.TabularCFR(tree, aegis.CFRConfig.cfr_plus())
    ours.iterate(1000)
    avg = ours.average_policy()
    nc = aegis.nash_conv(tree, avg)
    os_nc = os_expl.nash_conv(os_game, to_openspiel_policy(tree, avg, os_game, mapping))
    assert abs(nc - os_nc) < 1e-10
    assert nc < 1e-3
