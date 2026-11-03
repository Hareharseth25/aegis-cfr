"""Bridge between AEGIS game trees and OpenSpiel (used as an external oracle).

AEGIS implements its games and algorithms independently. To compare results we
walk an AEGIS GameTree and the corresponding OpenSpiel game in lockstep (both
use the same action ids), building a bijection between AEGIS information sets
and OpenSpiel information-state strings.
"""
from __future__ import annotations

import numpy as np

from . import aegis_core as ac


def infoset_mapping(tree: "ac.GameTree", game) -> dict[int, str]:
    """Return {aegis infoset index: OpenSpiel information_state_string}."""
    node_type = tree.node_type
    node_player = tree.node_player
    node_infoset = tree.node_infoset
    child_begin = tree.child_begin
    child_count = tree.child_count
    children = tree.children
    child_action = tree.child_action
    mapping: dict[int, str] = {}
    reverse: dict[str, int] = {}

    def walk(node: int, state) -> None:
        t = node_type[node]
        if t == int(ac.NodeType.TERMINAL):
            assert state.is_terminal(), "AEGIS terminal but OpenSpiel is not"
            return
        if t == int(ac.NodeType.CHANCE):
            assert state.is_chance_node()
            os_outcomes = dict(state.chance_outcomes())
        else:
            assert state.current_player() == node_player[node]
            key = state.information_state_string()
            iset = int(node_infoset[node])
            if iset in mapping:
                assert mapping[iset] == key, "infoset maps to two OpenSpiel strings"
            mapping[iset] = key
            if key in reverse:
                assert reverse[key] == iset, "OpenSpiel string maps to two infosets"
            reverse[key] = iset
            assert sorted(state.legal_actions()) == sorted(
                child_action[child_begin[node]: child_begin[node] + child_count[node]].tolist())
        b, c = child_begin[node], child_count[node]
        for k in range(c):
            a = int(child_action[b + k])
            if t == int(ac.NodeType.CHANCE):
                assert abs(os_outcomes[a] - tree.chance_prob[b + k]) < 1e-12
            walk(int(children[b + k]), state.child(a))

    walk(0, game.new_initial_state())
    return mapping


def to_openspiel_policy(tree: "ac.GameTree", policy: "ac.TabularPolicy", game, mapping=None):
    """Convert an AEGIS TabularPolicy to an OpenSpiel TabularPolicy."""
    from open_spiel.python import policy as os_policy

    mapping = mapping or infoset_mapping(tree, game)
    out = os_policy.TabularPolicy(game)
    probs = policy.probs
    for iset, key in mapping.items():
        info = tree.infoset(iset)
        row = out.state_lookup[key]
        out.action_probability_array[row, :] = 0.0
        for k, a in enumerate(info.actions):
            out.action_probability_array[row, a] = probs[info.offset + k]
    return out


def from_openspiel_policy(tree: "ac.GameTree", os_pol, game, mapping=None) -> "ac.TabularPolicy":
    """Convert an OpenSpiel TabularPolicy (or callable policy) to AEGIS."""
    mapping = mapping or infoset_mapping(tree, game)
    pol = ac.TabularPolicy(tree)
    probs = pol.probs
    for iset, key in mapping.items():
        info = tree.infoset(iset)
        row = os_pol.state_lookup[key]
        for k, a in enumerate(info.actions):
            probs[info.offset + k] = os_pol.action_probability_array[row, a]
    pol.probs = probs
    return pol


def policy_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
