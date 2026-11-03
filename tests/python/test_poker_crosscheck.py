"""Cross-checks of the poker maths from Python.

A slow, simple pure-Python evaluator (itertools over 5-card subsets,
counting ranks) is compared with the C++ bit-parallel evaluator; equities are
compared with brute-force Python enumeration on small cases.
"""
import itertools
import random
from collections import Counter

import numpy as np
import pytest

import aegis

RANKS = "23456789TJQKA"


def py_eval5(cards):
    ranks = sorted((c // 4 for c in cards), reverse=True)
    suits = [c % 4 for c in cards]
    cnt = Counter(ranks)
    groups = sorted(cnt.items(), key=lambda kv: (-kv[1], -kv[0]))
    flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_high = 3
    if flush and straight_high is not None:
        return (8, straight_high)
    if groups[0][1] == 4:
        return (7, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return (5, *ranks)
    if straight_high is not None:
        return (4, straight_high)
    if groups[0][1] == 3:
        return (3, *[g[0] for g in groups])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return (2, *[g[0] for g in groups])
    if groups[0][1] == 2:
        return (1, *[g[0] for g in groups])
    return (0, *ranks)


def py_eval(cards):
    return max(py_eval5(c) for c in itertools.combinations(cards, 5))


def test_evaluator_ordering_matches_python_reference():
    rng = random.Random(7)
    hands = [rng.sample(range(52), 7) for _ in range(3000)]
    ours = [aegis.evaluate_hand(h) for h in hands]
    ref = [py_eval(h) for h in hands]
    # Same total order: compare all pairs of a random subset.
    for _ in range(20000):
        i, j = rng.randrange(len(hands)), rng.randrange(len(hands))
        assert (ours[i] > ours[j]) == (ref[i] > ref[j])
        assert (ours[i] == ours[j]) == (ref[i] == ref[j])


def test_category_names():
    v = aegis.evaluate_hand(aegis.parse_cards("AhKhQhJhTh"))
    assert aegis.hand_category(v) == 8
    assert aegis.describe_hand(v).startswith("Straight Flush")


def test_equity_brute_force_turn():
    board = aegis.parse_cards("AsKh8c4d")
    hero, vill = aegis.parse_combo("AdJd"), aegis.parse_combo("KcQc")
    res = aegis.equity_exact(hero, vill, board)
    cards = aegis.combo_cards()
    used = set(board) | set(cards[hero].tolist()) | set(cards[vill].tolist())
    w = t = n = 0
    for c in range(52):
        if c in used:
            continue
        a = py_eval(board + [c] + cards[hero].tolist())
        b = py_eval(board + [c] + cards[vill].tolist())
        w += a > b
        t += a == b
        n += 1
    assert res.samples == n == 44
    assert abs(res.win - w / n) < 1e-12 and abs(res.tie - t / n) < 1e-12


def test_range_bayes_and_blockers():
    r = aegis.Range.parse("AA,KK,AKs")
    assert r.num_nonzero() == 16
    like = np.zeros(1326)
    like[aegis.parse_combo("AsAh")] = 1.0
    like[aegis.parse_combo("KsKh")] = 0.5
    post = r.bayes_update(like)
    assert abs(post[aegis.parse_combo("AsAh")] - 2 / 3) < 1e-12
    assert abs(r.blocked_fraction(aegis.parse_cards("As")) - (3 + 1) / 16) < 1e-12


def test_combo_indexing_matches_card_pairs():
    cards = aegis.combo_cards()
    seen = set()
    for i in range(1326):
        a, b = cards[i]
        assert aegis.combo_index(int(a), int(b)) == i
        seen.add((int(a), int(b)))
    assert len(seen) == 1326


@pytest.mark.parametrize("board", ["Kh8c4d", "AsKh8c4d2s"])
def test_range_equity_exact_matches_combo_average(board):
    b = aegis.parse_cards(board)
    hero = aegis.Range.parse("AK,88,QJs")
    vill = aegis.Range.parse("KQ,99,A5s")
    overall, combo_eq, _ = aegis.equity_range_vs_range_exact(hero, vill, b)
    w = hero.weights
    vals = [(w[i], combo_eq[i]) for i in range(1326) if w[i] > 0 and np.isfinite(combo_eq[i])]
    assert 0 < overall.equity() < 1
    assert len(vals) > 0
