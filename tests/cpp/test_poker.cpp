// Poker maths: cards, exhaustive hand-evaluator verification, ranges,
// Bayesian updates, blockers and equity.
#include <algorithm>
#include <cmath>
#include <set>

#include "poker/cards/cards.h"
#include "poker/equity/equity.h"
#include "poker/hand_evaluator/evaluator.h"
#include "poker/ranges/range.h"
#include "test_framework.h"
#include "util/rng.h"

using namespace aegis;
using namespace aegis::poker;

AEGIS_TEST(cards_parse_and_format) {
  for (Card c = 0; c < 52; ++c) EXPECT_EQ(parse_card(card_to_string(c)), c);
  EXPECT_EQ(card_to_string(parse_card("As")), std::string("As"));
  EXPECT_EQ(rank_of(parse_card("2c")), 0);
  EXPECT_EQ(rank_of(parse_card("Ad")), 12);
  const auto b = parse_cards("As Kh 8c,4d2s");
  EXPECT_EQ(b.size(), 5u);
  EXPECT_THROWS(parse_cards("AsAs"));
  EXPECT_THROWS(parse_card("Xx"));
}

AEGIS_TEST(combo_indexing_is_a_bijection) {
  const auto& C = combos();
  std::set<CardMask> masks;
  for (int i = 0; i < kNumCombos; ++i) {
    masks.insert(C.mask[i]);
    EXPECT_EQ(combo_index(C.cards[i][0], C.cards[i][1]), i);
    EXPECT_EQ(combo_index(C.cards[i][1], C.cards[i][0]), i);
    EXPECT_EQ(parse_combo(combo_to_string(i)), i);
    EXPECT_EQ(popcount(C.mask[i]), 2);
  }
  EXPECT_EQ(masks.size(), static_cast<size_t>(kNumCombos));
  int total = 0;
  std::array<int, 169> per{};
  for (int i = 0; i < kNumCombos; ++i) per[C.hand_class[i]]++;
  for (int k = 0; k < 169; ++k) {
    EXPECT_EQ(per[k], hand_class_combos(k));
    EXPECT_EQ(hand_class_index(hand_class_name(k)), k);
    total += per[k];
  }
  EXPECT_EQ(total, 1326);
  EXPECT_EQ(hand_class_name(0), std::string("AA"));
  EXPECT_EQ(hand_class_name(1), std::string("AKs"));
  EXPECT_EQ(hand_class_name(13), std::string("AKo"));
}

AEGIS_TEST(evaluator_exhaustive_five_card) {
  // All C(52,5) = 2,598,960 hands: fast == naive, known category counts,
  // and exactly 7,462 distinct equivalence classes.
  const long long expected[9] = {1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40};
  const int expected_classes[9] = {1277, 2860, 858, 858, 10, 1277, 156, 156, 10};
  long long count[9] = {};
  std::set<HandValue> distinct[9];
  long long mismatches = 0, total = 0;
  Card h[5];
  for (h[0] = 0; h[0] < 52; ++h[0])
    for (h[1] = h[0] + 1; h[1] < 52; ++h[1])
      for (h[2] = h[1] + 1; h[2] < 52; ++h[2])
        for (h[3] = h[2] + 1; h[3] < 52; ++h[3])
          for (h[4] = h[3] + 1; h[4] < 52; ++h[4]) {
            const HandValue a = evaluate(h, 5);
            const HandValue b = evaluate_naive5(h);
            if (a != b) ++mismatches;
            count[category_of(a)]++;
            distinct[category_of(a)].insert(a);
            ++total;
          }
  EXPECT_EQ(total, 2598960LL);
  EXPECT_EQ(mismatches, 0LL);
  size_t classes = 0;
  for (int c = 0; c < 9; ++c) {
    EXPECT_EQ(count[c], expected[c]);
    EXPECT_EQ(static_cast<int>(distinct[c].size()), expected_classes[c]);
    classes += distinct[c].size();
  }
  EXPECT_EQ(classes, 7462u);
}

AEGIS_TEST(evaluator_seven_card_matches_best_five_subset) {
  Rng rng(2024);
  const int subsets[21][5] = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 6}, {0, 1, 2, 4, 5}, {0, 1, 2, 4, 6},
                              {0, 1, 2, 5, 6}, {0, 1, 3, 4, 5}, {0, 1, 3, 4, 6}, {0, 1, 3, 5, 6}, {0, 1, 4, 5, 6},
                              {0, 2, 3, 4, 5}, {0, 2, 3, 4, 6}, {0, 2, 3, 5, 6}, {0, 2, 4, 5, 6}, {0, 3, 4, 5, 6},
                              {1, 2, 3, 4, 5}, {1, 2, 3, 4, 6}, {1, 2, 3, 5, 6}, {1, 2, 4, 5, 6}, {1, 3, 4, 5, 6},
                              {2, 3, 4, 5, 6}};
  long long mismatches = 0;
  for (int trial = 0; trial < 200000; ++trial) {
    CardMask used = 0;
    Card h[7];
    for (int k = 0; k < 7; ++k) {
      Card c;
      do c = static_cast<Card>(rng.below(52));
      while (used & card_bit(c));
      used |= card_bit(c);
      h[k] = c;
    }
    HandValue best = 0;
    for (const auto& s : subsets) {
      Card f[5] = {h[s[0]], h[s[1]], h[s[2]], h[s[3]], h[s[4]]};
      best = std::max(best, evaluate_naive5(f));
    }
    if (evaluate(h, 7) != best) ++mismatches;
    if (evaluate(used) != best) ++mismatches;
  }
  EXPECT_EQ(mismatches, 0LL);
}

AEGIS_SLOW_TEST(evaluator_exhaustive_seven_card_category_counts) {
  // All C(52,7) = 133,784,560 seven-card hands.
  const long long expected[9] = {23294460, 58627800, 31433400, 6461620, 6180020,
                                 4047644,  3473184,  224848,   41584};
  long long count[9] = {};
  Card h[7];
  for (h[0] = 0; h[0] < 52; ++h[0])
    for (h[1] = h[0] + 1; h[1] < 52; ++h[1])
      for (h[2] = h[1] + 1; h[2] < 52; ++h[2])
        for (h[3] = h[2] + 1; h[3] < 52; ++h[3])
          for (h[4] = h[3] + 1; h[4] < 52; ++h[4])
            for (h[5] = h[4] + 1; h[5] < 52; ++h[5])
              for (h[6] = h[5] + 1; h[6] < 52; ++h[6]) count[category_of(evaluate(h, 7))]++;
  long long total = 0;
  for (int c = 0; c < 9; ++c) {
    EXPECT_EQ(count[c], expected[c]);
    total += count[c];
  }
  EXPECT_EQ(total, 133784560LL);
}

AEGIS_TEST(evaluator_describes_hands) {
  const auto b = parse_cards("AsKh8c4d2s");
  std::vector<Card> h = b;
  h.push_back(parse_card("Ad"));
  h.push_back(parse_card("Jd"));
  const HandValue v = evaluate(h.data(), 7);
  EXPECT_EQ(category_of(v), static_cast<int>(kPair));
  EXPECT_EQ(describe(v), std::string("Pair, Aces"));
  const auto wheel = parse_cards("Ah2c3d4s5h9cTd");
  EXPECT_EQ(describe(evaluate(wheel.data(), 7)), std::string("Straight, Five-high"));
}

AEGIS_TEST(range_parser) {
  EXPECT_EQ(Range::parse("AA").num_nonzero(), 6);
  EXPECT_EQ(Range::parse("AKs").num_nonzero(), 4);
  EXPECT_EQ(Range::parse("AKo").num_nonzero(), 12);
  EXPECT_EQ(Range::parse("AK").num_nonzero(), 16);
  EXPECT_EQ(Range::parse("TT+").num_nonzero(), 30);
  EXPECT_EQ(Range::parse("A2s+").num_nonzero(), 48);
  EXPECT_EQ(Range::parse("KTo+").num_nonzero(), 36);  // KT, KJ, KQ offsuit
  EXPECT_EQ(Range::parse("22-55").num_nonzero(), 24);
  EXPECT_EQ(Range::parse("A2s-A5s").num_nonzero(), 16);
  EXPECT_EQ(Range::parse("AsKh").num_nonzero(), 1);
  EXPECT_EQ(Range::parse("random").num_nonzero(), 1326);
  const auto r = Range::parse("AA:0.5, KK, AKs:0.25");
  EXPECT_NEAR(r.total(), 6 * 0.5 + 6 + 4 * 0.25, 1e-12);
  EXPECT_THROWS(Range::parse("AZs"));
  EXPECT_THROWS(Range::parse("AA:-1"));
  // to_string round trip.
  const auto rt = Range::parse(r.to_string());
  for (int i = 0; i < kNumCombos; ++i) EXPECT_NEAR(rt[i], r[i], 1e-12);
}

AEGIS_TEST(range_card_removal_and_bayes_update) {
  auto r = Range::full();
  r.remove(mask_of(parse_cards("AsKh8c4d2s")));
  EXPECT_EQ(r.num_nonzero(), 1081);  // C(47, 2)
  r.normalize();
  EXPECT_NEAR(r.total(), 1.0, 1e-12);
  // Likelihood: hands containing an Ace always bet, others bet 25%.
  std::array<double, kNumCombos> like{};
  const auto& C = combos();
  double z = 0.0;
  for (int i = 0; i < kNumCombos; ++i) {
    const bool ace = rank_of(C.cards[i][0]) == 12 || rank_of(C.cards[i][1]) == 12;
    like[i] = ace ? 1.0 : 0.25;
    z += r[i] * like[i];
  }
  const auto post = r.bayes_update(like);
  EXPECT_NEAR(post.total(), 1.0, 1e-12);
  for (int i = 0; i < kNumCombos; ++i) EXPECT_NEAR(post[i], r[i] * like[i] / z, 1e-15);
  // Dead combos stay dead.
  EXPECT_NEAR(post[parse_combo("AsAd")], 0.0, 0.0);
  std::array<double, kNumCombos> zero{};
  EXPECT_THROWS(r.bayes_update(zero));
}

AEGIS_TEST(blockers) {
  // Holding the Ace of spades removes 3 of 6 AA combos (and 1 of 4 AKs).
  const auto r = Range::parse("AA,AKs");
  const CardMask hero = mask_of(parse_cards("AsJd"));
  EXPECT_NEAR(r.blocked_fraction(hero), (3.0 + 1.0) / 10.0, 1e-12);
  const auto cond = r.conditioned_on(hero);
  EXPECT_EQ(cond.num_nonzero(), 6);
  EXPECT_NEAR(cond.total(), 1.0, 1e-12);
}

AEGIS_TEST(equity_symmetry_and_consistency) {
  // Suit-symmetric matchup is exactly 50%.
  const auto e = equity_exact(parse_combo("AsKs"), parse_combo("AhKh"), {});
  EXPECT_NEAR(e.win, e.lose, 1e-12);
  EXPECT_NEAR(e.win + e.tie + e.lose, 1.0, 1e-12);
  EXPECT_EQ(e.samples, 1712304LL);  // C(48, 5)
  // AA vs KK is roughly 82/18.
  const auto aa = equity_exact(parse_combo("AcAd"), parse_combo("KhKs"), {});
  EXPECT_TRUE(aa.equity() > 0.80 && aa.equity() < 0.84);
  // Hand vs range equals the average of hand-vs-hand equities on a flop.
  const auto board = parse_cards("Kh8c4d");
  const int hero = parse_combo("AdJd");
  const auto villain = Range::parse("KQs,88,A4s,QJo:0.5");
  const auto hr = equity_vs_range_exact(hero, villain, board);
  double num = 0.0, den = 0.0;
  const auto& C = combos();
  const CardMask dead = mask_of(board) | C.mask[hero];
  for (int v = 0; v < kNumCombos; ++v) {
    if (villain[v] <= 0.0 || (C.mask[v] & dead)) continue;
    num += villain[v] * equity_exact(hero, v, board).equity();
    den += villain[v];
  }
  EXPECT_NEAR(hr.equity(), num / den, 1e-12);
}

AEGIS_TEST(range_vs_range_exact_matches_brute_force) {
  for (const char* board_str : {"AsKh8c4d2s", "Kh8c4d", "9h9c4d7s"}) {
    const auto board = parse_cards(board_str);
    const auto hero = Range::parse("AK,88,A4s,QJs:0.7,T9s");
    const auto villain = Range::parse("KQ,99:0.3,A5s,JTs,76s");
    const auto fast = equity_range_vs_range_exact(hero, villain, board);
    const auto& C = combos();
    const CardMask bm = mask_of(board);
    double num = 0.0, den = 0.0;
    for (int h = 0; h < kNumCombos; ++h) {
      if (hero[h] <= 0.0 || (C.mask[h] & bm)) continue;
      double hn = 0.0, hd = 0.0;
      for (int v = 0; v < kNumCombos; ++v) {
        if (villain[v] <= 0.0 || (C.mask[v] & (bm | C.mask[h]))) continue;
        const double eq = equity_exact(h, v, board).equity();
        hn += villain[v] * eq;
        hd += villain[v];
      }
      if (hd > 0) EXPECT_NEAR(fast.combo_equity[h], hn / hd, 1e-9);
      num += hero[h] * hn;
      den += hero[h] * hd;
    }
    EXPECT_NEAR(fast.overall.equity(), num / den, 1e-9);
  }
}

AEGIS_TEST(monte_carlo_equity_agrees_with_exact) {
  const auto board = parse_cards("Kh8c4d");
  const auto hero = Range::parse("AK,88,A4s");
  const auto villain = Range::parse("KQ,99,A5s,JTs");
  const auto ex = equity_range_vs_range_exact(hero, villain, board);
  const auto mc = equity_range_vs_range_mc(hero, villain, board, 400000, 17);
  EXPECT_TRUE(std::abs(mc.equity() - ex.overall.equity()) < 4.5 * mc.std_error);
}

AEGIS_TEST(random_range_generator) {
  Rng rng(5);
  std::vector<int> order(kNumCombos);
  for (int i = 0; i < kNumCombos; ++i) order[i] = i;
  const auto r = random_range_by_strength(order, rng);
  EXPECT_TRUE(r.num_nonzero() > 0);
  double mx = 0.0;
  for (double x : r.w) mx = std::max(mx, x);
  EXPECT_NEAR(mx, 1.0, 1e-12);
}
