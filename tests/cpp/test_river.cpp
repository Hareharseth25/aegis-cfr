// No-Limit betting rules, vector-form river/turn CFR,
// analytic equilibria, and cross-validation against the generic tabular
// CFR + exact best response on the same subgame.
#include <cmath>

#include "core/best_response/best_response.h"
#include "core/game_tree/game_tree.h"
#include "games/holdem_subgame_efg.h"
#include "poker/betting_rules/betting.h"
#include "solvers/subgame_resolver/subgame_context.h"
#include "solvers/subgame_resolver/vector_cfr.h"
#include "solvers/vanilla_cfr/cfr.h"
#include "test_framework.h"
#include "util/check.h"

using namespace aegis;
using namespace aegis::poker;

namespace {

TreeConfig river_config(const char* board, double pot, double stack) {
  TreeConfig c;
  c.board = parse_cards(board);
  c.pot = pot;
  c.effective_stack = stack;
  StreetSizes s;
  s.bet_fractions = {0.33, 0.75, 1.5};
  s.raise_fractions = {1.0};
  s.allow_all_in = true;
  s.max_raises = 3;
  c.set_sizes(s, s);
  return c;
}

Range random_small_range(Rng& rng, CardMask dead, int n) {
  Range r;
  int added = 0;
  while (added < n) {
    const int c = static_cast<int>(rng.below(kNumCombos));
    if ((combos().mask[c] & dead) || r[c] > 0) continue;
    r[c] = 0.2 + rng.uniform();
    ++added;
  }
  return r;
}

TabularPolicy to_tabular(const GameTree& t, const SubgameContext& ctx, const VectorPolicy& vp) {
  TabularPolicy tp(t);
  const auto& tree = ctx.tree();
  for (int I = 0; I < t.num_infosets(); ++I) {
    const auto& info = t.infosets[I];
    // key "n<node>h<hand>"
    const auto hpos = info.key.find('h');
    const int node = std::stoi(info.key.substr(1, hpos - 1));
    const int hand = std::stoi(info.key.substr(hpos + 1));
    for (int a = 0; a < info.num_actions; ++a)
      tp.probs[info.offset + a] = vp.prob(tree, ctx.hand_spaces(), node, a, hand);
  }
  return tp;
}

}  // namespace

AEGIS_TEST(nl_betting_tree_rules) {
  auto cfg = river_config("AsKh8c4d2s", 100, 150);
  auto t = PublicTree::build(cfg);
  const auto& root = t.nodes[0];
  EXPECT_EQ(root.num_children, 4);  // Check, Bet 33, Bet 75, AllIn 150 (1.5 pot = stack merges)
  EXPECT_EQ(t.nodes[root.first_child].action.to_string(), std::string("Check"));
  EXPECT_EQ(t.nodes[root.first_child + 1].action.to_string(), std::string("Bet 33"));
  EXPECT_EQ(t.nodes[root.first_child + 2].action.to_string(), std::string("Bet 75"));
  EXPECT_EQ(t.nodes[root.first_child + 3].action.to_string(), std::string("AllIn 150"));
  const int bet75 = root.first_child + 2;
  EXPECT_NEAR(t.nodes[bet75].commit[0], 125.0, 1e-12);
  // Facing 75: Fold, Call, AllIn (pot raise to 325 exceeds the stack and merges).
  const auto& f = t.nodes[bet75];
  EXPECT_EQ(f.num_children, 3);
  EXPECT_EQ(t.nodes[f.first_child].action.to_string(), std::string("Fold"));
  EXPECT_EQ(t.nodes[f.first_child + 1].action.to_string(), std::string("Call"));
  EXPECT_EQ(t.nodes[f.first_child + 2].action.to_string(), std::string("AllIn 150"));
  EXPECT_TRUE(t.nodes[f.first_child + 1].type == PNodeType::kShowdown);
  EXPECT_NEAR(t.nodes[f.first_child + 1].commit[1], 125.0, 1e-12);
  // Check-check ends in a showdown for the pot.
  const int chk = root.first_child;
  const int chkchk = t.nodes[chk].first_child;
  EXPECT_TRUE(t.nodes[chkchk].type == PNodeType::kShowdown);
  EXPECT_NEAR(t.nodes[chkchk].commit[0], 50.0, 1e-12);
  // Min-raise: facing a 33 bet, a raise must add at least 33.
  const int bet33 = root.first_child + 1;
  for (int k = 0; k < t.nodes[bet33].num_children; ++k) {
    const auto& c = t.nodes[t.nodes[bet33].first_child + k];
    if (c.action.kind == ActionKind::kRaise) EXPECT_TRUE(c.action.amount >= 66.0 - 1e-9);
  }
  // Every fold/showdown is zero-sum-consistent: commits never exceed pot/2 + stack.
  for (const auto& n : t.nodes)
    for (int p = 0; p < 2; ++p) EXPECT_TRUE(n.commit[p] <= 50.0 + 150.0 + 1e-9);
}

AEGIS_TEST(turn_tree_has_river_chance_nodes) {
  TreeConfig c = river_config("AsKh8c4d", 100, 200);
  StreetSizes s;
  s.bet_fractions = {0.75};
  s.raise_fractions = {};
  s.max_raises = 1;
  c.set_sizes(s, s);
  auto t = PublicTree::build(c);
  int chance = 0;
  for (int n = 0; n < t.num_nodes(); ++n)
    if (t.nodes[n].type == PNodeType::kChance) {
      ++chance;
      EXPECT_EQ(t.nodes[n].num_children, 48);
    }
  EXPECT_TRUE(chance > 0);
  c.depth_limited = true;
  auto dl = PublicTree::build(c);
  EXPECT_TRUE(dl.count_type(PNodeType::kLeaf) > 0);
  EXPECT_TRUE(dl.num_nodes() < t.num_nodes());
}

AEGIS_TEST(polarized_river_matches_analytic_equilibrium) {
  // P0: 3 nut combos (KK, set) + 6 air combos (65). P1: 12 bluff-catchers (AQ).
  // P0 may check or bet pot (= all-in); P1 may only call/fold or check back.
  // Theory: P0 bluffs B/(P+B)*|value| = 1.5 combos (25% of air), P1 calls
  // P/(P+B) = 50%, game value 0.
  TreeConfig c;
  c.board = parse_cards("KsQd7h4c2s");
  c.pot = 100;
  c.effective_stack = 100;
  StreetSizes p0;
  p0.bet_fractions = {1.0};
  p0.raise_fractions = {};
  p0.allow_all_in = false;
  p0.max_raises = 0;
  StreetSizes p1 = p0;
  p1.bet_fractions = {};
  c.sizes[1][0] = p0;
  c.sizes[1][1] = p1;
  auto tree = PublicTree::build(c);
  const auto r0 = Range::parse("KhKc,KhKd,KcKd,6h5h,6c5c,6d5d,6s5s,6h5c,6c5h");
  const auto r1 = Range::parse("AQ");
  SubgameContext ctx(tree, r0, r1);
  EXPECT_EQ(ctx.n(1), 12);
  VectorCFR solver(ctx, {CFRConfig::dcfr()});
  solver.iterate(4000);
  const auto avg = solver.average_policy();
  const auto ex = ctx.exploitability(avg);
  EXPECT_TRUE(ex.exploitability_pct_pot < 0.05);
  const auto ev = ctx.evaluate(avg);
  EXPECT_NEAR(ev.ev[0], 0.0, 0.1);
  // Root: actions [Check, AllIn 100].
  const int root = 0;
  double value_bet = 0.0, air_bet = 0.0;
  const auto& H0 = ctx.hands(0);
  for (int i = 0; i < H0.size(); ++i) {
    const bool kings = rank_of(H0.cards[i][0]) == 11;
    const double b = avg.prob(tree, ctx.hand_spaces(), root, 1, i);
    if (kings) value_bet += b;
    else air_bet += b;
  }
  EXPECT_NEAR(value_bet, 3.0, 0.01);
  EXPECT_NEAR(air_bet, 1.5, 0.05);
  const int bet_node = tree.nodes[root].first_child + 1;
  double call = 0.0;
  for (int i = 0; i < ctx.n(1); ++i) call += avg.prob(tree, ctx.hand_spaces(), bet_node, 1, i) / ctx.n(1);
  EXPECT_NEAR(call, 0.5, 0.02);
}

AEGIS_TEST(vector_cfr_matches_tabular_cfr_iterates_on_river) {
  Rng rng(42);
  auto cfg = river_config("AsKh8c4d2s", 100, 150);
  cfg.sizes[1][0].bet_fractions = cfg.sizes[1][1].bet_fractions = {0.5, 1.0};
  cfg.sizes[1][0].max_raises = cfg.sizes[1][1].max_raises = 2;
  auto tree = PublicTree::build(cfg);
  const CardMask dead = tree.board_masks[0];
  const auto r0 = random_small_range(rng, dead, 14);
  const auto r1 = random_small_range(rng, dead, 12);
  SubgameContext ctx(tree, r0, r1);
  HoldemSubgameGame efg(ctx);
  auto gt = GameTree::build(efg);
  for (const auto& rule : {CFRConfig::vanilla(true), CFRConfig::cfr_plus(), CFRConfig::dcfr(), CFRConfig::linear()}) {
    VectorCFR vec(ctx, {rule});
    TabularCFR tab(gt, rule);
    vec.iterate(60);
    tab.iterate(60);
    const auto vpol = vec.average_policy();
    const auto tpol = tab.average_policy();
    const auto converted = to_tabular(gt, ctx, vpol);
    double max_diff = 0.0;
    for (int s = 0; s < gt.total_actions; ++s) max_diff = std::max(max_diff, std::abs(converted.probs[s] - tpol.probs[s]));
    EXPECT_TRUE(max_diff < 1e-9);
    const auto vex = ctx.exploitability(vpol);
    const auto tex = exploitability_report(gt, tpol);
    EXPECT_NEAR(vex.nash_conv, tex.nash_conv, 1e-8);
    EXPECT_NEAR(ctx.evaluate(vpol).ev[0], expected_values(gt, tpol)[0], 1e-8);
  }
}

AEGIS_TEST(vector_best_response_matches_tabular_best_response) {
  Rng rng(9);
  auto cfg = river_config("Td9d6c3s2h", 60, 100);
  auto tree = PublicTree::build(cfg);
  const CardMask dead = tree.board_masks[0];
  const auto r0 = random_small_range(rng, dead, 18);
  const auto r1 = random_small_range(rng, dead, 18);
  SubgameContext ctx(tree, r0, r1);
  HoldemSubgameGame efg(ctx);
  auto gt = GameTree::build(efg);
  for (int trial = 0; trial < 3; ++trial) {
    VectorPolicy vp = ctx.uniform_policy();
    for (auto& s : vp.strategy) {
      // Random per-hand distributions (action-major layout).
      (void)s;
    }
    for (int n = 0; n < tree.num_nodes(); ++n) {
      const auto& nd = tree.nodes[n];
      if (nd.type != PNodeType::kDecision) continue;
      const int np = ctx.n(nd.player), A = nd.num_children;
      auto& s = vp.strategy[nd.decision_index];
      for (int i = 0; i < np; ++i) {
        double tot = 0.0;
        for (int a = 0; a < A; ++a) tot += (s[static_cast<size_t>(a) * np + i] = rng.gamma(0.7) + 1e-6);
        for (int a = 0; a < A; ++a) s[static_cast<size_t>(a) * np + i] /= tot;
      }
    }
    const auto tp = to_tabular(gt, ctx, vp);
    for (int p = 0; p < 2; ++p)
      EXPECT_NEAR(ctx.best_response(vp, p).ev[p], best_response(gt, tp, p).value, 1e-9);
  }
}

AEGIS_TEST(vector_cfr_matches_tabular_on_turn_with_chance) {
  Rng rng(3);
  TreeConfig c = river_config("AsKh8c4d", 80, 120);
  StreetSizes s;
  s.bet_fractions = {0.75};
  s.raise_fractions = {};
  s.allow_all_in = true;
  s.max_raises = 1;
  c.set_sizes(s, s);
  auto tree = PublicTree::build(c);
  const CardMask dead = tree.board_masks[0];
  const auto r0 = random_small_range(rng, dead, 7);
  const auto r1 = random_small_range(rng, dead, 7);
  SubgameContext ctx(tree, r0, r1);
  HoldemSubgameGame efg(ctx);
  auto gt = GameTree::build(efg);
  VectorCFR vec(ctx, {CFRConfig::cfr_plus()});
  TabularCFR tab(gt, CFRConfig::cfr_plus());
  vec.iterate(25);
  tab.iterate(25);
  const auto conv = to_tabular(gt, ctx, vec.average_policy());
  const auto tpol = tab.average_policy();
  double max_diff = 0.0;
  for (int k = 0; k < gt.total_actions; ++k) max_diff = std::max(max_diff, std::abs(conv.probs[k] - tpol.probs[k]));
  EXPECT_TRUE(max_diff < 1e-9);
  EXPECT_NEAR(ctx.exploitability(vec.average_policy()).nash_conv, nash_conv(gt, tpol), 1e-8);
}

AEGIS_TEST(multithreaded_turn_solve_is_deterministic) {
  TreeConfig c = river_config("Jh9s5c2d", 100, 200);
  StreetSizes s;
  s.bet_fractions = {0.5, 1.0};
  s.raise_fractions = {};
  s.max_raises = 1;
  c.set_sizes(s, s);
  auto tree = PublicTree::build(c);
  const auto r0 = Range::parse("random");
  const auto r1 = Range::parse("random");
  SubgameContext serial(tree, r0, r1, 1), parallel(tree, r0, r1, 4);
  VectorCFR a(serial, {CFRConfig::dcfr()}), b(parallel, {CFRConfig::dcfr()});
  a.iterate(3);
  b.iterate(3);
  const auto pa = a.average_policy(), pb = b.average_policy();
  EXPECT_TRUE(pa.strategy == pb.strategy);
}

AEGIS_TEST(full_range_river_converges) {
  auto cfg = river_config("AsKh8c4d2s", 100, 150);
  auto tree = PublicTree::build(cfg);
  SubgameContext ctx(tree, Range::full(), Range::full());
  EXPECT_EQ(ctx.n(0), 1081);
  VectorCFR solver(ctx, {CFRConfig::dcfr()});
  solver.iterate(10);
  const double e10 = ctx.exploitability(solver.average_policy()).exploitability_pct_pot;
  solver.iterate(290);
  const double e300 = ctx.exploitability(solver.average_policy()).exploitability_pct_pot;
  EXPECT_TRUE(e300 < e10);
  EXPECT_TRUE(e300 < 0.5);  // below half a percent of the pot
}

AEGIS_TEST(resolving_gadget_with_equilibrium_constraints_is_safe) {
  // Solve a river game, then re-solve player 0's strategy from player 0's
  // range only, constrained by player 1's equilibrium counterfactual values.
  // The re-solved player-0 strategy must not be more exploitable than the
  // original (up to solver precision).
  auto cfg = river_config("Qs9h7d4c2c", 100, 100);
  auto tree = PublicTree::build(cfg);
  const auto r0 = Range::parse("QQ+,AQ,KQ,99,77,A9s+,T8s,65s,JTs");
  const auto r1 = Range::parse("AQ,KQ,QJ,JJ-TT,98s,A7s,87s,K9s");
  SubgameContext ctx(tree, r0, r1, 1, {false, true});
  VectorCFR blue(ctx, {CFRConfig::dcfr()});
  blue.iterate(1500);
  const auto bp = blue.average_policy();
  const auto bex = ctx.exploitability(bp);
  // Opponent constraint: counterfactual best-response values of P1 against P0's blueprint.
  const auto cbv = ctx.best_response(bp, 1).root_cfv[1];
  VectorCFRConfig rc{CFRConfig::dcfr()};
  rc.gadget.enabled = true;
  rc.gadget.opponent = 1;
  rc.gadget.constraint = cbv;
  VectorCFR re(ctx, rc);
  re.iterate(1500);
  auto rp = re.average_policy();
  // Combine: re-solved P0 with blueprint P1 (exploitability of P0's part).
  auto combined = bp;
  for (int n = 0; n < tree.num_nodes(); ++n)
    if (tree.nodes[n].type == PNodeType::kDecision && tree.nodes[n].player == 0)
      combined.strategy[tree.nodes[n].decision_index] = rp.strategy[tree.nodes[n].decision_index];
  const double br1_blue = ctx.best_response(bp, 1).ev[1];
  const double br1_re = ctx.best_response(combined, 1).ev[1];
  EXPECT_TRUE(br1_re <= br1_blue + 0.002 * cfg.pot);
  const auto reach = re.gadget_root_reach();
  for (double x : reach) EXPECT_TRUE(x >= 0.0 && x <= 1.0);
  EXPECT_TRUE(bex.exploitability_pct_pot < 0.2);
}
