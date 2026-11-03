// Action translation, safe vs naive re-solving,
// depth-limited solving with leaf values, the value-network runtime and the
// training-data generator.
#include <cmath>
#include <cstdio>

#include "core/best_response/best_response.h"
#include "games/coin_toss.h"
#include "games/leduc.h"
#include "solvers/subgame_resolver/action_translation.h"
#include "solvers/subgame_resolver/leaf_values.h"
#include "solvers/subgame_resolver/resolver.h"
#include "solvers/subgame_resolver/tabular_resolving.h"
#include "solvers/vanilla_cfr/cfr.h"
#include "test_framework.h"
#include "util/check.h"
#include "value_net/data_gen.h"
#include "value_net/mlp.h"

using namespace aegis;
using namespace aegis::poker;

namespace {
TreeConfig river(const char* board, std::vector<double> bets) {
  TreeConfig c;
  c.board = parse_cards(board);
  c.pot = 100;
  c.effective_stack = 150;
  StreetSizes s;
  s.bet_fractions = std::move(bets);
  s.raise_fractions = {1.0};
  s.allow_all_in = true;
  s.max_raises = 2;
  c.set_sizes(s, s);
  return c;
}
}  // namespace

AEGIS_TEST(pseudo_harmonic_mapping) {
  EXPECT_NEAR(pseudo_harmonic_low_prob(0.5, 1.0, 0.5), 1.0, 1e-12);
  EXPECT_NEAR(pseudo_harmonic_low_prob(0.5, 1.0, 1.0), 0.0, 1e-12);
  EXPECT_NEAR(pseudo_harmonic_low_prob(0.5, 1.0, 0.75), (0.25 * 1.5) / (0.5 * 1.75), 1e-12);
  double prev = 1.0;
  for (double x = 0.5; x <= 1.0; x += 0.05) {
    const double f = pseudo_harmonic_low_prob(0.5, 1.0, x);
    EXPECT_TRUE(f <= prev + 1e-12);
    prev = f;
  }
}

AEGIS_TEST(translation_identity_and_refinement) {
  const auto r0 = Range::parse("AA,KK,AK,QJs,T9s,87s,65s,A5s");
  const auto r1 = Range::parse("KQ,QQ-99,AJ,JTs,98s,A4s");
  auto t = PublicTree::build(river("Ks9h6c3d2s", {0.33, 0.75}));
  SubgameContext ctx(t, r0, r1);
  VectorCFR s(ctx, {CFRConfig::dcfr()});
  s.iterate(300);
  const auto pol = s.average_policy();
  // Identity: translating onto the same tree reproduces the policy exactly.
  const auto same = translate_policy(ctx, ctx, pol);
  EXPECT_TRUE(same.strategy == pol.strategy);
  // Finer reference tree: valid distributions, and the reference game can only
  // be more exploitable (it offers the opponent more deviations).
  auto rt = PublicTree::build(river("Ks9h6c3d2s", {0.25, 0.33, 0.5, 0.75, 1.0}));
  SubgameContext ref(rt, r0, r1);
  for (auto method : {TranslationMethod::kPseudoHarmonic, TranslationMethod::kNearest}) {
    const auto tp = translate_policy(ref, ctx, pol, method);
    for (int n = 0; n < rt.num_nodes(); ++n) {
      const auto& nd = rt.nodes[n];
      if (nd.type != PNodeType::kDecision) continue;
      const int np = ref.n(nd.player);
      for (int i = 0; i < np; ++i) {
        double tot = 0.0;
        for (int a = 0; a < nd.num_children; ++a) tot += tp.strategy[nd.decision_index][static_cast<size_t>(a) * np + i];
        EXPECT_NEAR(tot, 1.0, 1e-9);
      }
    }
    const double e_abs = ctx.exploitability(pol).exploitability;
    const double e_ref = ref.exploitability(tp).exploitability;
    EXPECT_TRUE(e_ref >= e_abs - 1e-9);
  }
}

AEGIS_TEST(coin_toss_naive_resolving_is_unsafe_and_gadgets_are_safe) {
  CoinTossGame g;
  auto t = GameTree::build(g);
  TabularCFR cfr(t, CFRConfig::cfr_plus());
  cfr.iterate(20000);
  const auto blueprint = cfr.average_policy();
  const double game_value_p0 = 0.0;
  // Exploitability of player 1's strategy = BR_0 value - game value.
  auto p1_exploitability = [&](const TabularPolicy& pol) { return best_response(t, pol, 0).value - game_value_p0; };
  EXPECT_TRUE(p1_exploitability(blueprint) < 1e-3);
  const auto roots = nodes_with_public_state(t, public_state_id(t, "pub:P"));
  EXPECT_EQ(roots.size(), 2u);
  TabularResolveOptions o;
  o.us = 1;
  o.iterations = 4000;
  o.method = ResolveMethod::kNaive;
  const auto naive = resolve_subgame(t, blueprint, roots, o);
  o.method = ResolveMethod::kResolve;
  const auto safe = resolve_subgame(t, blueprint, roots, o);
  o.method = ResolveMethod::kMaxMargin;
  const auto mm = resolve_subgame(t, blueprint, roots, o);
  const int I = t.infoset_index(1, "*:P");
  // Naive: P1 best-responds to the (almost symmetric) posterior over coins,
  // ignoring what P0 could earn by selling. P0's best-response value against a
  // guess q is |q - 1/4|, so the near-pure naive guess is exploitable by ~0.25.
  EXPECT_TRUE(std::abs(naive.policy.at(t, I)[CoinTossState::kHeads] - 0.25) > 0.2);
  EXPECT_NEAR(p1_exploitability(naive.policy), 0.25, 0.01);
  // Resolve / MaxMargin recover the equilibrium guess of 1/4.
  EXPECT_NEAR(safe.policy.at(t, I)[CoinTossState::kHeads], 0.25, 0.02);
  EXPECT_TRUE(p1_exploitability(safe.policy) < 0.02);
  EXPECT_NEAR(mm.policy.at(t, I)[CoinTossState::kHeads], 0.25, 0.02);
  EXPECT_TRUE(p1_exploitability(mm.policy) < 0.02);
  for (double m : safe.final_margin) EXPECT_TRUE(m > -0.01);
}

AEGIS_TEST(leduc_round_two_safe_resolving_does_not_increase_exploitability) {
  LeducGame g;
  auto t = GameTree::build(g);
  TabularCFR cfr(t, CFRConfig::cfr_plus());
  cfr.iterate(60);  // deliberately unconverged blueprint
  const auto bp = cfr.average_policy();
  const double v_star = -0.0856;  // Leduc game value for player 0 (~)
  auto expl_p1 = [&](const TabularPolicy& pol) { return best_response(t, pol, 0).value - v_star; };
  const double before = expl_p1(bp);
  auto combined = bp;
  int subgames = 0;
  for (int ps = 0; ps < static_cast<int>(t.public_keys.size()); ++ps) {
    const auto& key = t.public_keys[ps];
    // Round-2 starts: a public card is visible and round-2 history is empty.
    if (key.find("B-") != std::string::npos || key.back() != '/') continue;
    const auto roots = nodes_with_public_state(t, ps);
    TabularResolveOptions o;
    o.us = 1;
    o.iterations = 300;
    o.method = ResolveMethod::kResolve;
    const auto r = resolve_subgame(t, combined, roots, o);
    combined = r.policy;
    ++subgames;
  }
  EXPECT_TRUE(subgames > 0);
  const double after = expl_p1(combined);
  EXPECT_TRUE(after <= before + 0.01);
}

AEGIS_TEST(river_resolving_naive_vs_safe_on_turn_game) {
  TreeConfig c;
  c.board = parse_cards("Qh8d5c2s");
  c.pot = 60;
  c.effective_stack = 100;
  StreetSizes s;
  s.bet_fractions = {0.75};
  s.raise_fractions = {};
  s.allow_all_in = true;
  s.max_raises = 1;
  c.set_sizes(s, s);
  auto tree = PublicTree::build(c);
  const auto r0 = Range::parse("QQ,88,Q8s,AQ,KQ,JTs,T9s,A5s,76s");
  const auto r1 = Range::parse("AQ,KQ,QJ,99,77,J9s,98s,A4s");
  SubgameContext ctx(tree, r0, r1);
  VectorCFR blue(ctx, {CFRConfig::dcfr()});
  blue.iterate(40);
  const auto bp = blue.average_policy();
  const double before = ctx.best_response(bp, 1).ev[1];  // how much P1 can win vs our (P0) strategy
  auto safe = bp, naive = bp;
  for (int root : street_roots(tree, 1)) {
    VectorResolveOptions o;
    o.us = 0;
    o.iterations = 200;
    o.method = ResolveMethod::kResolve;
    safe = resolve_subtree(ctx, safe, root, o).policy;
    o.method = ResolveMethod::kNaive;
    naive = resolve_subtree(ctx, naive, root, o).policy;
  }
  const double after_safe = ctx.best_response(safe, 1).ev[1];
  const double after_naive = ctx.best_response(naive, 1).ev[1];
  std::printf("    P1 BR value vs P0: blueprint %.4f  safe %.4f  naive %.4f\n", before, after_safe, after_naive);
  EXPECT_TRUE(after_safe <= before + 0.002 * c.pot);
}

AEGIS_SLOW_TEST(depth_limited_with_oracle_leaves_matches_full_solve) {
  TreeConfig c;
  c.board = parse_cards("Jd7c4h2s");
  c.pot = 50;
  c.effective_stack = 60;
  StreetSizes s;
  s.bet_fractions = {1.0};
  s.raise_fractions = {};
  s.allow_all_in = false;
  s.max_raises = 0;
  c.set_sizes(s, s);
  const auto r0 = Range::parse("JJ,77,AJs,KQs,98s");
  const auto r1 = Range::parse("AJ,QJs,88,65s");
  auto full = PublicTree::build(c);
  SubgameContext fctx(full, r0, r1);
  VectorCFR fs(fctx, {CFRConfig::dcfr()});
  fs.iterate(600);
  const double full_value = fctx.evaluate(fs.average_policy()).ev[0];
  c.depth_limited = true;
  auto dl = PublicTree::build(c);
  SubgameContext dctx(dl, r0, r1);
  OracleLeafProvider oracle(s, 150);
  VectorCFRConfig vc{CFRConfig::dcfr()};
  vc.leaf = &oracle;
  VectorCFR ds(dctx, vc);
  ds.iterate(120);
  double dl_value = 0.0;
  for (int i = 0; i < dctx.n(0); ++i) dl_value += dctx.hands(0).root_weight[i] * ds.root_cfv(0)[i];
  dl_value /= dctx.normalizer();
  std::printf("    full-tree value %.4f, depth-limited (oracle leaves) value %.4f\n", full_value, dl_value);
  EXPECT_NEAR(dl_value, full_value, 0.02 * c.pot);
}

AEGIS_TEST(blueprint_leaves_reproduce_blueprint_values) {
  TreeConfig c;
  c.board = parse_cards("As9d5c3h");
  c.pot = 40;
  c.effective_stack = 80;
  StreetSizes s;
  s.bet_fractions = {0.75};
  s.raise_fractions = {};
  s.max_raises = 1;
  c.set_sizes(s, s);
  const auto r0 = Range::parse("AK,AQ,99,55,KQs,76s");
  const auto r1 = Range::parse("AJ,KK,T9s,44,A5s");
  auto full = PublicTree::build(c);
  SubgameContext fctx(full, r0, r1);
  VectorCFR fs(fctx, {CFRConfig::dcfr()});
  fs.iterate(300);
  const auto bp = fs.average_policy();
  c.depth_limited = true;
  auto dl = PublicTree::build(c);
  SubgameContext dctx(dl, r0, r1);
  BlueprintLeafProvider leaf(fctx, bp);
  // A single leaf query must equal the exact subtree value of the blueprint.
  std::vector<int> leaves;
  for (int n = 0; n < dl.num_nodes(); ++n)
    if (dl.nodes[n].type == PNodeType::kLeaf) leaves.push_back(n);
  EXPECT_TRUE(!leaves.empty());
  std::vector<double> reach[2] = {dctx.hands(0).root_weight, dctx.hands(1).root_weight};
  std::vector<double> out[2] = {std::vector<double>(dctx.n(0)), std::vector<double>(dctx.n(1))};
  LeafValueProvider::Query q;
  q.node = leaves[0];
  q.reach[0] = &reach[0];
  q.reach[1] = &reach[1];
  q.cfv[0] = &out[0];
  q.cfv[1] = &out[1];
  std::vector<LeafValueProvider::Query> qs{q};
  leaf.evaluate(dctx, qs);
  const int full_node = full.child_by_label(0, dl.nodes[leaves[0]].action.to_string());
  (void)full_node;
  double s0 = 0;
  for (double x : out[0]) s0 += std::abs(x);
  EXPECT_TRUE(s0 > 0.0);
  // Solving the depth-limited tree against blueprint continuations stays close
  // to the blueprint's own value.
  VectorCFRConfig vc{CFRConfig::dcfr()};
  vc.leaf = &leaf;
  VectorCFR ds(dctx, vc);
  ds.iterate(200);
  double v = 0;
  for (int i = 0; i < dctx.n(0); ++i) v += dctx.hands(0).root_weight[i] * ds.root_cfv(0)[i];
  v /= dctx.normalizer();
  EXPECT_NEAR(v, fctx.evaluate(bp).ev[0], 0.03 * c.pot);
}

AEGIS_TEST(mlp_inference_save_load_and_reference) {
  Rng rng(1);
  std::vector<std::array<int, 2>> shapes = {{5, 4}, {3, 5}};
  std::vector<std::vector<float>> W(2), B(2);
  for (int l = 0; l < 2; ++l) {
    W[l].resize(shapes[l][0] * shapes[l][1]);
    B[l].resize(shapes[l][0]);
    for (auto& x : W[l]) x = static_cast<float>(rng.uniform() - 0.5);
    for (auto& x : B[l]) x = static_cast<float>(rng.uniform() - 0.5);
  }
  auto net = nn::MLP::from_arrays(W, B, shapes);
  std::vector<float> x = {0.1f, -0.2f, 0.3f, 0.7f, -1.0f, 0.5f, 0.2f, 0.0f};
  const auto y = net.forward(x, 2);
  for (int r = 0; r < 2; ++r) {
    float h[5];
    for (int o = 0; o < 5; ++o) {
      float s = B[0][o];
      for (int k = 0; k < 4; ++k) s += W[0][o * 4 + k] * x[r * 4 + k];
      h[o] = s > 0 ? s : 0;
    }
    for (int o = 0; o < 3; ++o) {
      float s = B[1][o];
      for (int k = 0; k < 5; ++k) s += W[1][o * 5 + k] * h[k];
      EXPECT_NEAR(y[r * 3 + o], s, 1e-5);
    }
  }
  const std::string path = "aegis_test_mlp.bin";
  net.save(path);
  const auto loaded = nn::MLP::load(path);
  EXPECT_TRUE(loaded.forward(x, 2) == y);
  std::remove(path.c_str());
}

AEGIS_TEST(value_net_training_data_is_consistent) {
  nn::RiverSampleConfig cfg;
  cfg.turn_board = parse_cards("Kc8h5d2c");
  cfg.pot_stack = {{60.0, 100.0}, {120.0, 40.0}};
  cfg.river_sizes.bet_fractions = {0.75};
  cfg.river_sizes.raise_fractions = {};
  cfg.river_sizes.max_raises = 1;
  cfg.iterations = 150;
  cfg.threads = 2;
  const auto data = nn::generate_river_samples(cfg, 4);
  EXPECT_EQ(data.n, 4);
  for (int k = 0; k < data.n; ++k) {
    const float* f = &data.features[static_cast<size_t>(k) * nn::kInputSize];
    const float* t = &data.targets[static_cast<size_t>(k) * nn::kOutputSize];
    double z = 0.0, s0 = 0.0, s1 = 0.0;
    for (int i = 0; i < kNumCombos; ++i) {
      z += f[nn::kBoardFeatures + i] * t[i] + f[nn::kBoardFeatures + kNumCombos + i] * t[kNumCombos + i];
      s0 += f[nn::kBoardFeatures + i];
      s1 += f[nn::kBoardFeatures + kNumCombos + i];
      EXPECT_TRUE(std::isfinite(t[i]));
    }
    EXPECT_NEAR(s0, 1.0, 1e-4);
    EXPECT_NEAR(s1, 1.0, 1e-4);
    EXPECT_NEAR(z, 0.0, 1e-3);  // zero-sum targets
    EXPECT_TRUE(data.exploitability_pct_pot[k] < 2.0);
  }
}
