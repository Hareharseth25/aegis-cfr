// Adaptive computation: adaptive sampling stays unbiased, adaptive MCCFR converges,
// budget allocation respects its budget, and refinement diagnostics are sane.
#include <cmath>

#include "adaptive/allocation/adaptive_sampler.h"
#include "adaptive/allocation/budget_allocator.h"
#include "adaptive/refinement/action_refinement.h"
#include "core/best_response/best_response.h"
#include "games/kuhn.h"
#include "games/leduc.h"
#include "solvers/mccfr/mccfr.h"
#include "test_framework.h"

using namespace aegis;
using namespace aegis::poker;

AEGIS_TEST(adaptive_sampling_estimator_is_unbiased) {
  KuhnGame g;
  auto t = GameTree::build(g);
  const auto exact = counterfactual_regrets(t, TabularPolicy::uniform(t), 0);
  for (auto mode : {AllocationMode::kRandom, AllocationMode::kReach, AllocationMode::kAdaptive}) {
    MCCFRConfig c;
    c.scheme = MCCFRScheme::kImportance;
    c.seed = 77;
    MCCFR m(t, c);
    AdaptiveSampler s(t, m, mode);
    m.set_sampling_policy(&s);
    const int n = 300000;
    std::vector<double> mean(t.total_actions, 0.0);
    for (int k = 0; k < n; ++k) {
      s.begin_traversal(0);
      const auto r = m.sample_regret_estimate(0);
      for (int x = 0; x < t.total_actions; ++x) mean[x] += r[x] / n;
    }
    for (const auto& I : t.infosets)
      if (I.player == 0)
        for (int a = 0; a < I.num_actions; ++a) EXPECT_NEAR(mean[I.offset + a], exact[I.offset + a], 0.012);
  }
}

AEGIS_TEST(adaptive_mccfr_converges_on_leduc) {
  LeducGame g;
  auto t = GameTree::build(g);
  for (auto mode : {AllocationMode::kExternal, AllocationMode::kAdaptive}) {
    MCCFRConfig c;
    c.scheme = MCCFRScheme::kImportance;
    c.seed = 5;
    MCCFR m(t, c);
    AdaptiveSampler s(t, m, mode);
    m.set_sampling_policy(&s);
    m.iterate(2000);
    const double early = nash_conv(t, m.average_policy());
    m.iterate(18000);
    const double late = nash_conv(t, m.average_policy());
    EXPECT_TRUE(late < early);
    EXPECT_TRUE(late < 0.8);
  }
}

AEGIS_TEST(budget_allocator_respects_budget) {
  std::vector<std::unique_ptr<PublicTree>> trees;
  std::vector<std::unique_ptr<SubgameContext>> ctxs;
  std::vector<BudgetSubgame> games;
  const char* boards[] = {"AsKh8c4d2s", "9h9c4d7s2c", "QdJdTc3s2h"};
  const double weights[] = {0.6, 0.3, 0.1};
  for (int k = 0; k < 3; ++k) {
    TreeConfig c;
    c.board = parse_cards(boards[k]);
    c.pot = 100;
    c.effective_stack = 100;
    StreetSizes s;
    s.bet_fractions = {0.75};
    s.raise_fractions = {};
    s.max_raises = 1;
    c.set_sizes(s, s);
    trees.push_back(std::make_unique<PublicTree>(PublicTree::build(c)));
    ctxs.push_back(std::make_unique<SubgameContext>(*trees.back(), Range::parse("TT+,AK,AQs,KQs,JTs"),
                                                    Range::parse("99+,AJ+,KQ,QJs,T9s")));
    games.push_back({ctxs.back().get(), weights[k]});
  }
  BudgetOptions o;
  o.total_work = 2e6;
  for (auto pol : {BudgetPolicy::kUniform, BudgetPolicy::kWeighted, BudgetPolicy::kAdaptive}) {
    const auto r = allocate_budget(games, pol, o);
    EXPECT_TRUE(r.work_used <= o.total_work * 1.0001);
    EXPECT_EQ(r.iterations.size(), 3u);
    EXPECT_TRUE(r.weighted_exploitability >= 0.0 && std::isfinite(r.weighted_exploitability));
  }
}

AEGIS_TEST(action_refiner_identity_and_candidates) {
  RefinementSpot spot;
  spot.base.board = parse_cards("Kd9s6h4c2c");
  spot.base.pot = 100;
  spot.base.effective_stack = 150;
  StreetSizes s;
  s.raise_fractions = {};
  s.allow_all_in = true;
  s.max_raises = 1;
  spot.base.set_sizes(s, s);
  spot.r0 = Range::parse("KK,99,66,AK,KQs,QJs,T8s,75s,A5s");
  spot.r1 = Range::parse("AK,KQ,KJs,QQ-TT,98s,87s");
  spot.reference_bets = {0.25, 0.5, 0.75, 1.0};
  spot.iterations = 300;
  ActionRefiner ref(spot);
  const auto full = ref.evaluate({spot.reference_bets, spot.reference_bets});
  EXPECT_NEAR(full.ref_exploitability_pct, full.abs_exploitability_pct, 1e-9);
  EXPECT_TRUE(full.candidates.empty());
  const std::array<std::vector<double>, 2> coarse_bets = {std::vector<double>{0.75}, std::vector<double>{0.75}};
  const auto coarse = ref.evaluate(coarse_bets);
  EXPECT_TRUE(!coarse.candidates.empty());
  for (const auto& c : coarse.candidates) EXPECT_TRUE(c.hole >= 0.0);
  EXPECT_TRUE(coarse.ref_exploitability_pct >= coarse.abs_exploitability_pct - 1e-9);
}

namespace {
struct BudgetFixture {
  std::unique_ptr<PublicTree> tree;
  std::unique_ptr<SubgameContext> ctx;
  BudgetFixture() {
    TreeConfig c;
    c.board = parse_cards("AsKh8c4d2s");
    c.pot = 100;
    c.effective_stack = 150;
    StreetSizes s;
    s.bet_fractions = {0.33, 0.75, 1.5};
    s.raise_fractions = {1.0};
    s.max_raises = 3;
    c.set_sizes(s, s);
    tree = std::make_unique<PublicTree>(PublicTree::build(c));
    ctx = std::make_unique<SubgameContext>(*tree, Range::parse("AA,KK"), Range::parse("QQ,JJ"));
  }
};
}  // namespace

AEGIS_TEST(budget_allocator_never_exceeds_tiny_or_boundary_budgets) {
  BudgetFixture f;
  std::vector<BudgetSubgame> games = {{f.ctx.get(), 1.0}};
  // Budget smaller than one iteration: nothing may run.
  for (auto pol : {BudgetPolicy::kUniform, BudgetPolicy::kWeighted, BudgetPolicy::kAdaptive}) {
    BudgetOptions o;
    o.total_work = 1.0;
    o.warmup_iterations = 2;
    const auto r = allocate_budget(games, pol, o);
    EXPECT_TRUE(r.work_used <= o.total_work);
    EXPECT_EQ(r.iterations[0], 0);
  }
  // Boundary case where chunk + refit measurement used to overshoot.
  const double per_iter = 2.0 * f.tree->num_nodes() * (f.ctx->n(0) + f.ctx->n(1));
  for (double budget : {18.0 * per_iter, 18.5 * per_iter, 19.0 * per_iter, 23.0 * per_iter}) {
    BudgetOptions o;
    o.total_work = budget;
    o.warmup_iterations = 2;
    o.chunk_iterations = 5;
    o.refit_every = 1;
    const auto r = allocate_budget(games, BudgetPolicy::kAdaptive, o);
    EXPECT_TRUE(r.work_used <= budget * (1 + 1e-12));
  }
  BudgetOptions bad;
  bad.chunk_iterations = 0;
  EXPECT_THROWS(allocate_budget(games, BudgetPolicy::kAdaptive, bad));
  bad = BudgetOptions{};
  bad.total_work = -1.0;
  EXPECT_THROWS(allocate_budget(games, BudgetPolicy::kUniform, bad));
}

AEGIS_TEST(native_accessors_reject_bad_indices) {
  BudgetFixture f;
  EXPECT_THROWS(f.ctx->hands(2));
  EXPECT_THROWS(f.ctx->hands(-1));
  EXPECT_THROWS(f.tree->node(-1));
  EXPECT_THROWS(f.tree->node(f.tree->num_nodes()));
  EXPECT_THROWS(f.tree->pot(-1));
  EXPECT_THROWS(f.tree->path_string(1 << 20));
  VectorCFR s(*f.ctx, {CFRConfig::dcfr()});
  EXPECT_THROWS(s.regrets(-1));
  EXPECT_THROWS(s.root_cfv(3));
  VectorPolicy wrong;
  EXPECT_THROWS(f.ctx->evaluate(wrong));
}
