// Core tests: game trees, exact best responses, CFR-family
// convergence to the known Kuhn equilibrium, and sampled CFR.
#include <map>

#include "core/best_response/best_response.h"
#include "core/game_tree/game_tree.h"
#include "games/coin_toss.h"
#include "games/kuhn.h"
#include "games/leduc.h"
#include "solvers/mccfr/mccfr.h"
#include "solvers/vanilla_cfr/cfr.h"
#include "solvers/vanilla_cfr/naive_cfr.h"
#include "test_framework.h"
#include "util/check.h"

using namespace aegis;

namespace {

// The closed-form Kuhn equilibrium family (player 0 parameter a ∈ [0, 1/3]).
TabularPolicy kuhn_equilibrium(const GameTree& t, double a) {
  TabularPolicy p(t);
  auto set = [&](int player, const std::string& key, double bet) {
    const int I = t.infoset_index(player, key);
    AEGIS_CHECK_MSG(I >= 0, "missing " << key);
    p.at(t, I)[0] = 1.0 - bet;  // pass
    p.at(t, I)[1] = bet;        // bet / call
  };
  // Player 0 first action: J bets a, Q checks, K bets 3a.
  set(0, "0", a);
  set(0, "1", 0.0);
  set(0, "2", 3.0 * a);
  // Player 0 facing a bet after checking: J folds, Q calls a + 1/3, K calls.
  set(0, "0pb", 0.0);
  set(0, "1pb", a + 1.0 / 3.0);
  set(0, "2pb", 1.0);
  // Player 1 facing a bet: J folds, Q calls 1/3, K calls.
  set(1, "0b", 0.0);
  set(1, "1b", 1.0 / 3.0);
  set(1, "2b", 1.0);
  // Player 1 facing a check: J bets 1/3, Q checks, K bets.
  set(1, "0p", 1.0 / 3.0);
  set(1, "1p", 0.0);
  set(1, "2p", 1.0);
  return p;
}

// Imperfect-recall test game: player 0 acts twice but forgets the first action.
class ForgetfulState final : public State {
 public:
  bool is_terminal() const override { return h_.size() == 2; }
  int current_player() const override { return is_terminal() ? kTerminalPlayer : 0; }
  std::vector<Action> legal_actions() const override { return {0, 1}; }
  std::vector<std::pair<Action, double>> chance_outcomes() const override { return {}; }
  std::unique_ptr<State> child(Action a) const override {
    auto s = std::make_unique<ForgetfulState>(*this);
    s->h_.push_back(a);
    return s;
  }
  std::string information_state_key(int) const override { return h_.empty() ? "first" : "second"; }
  std::array<double, 2> returns() const override {
    const double u = h_[0] == h_[1] ? 1.0 : -1.0;
    return {u, -u};
  }
  std::string action_to_string(Action a) const override { return std::to_string(a); }
  std::string to_string() const override { return "forgetful"; }

 private:
  std::vector<Action> h_;
};
class ForgetfulGame final : public Game {
 public:
  std::unique_ptr<State> initial_state() const override { return std::make_unique<ForgetfulState>(); }
  std::string name() const override { return "forgetful"; }
};

}  // namespace

AEGIS_TEST(kuhn_tree_shape) {
  KuhnGame g;
  auto t = GameTree::build(g);
  EXPECT_EQ(t.num_nodes(), 58);  // matches OpenSpiel's history count
  EXPECT_EQ(t.num_infosets(), 12);
  EXPECT_EQ(t.num_terminals(), 30);
  EXPECT_TRUE(t.zero_sum);
}

AEGIS_TEST(leduc_tree_shape) {
  LeducGame g;
  auto t = GameTree::build(g);
  EXPECT_EQ(t.num_nodes(), 9457);  // matches OpenSpiel's history count
  EXPECT_EQ(t.num_infosets(), 936);
  EXPECT_TRUE(t.zero_sum);
}

AEGIS_TEST(perfect_recall_violation_is_detected) {
  ForgetfulGame g;
  EXPECT_THROWS(GameTree::build(g));
  GameTree::BuildOptions o;
  o.check_perfect_recall = false;
  auto t = GameTree::build(g, o);
  EXPECT_EQ(t.num_infosets(), 2);
}

AEGIS_TEST(kuhn_closed_form_equilibrium_is_exact) {
  KuhnGame g;
  auto t = GameTree::build(g);
  for (double a : {0.0, 0.1, 0.2, 1.0 / 3.0}) {
    auto pol = kuhn_equilibrium(t, a);
    pol.validate(t);
    const auto ev = expected_values(t, pol);
    EXPECT_NEAR(ev[0], KuhnGame::kGameValueP0, 1e-12);
    const auto rep = exploitability_report(t, pol);
    EXPECT_NEAR(rep.nash_conv, 0.0, 1e-12);
    // Best responses achieve exactly the game value.
    EXPECT_NEAR(rep.br_value[0], -1.0 / 18.0, 1e-12);
    EXPECT_NEAR(rep.br_value[1], 1.0 / 18.0, 1e-12);
  }
}

AEGIS_TEST(kuhn_uniform_policy_nash_conv) {
  KuhnGame g;
  auto t = GameTree::build(g);
  // Also asserted against OpenSpiel in tests/python/test_openspiel_oracle.py.
  const auto rep = exploitability_report(t, TabularPolicy::uniform(t));
  EXPECT_NEAR(rep.nash_conv, 0.9166666666666666, 1e-12);
  EXPECT_NEAR(rep.exploitability, 0.4583333333333333, 1e-12);
}

AEGIS_TEST(best_response_is_information_set_constrained) {
  // The BR policy must be a legal strategy: evaluating it with the ordinary
  // expected-value routine must reproduce the BR value, and no random policy
  // may beat it.
  LeducGame g;
  auto t = GameTree::build(g);
  Rng rng(7);
  for (int trial = 0; trial < 5; ++trial) {
    auto pol = TabularPolicy::random(t, rng);
    for (int i = 0; i < 2; ++i) {
      auto br = best_response(t, pol, i);
      br.policy.validate(t);
      EXPECT_NEAR(expected_values(t, br.policy)[i], br.value, 1e-9);
      for (int k = 0; k < 20; ++k) {
        auto other = TabularPolicy::random(t, rng, 0.3);
        auto mixed = pol;
        mixed.set_player_from(t, other, i);
        EXPECT_TRUE(expected_values(t, mixed)[i] <= br.value + 1e-9);
      }
    }
  }
}

AEGIS_TEST(best_response_audit_values_are_consistent) {
  // Σ over the BR player's root-level infosets of the counterfactual BR value
  // equals the BR value (Kuhn: player 0's first decision covers all mass).
  KuhnGame g;
  auto t = GameTree::build(g);
  auto pol = TabularPolicy::uniform(t);
  auto br = best_response(t, pol, 0);
  double total = 0.0;
  for (const std::string key : {"0", "1", "2"}) {
    const int I = t.infoset_index(0, key);
    const auto& info = t.infosets[I];
    total += br.cf_action_value[info.offset + br.best_action[I]];
  }
  EXPECT_NEAR(total, br.value, 1e-12);
}

AEGIS_TEST(vanilla_cfr_converges_to_kuhn_equilibrium) {
  KuhnGame g;
  auto t = GameTree::build(g);
  for (bool alternating : {true, false}) {
    TabularCFR cfr(t, CFRConfig::vanilla(alternating));
    cfr.iterate(20000);
    auto avg = cfr.average_policy();
    avg.validate(t);
    const double nc = nash_conv(t, avg);
    // Measured: alternating 1.4e-4, simultaneous 3.3e-3 after 20k iterations.
    EXPECT_TRUE(nc < (alternating ? 5e-4 : 5e-3));
    EXPECT_NEAR(expected_values(t, avg)[0], KuhnGame::kGameValueP0, 3e-3);
    // Player 1's equilibrium strategy is unique.
    auto prob = [&](int p, const char* key, int a) { return avg.at(t, t.infoset_index(p, key))[a]; };
    EXPECT_NEAR(prob(1, "1b", 1), 1.0 / 3.0, 0.02);
    EXPECT_NEAR(prob(1, "0p", 1), 1.0 / 3.0, 0.02);
    EXPECT_NEAR(prob(1, "2b", 1), 1.0, 0.01);
    EXPECT_NEAR(prob(1, "0b", 1), 0.0, 0.01);
    // Player 0 lies in the equilibrium family: K-bet = 3*J-bet, Q-call = J-bet + 1/3.
    const double a = prob(0, "0", 1);
    EXPECT_TRUE(a >= -1e-9 && a <= 1.0 / 3.0 + 0.02);
    EXPECT_NEAR(prob(0, "2", 1), 3.0 * a, 0.05);
    EXPECT_NEAR(prob(0, "1pb", 1), a + 1.0 / 3.0, 0.05);
  }
}

AEGIS_TEST(cfr_variants_converge_on_kuhn_and_leduc) {
  KuhnGame kg;
  LeducGame lg;
  auto kt = GameTree::build(kg);
  auto lt = GameTree::build(lg);
  for (const auto& cfg : {CFRConfig::cfr_plus(), CFRConfig::dcfr(), CFRConfig::linear()}) {
    TabularCFR k(kt, cfg);
    k.iterate(2000);
    EXPECT_TRUE(nash_conv(kt, k.average_policy()) < 1e-3);
    TabularCFR l(lt, cfg);
    l.iterate(500);
    const double nc = nash_conv(lt, l.average_policy());
    // Measured after 500 iterations: CFR+ 1.9e-3, DCFR 1.0e-3, LCFR 1.9e-2.
    EXPECT_TRUE(nc < 0.03);
    // Leduc's game value for player 0 is about -0.0856.
    EXPECT_NEAR(expected_values(lt, l.average_policy())[0], -0.0856, 0.01);
  }
}

AEGIS_TEST(naive_cfr_matches_flat_cfr_iterates) {
  for (int which = 0; which < 2; ++which) {
    std::unique_ptr<Game> g;
    if (which == 0) g = std::make_unique<KuhnGame>();
    else g = std::make_unique<LeducGame>();
    auto t = GameTree::build(*g);
    TabularCFR flat(t, CFRConfig::vanilla(true));
    NaiveCFR naive(*g);
    flat.iterate(50);
    naive.iterate(50);
    auto avg = flat.average_policy();
    double max_diff = 0.0;
    for (int I = 0; I < t.num_infosets(); ++I) {
      const auto& info = t.infosets[I];
      const auto s = naive.average_strategy(info.player, info.key);
      for (int a = 0; a < info.num_actions; ++a)
        max_diff = std::max(max_diff, std::abs(s[a] - avg.probs[info.offset + a]));
    }
    EXPECT_TRUE(max_diff < 1e-12);
  }
}

AEGIS_TEST(nash_conv_decreases_with_iterations) {
  LeducGame g;
  auto t = GameTree::build(g);
  TabularCFR cfr(t, CFRConfig::cfr_plus());
  double prev = 1e9;
  for (int k = 0; k < 5; ++k) {
    cfr.iterate(k == 0 ? 10 : 10 * (1 << k));
    const double nc = nash_conv(t, cfr.average_policy());
    EXPECT_TRUE(nc < prev);
    prev = nc;
  }
}

AEGIS_TEST(external_sampling_mccfr_converges) {
  KuhnGame kg;
  auto kt = GameTree::build(kg);
  MCCFRConfig c;
  c.scheme = MCCFRScheme::kExternal;
  c.seed = 3;
  MCCFR es(kt, c);
  es.iterate(50000);
  EXPECT_TRUE(nash_conv(kt, es.average_policy()) < 0.02);

  LeducGame lg;
  auto lt = GameTree::build(lg);
  MCCFR les(lt, c);
  les.iterate(20000);
  EXPECT_TRUE(nash_conv(lt, les.average_policy()) < 0.5);  // measured 0.36
}

AEGIS_TEST(outcome_sampling_mccfr_converges) {
  KuhnGame kg;
  auto kt = GameTree::build(kg);
  MCCFRConfig c;
  c.scheme = MCCFRScheme::kOutcome;
  c.seed = 5;
  MCCFR os(kt, c);
  os.iterate(200000);
  EXPECT_TRUE(nash_conv(kt, os.average_policy()) < 0.05);
}

namespace {
class OnPolicySampler final : public SamplingPolicy {
 public:
  void sampling_distribution(int, int, std::span<const double> p, std::span<double> q) override {
    for (size_t k = 0; k < p.size(); ++k) q[k] = p[k];
  }
};
class SkewedSampler final : public SamplingPolicy {
 public:
  void sampling_distribution(int, int, std::span<const double> p, std::span<double> q) override {
    // Heavily distorted but full-support sampling: q prop. to p + 0.5*(k+1).
    double s = 0.0;
    for (size_t k = 0; k < p.size(); ++k) {
      q[k] = p[k] > 0.0 ? p[k] + 0.5 * (k + 1) : 0.0;
      s += q[k];
    }
    for (size_t k = 0; k < p.size(); ++k) q[k] /= s;
  }
};
}  // namespace

AEGIS_TEST(importance_sampling_with_on_policy_q_equals_external_sampling) {
  LeducGame g;
  auto t = GameTree::build(g);
  MCCFRConfig c;
  c.seed = 11;
  c.scheme = MCCFRScheme::kExternal;
  MCCFR es(t, c);
  c.scheme = MCCFRScheme::kImportance;
  MCCFR is(t, c);
  OnPolicySampler s;
  is.set_sampling_policy(&s);
  es.iterate(300);
  is.iterate(300);
  EXPECT_TRUE(es.cumulative_regret() == is.cumulative_regret());
  EXPECT_TRUE(es.strategy_sum() == is.strategy_sum());
}

AEGIS_TEST(importance_sampling_regret_estimator_is_unbiased) {
  // At a fixed σ (no updates), the mean of many sampled regret estimates must
  // equal the exact counterfactual regret from a full traversal.
  KuhnGame g;
  auto t = GameTree::build(g);
  TabularCFR exact(t, CFRConfig::vanilla(true));
  // Exact instantaneous regrets for player 0 at σ = uniform: one traversal.
  exact.iterate(1);  // regrets after iteration 1 for both players at uniform σ (player 0 part)
  std::vector<double> exact_r0(t.total_actions, 0.0);
  for (const auto& I : t.infosets)
    if (I.player == 0)
      for (int a = 0; a < I.num_actions; ++a) exact_r0[I.offset + a] = exact.cumulative_regret()[I.offset + a];

  MCCFRConfig c;
  c.scheme = MCCFRScheme::kImportance;
  c.seed = 99;
  MCCFR is(t, c);
  SkewedSampler s;
  is.set_sampling_policy(&s);
  const int n = 400000;
  std::vector<double> mean(t.total_actions, 0.0);
  for (int k = 0; k < n; ++k) {
    auto r = is.sample_regret_estimate(0);
    for (int x = 0; x < t.total_actions; ++x) mean[x] += r[x] / n;
  }
  for (const auto& I : t.infosets)
    if (I.player == 0)
      for (int a = 0; a < I.num_actions; ++a) EXPECT_NEAR(mean[I.offset + a], exact_r0[I.offset + a], 0.01);
}

AEGIS_TEST(parallel_mccfr_is_deterministic_and_converges) {
  LeducGame g;
  auto t = GameTree::build(g);
  MCCFRConfig c;
  c.seed = 21;
  MCCFR a(t, c), b(t, c);
  a.iterate_parallel(200, 25, 4);
  b.iterate_parallel(200, 25, 4);
  EXPECT_TRUE(a.cumulative_regret() == b.cumulative_regret());
  // 200 batches x 100 traversals: measured 0.41 (sequential 20k iterations: 0.43).
  EXPECT_TRUE(nash_conv(t, a.average_policy()) < 0.6);
}

AEGIS_TEST(coin_toss_equilibrium) {
  CoinTossGame g;
  auto t = GameTree::build(g);
  TabularCFR cfr(t, CFRConfig::cfr_plus());
  cfr.iterate(5000);
  auto avg = cfr.average_policy();
  EXPECT_TRUE(nash_conv(t, avg) < 1e-3);
  const int I = t.infoset_index(1, "*:P");
  EXPECT_NEAR(avg.at(t, I)[CoinTossState::kHeads], 0.25, 0.01);
  EXPECT_NEAR(expected_values(t, avg)[0], 0.0, 1e-3);
}
