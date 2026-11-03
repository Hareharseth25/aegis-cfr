// Full-traversal tabular CFR on a materialised GameTree.
//
// One engine implements vanilla CFR, CFR+, DCFR and Linear CFR through
// CFRConfig (see solvers/common/regret_rules.h). Counterfactual values are
// weighted by the OPPONENT's and CHANCE's reach, never by the acting player's
// own reach; the acting player's reach only weights the average strategy.
//
// Per iteration and updated player i:
//   1. traverse the tree with σ^t, accumulating instantaneous regrets
//        r(I,a) += π_{-i}(h) * (v_i(h*a) - v_i(h))   for every h ∈ I
//      into a buffer, plus the average-strategy contribution
//        S(I,a) += w_t * π_i(h) * σ^t(I,a);
//   2. apply the regret rule (vanilla / CFR+ clip / DCFR discount) to every
//      information set of player i;
//   3. recompute σ_i by regret matching.
// With alternating updates player 1's traversal already sees player 0's new
// strategy, matching the standard alternating CFR / CFR+ formulation.
#pragma once

#include <cstdint>
#include <vector>

#include "core/game_tree/game_tree.h"
#include "core/policy/tabular_policy.h"
#include "solvers/common/regret_rules.h"

namespace aegis {

class TabularCFR {
 public:
  TabularCFR(const GameTree& tree, CFRConfig config);

  void iterate(int n = 1);
  int iteration() const { return t_; }

  TabularPolicy average_policy() const;
  TabularPolicy current_policy() const;

  const std::vector<double>& cumulative_regret() const { return regret_; }
  const std::vector<double>& strategy_sum() const { return strategy_sum_; }
  const CFRConfig& config() const { return cfg_; }
  const GameTree& tree() const { return tree_; }

  // Work counter: number of node visits performed so far.
  int64_t nodes_touched() const { return nodes_touched_; }

  // Optional warm start (e.g. from a previous solve on the same tree).
  void set_regrets(const std::vector<double>& r);

 private:
  double traverse(int h, int i, double reach0, double reach1, double reach_c);
  void traverse_both(int h, double reach0, double reach1, double reach_c, double out[2]);
  void finish_player(int i);

  const GameTree& tree_;
  CFRConfig cfg_;
  int t_ = 0;
  std::vector<double> regret_;
  std::vector<double> inst_regret_;
  std::vector<double> strategy_sum_;
  std::vector<double> sigma_;
  int64_t nodes_touched_ = 0;
  double avg_w_ = 1.0;  // average-strategy weight of the current iteration
};

}  // namespace aegis
