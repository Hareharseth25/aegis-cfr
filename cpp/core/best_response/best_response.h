// Exact best responses and NashConv on a materialised game tree.
//
// The best response BR_i(σ_{-i}) is computed over player i's INFORMATION SETS:
// at every I ∈ I_i it picks the action maximising
//     Σ_{h ∈ I} π^σ_{-i}(h) * v_i(h*a)
// where π_{-i} is the opponent-and-chance reach and v_i is the value under the
// best response below. It never conditions on the opponent's hidden cards, so
// it is a legal strategy of the game, not an omniscient oracle.
//
//   NashConv(σ) = Σ_i [ u_i(BR_i(σ_{-i}), σ_{-i}) - u_i(σ) ]
//   exploitability(σ) = NashConv(σ) / 2      (two-player zero-sum convention)
#pragma once

#include <array>
#include <vector>

#include "core/game_tree/game_tree.h"
#include "core/policy/tabular_policy.h"

namespace aegis {

struct BestResponseResult {
  int player = 0;
  double value = 0.0;            // u_i(BR_i(σ_{-i}), σ_{-i})
  TabularPolicy policy;          // σ with player i's infosets replaced by the pure BR
  std::vector<int> best_action;  // per infoset of player i (index into its actions), else -1
  // Per-infoset audit: counterfactual action values Σ_h π_{-i}(h) v(h*a) under
  // the best response (slot-indexed) and the counterfactual value of σ_i at
  // that infoset. Their difference is how much the BR gains at that infoset.
  std::vector<double> cf_action_value;
  std::vector<double> cf_policy_value;   // per infoset
  std::vector<double> opponent_reach;    // per node: π_c(h) π_{-i}(h)
  std::vector<double> node_value;        // per node: v_i(h) under (BR_i, σ_{-i})
};

// Expected utilities (u_0, u_1) when both players follow `policy`.
std::array<double, 2> expected_values(const GameTree& tree, const TabularPolicy& policy);

BestResponseResult best_response(const GameTree& tree, const TabularPolicy& policy, int br_player);

struct ExploitabilityReport {
  double br_value[2] = {0, 0};   // value of each player's best response
  double policy_value[2] = {0, 0};
  double nash_conv = 0.0;
  double exploitability = 0.0;   // nash_conv / 2
};
ExploitabilityReport exploitability_report(const GameTree& tree, const TabularPolicy& policy);
double nash_conv(const GameTree& tree, const TabularPolicy& policy);
double exploitability(const GameTree& tree, const TabularPolicy& policy);

// Reach probabilities π_i(h) for each node under `policy` (player 0, player 1,
// chance), computed top-down.
struct ReachProbabilities {
  std::vector<double> player[2];
  std::vector<double> chance;
};
ReachProbabilities compute_reach(const GameTree& tree, const TabularPolicy& policy);

// Exact instantaneous counterfactual regrets of player i at σ, per action slot:
//   r(I, a) = Σ_{h ∈ I} π^σ_{-i}(h) * (v_i(h*a) - v_i(h)).
// Ground truth for measuring the bias/variance of sampled estimators.
std::vector<double> counterfactual_regrets(const GameTree& tree, const TabularPolicy& policy, int player);

}  // namespace aegis
