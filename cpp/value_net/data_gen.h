// Solver-generated training data for the counterfactual value network.
//
// Each sample is a river subgame: a board (a fixed turn board plus a random
// river card, or a fully random board), random strength-correlated ranges for
// both players, and a pot/stack configuration. The subgame is solved with
// DCFR; targets are the normalised per-hand counterfactual values
//     v_p(h) = cfv_p(h) / (pot * Σ r_o)
// and the per-sample exploitability (% pot) records target quality.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "poker/betting_rules/betting.h"
#include "solvers/common/regret_rules.h"

namespace aegis::nn {

struct RiverSampleConfig {
  std::vector<poker::Card> turn_board;               // empty -> random 5-card boards
  std::vector<std::pair<double, double>> pot_stack;  // sampled uniformly
  poker::StreetSizes river_sizes;
  int iterations = 300;
  CFRConfig cfr = CFRConfig::dcfr();
  double uniform_mix_prob = 0.15;  // chance of blending a range with uniform
  uint64_t seed = 1;
  int threads = 1;
};

struct RiverSamples {
  int n = 0;
  std::vector<float> features;     // n x kInputSize
  std::vector<float> targets;      // n x kOutputSize
  std::vector<float> mask;         // n x kOutputSize (1 for live combos)
  std::vector<int> boards;         // n x 5 cards
  std::vector<double> exploitability_pct_pot;
  std::vector<double> pot, stack;
};

RiverSamples generate_river_samples(const RiverSampleConfig& cfg, int n);

}  // namespace aegis::nn
