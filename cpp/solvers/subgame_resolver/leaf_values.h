// Leaf value providers for depth-limited turn solving.
//
// At a depth-limited leaf (end of turn betting, chips behind) the river is
// replaced by values: for every river card c, the river subgame's per-hand
// counterfactual values given the ranges that reach the leaf, averaged as a
// chance node:
//     cfv_p(L, h) = (1/44) Σ_{c ∉ h} cfv_p^{river(L,c)}(h).
//
//  * NeuralLeafProvider    - learned values from the counterfactual value
//                            network, one batched inference per pass.
//  * BlueprintLeafProvider - values of following a blueprint river strategy
//                            (from a full-tree solve) with the current ranges.
//  * OracleLeafProvider    - solves every river subgame from the current
//                            ranges with CFR (expensive; correctness oracle).
#pragma once

#include <map>
#include <memory>

#include "solvers/subgame_resolver/vector_cfr.h"
#include "value_net/mlp.h"

namespace aegis::poker {

struct LeafStats {
  long long calls = 0;
  long long rows = 0;
  double seconds = 0.0;
};

class NeuralLeafProvider final : public LeafValueProvider {
 public:
  explicit NeuralLeafProvider(nn::MLP net, bool zero_sum = true) : net_(std::move(net)), zero_sum_(zero_sum) {}
  void evaluate(const SubgameContext& ctx, std::span<Query> queries) override;
  const LeafStats& stats() const { return stats_; }

 private:
  nn::MLP net_;
  bool zero_sum_;
  LeafStats stats_;
};

class BlueprintLeafProvider final : public LeafValueProvider {
 public:
  // `full` is the context of the non-depth-limited tree (same turn sizes and
  // hand lists) and `blueprint` a policy on it.
  BlueprintLeafProvider(const SubgameContext& full, VectorPolicy blueprint)
      : full_(full), blueprint_(std::move(blueprint)) {}
  void evaluate(const SubgameContext& ctx, std::span<Query> queries) override;
  const LeafStats& stats() const { return stats_; }

 private:
  int map_leaf(const SubgameContext& ctx, int leaf);
  const SubgameContext& full_;
  VectorPolicy blueprint_;
  std::map<int, int> leaf_to_full_;
  LeafStats stats_;
};

class OracleLeafProvider final : public LeafValueProvider {
 public:
  OracleLeafProvider(StreetSizes river_sizes, int iterations, CFRConfig cfr = CFRConfig::dcfr())
      : sizes_(std::move(river_sizes)), iterations_(iterations), cfr_(cfr) {}
  void evaluate(const SubgameContext& ctx, std::span<Query> queries) override;
  const LeafStats& stats() const { return stats_; }

 private:
  StreetSizes sizes_;
  int iterations_;
  CFRConfig cfr_;
  LeafStats stats_;
};

// Exact counterfactual values of a river subgame solved from given ranges.
// Returns per-combo (1326) cfv for both players; used by the oracle and by
// the training-data generator.
struct RiverSolveOutput {
  std::array<std::vector<double>, 2> cfv;  // 1326 each (0 for dead combos)
  double exploitability_pct_pot = 0.0;
};
// Hands in the solve: combos with positive range weight, plus (when given)
// every combo with positive `support` weight; nullptr support = all live hands.
RiverSolveOutput solve_river_values(const std::vector<Card>& board5, const double* r0, const double* r1, double pot,
                                    double stack, const StreetSizes& sizes, int iterations, CFRConfig cfr,
                                    const double* support0 = nullptr, const double* support1 = nullptr);

}  // namespace aegis::poker
