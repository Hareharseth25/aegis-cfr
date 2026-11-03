// Adaptive trajectory sampling for importance-sampled MCCFR (Experiment A).
//
// Priority of a traverser information set I:
//     S(I) = (ε + ρ(I))^α (ε + U(I))^β (ε + G(I))^γ / (ε + C(I))^δ
//  ρ(I): estimated opponent-and-chance reach. Horvitz-Thompson: each
//        traversal adds Σ_{visited h ∈ I} π_{-i}(h)/q_{-i}(h); ρ̂ = sum / T_i.
//  U(I): uncertainty of the sampled regret estimates (mean over actions of
//        the Welford standard error of r̃(I, a)).
//  G(I): decision sensitivity, Σ_a R+(I, a) / T_i (this infoset's share of
//        the average-regret bound).
//  C(I): cost, EMA of nodes touched below I.
//
// Every edge (h, k) at a non-traverser node keeps an EMA P(h, k) of the
// priority mass found below it. Sampling distribution:
//     q(k) = (1 - λ) p(k) + λ ψ(k),   ψ(k) prop. to (P(h,k) + ε_P)^τ * [p(k) > 0]
// so q(k) > 0 wherever p(k) > 0 and the MCCFR estimator stays unbiased
// (likelihood ratios p/q are applied by MCCFR). The priority score is a
// heuristic for WHERE to sample; it never changes what is estimated.
//
// Modes (all share this code, so comparisons are matched):
//   kExternal - λ = 0: plain external sampling (baseline);
//   kRandom   - ψ uniform over p-supported children ("random allocation");
//   kReach    - S = ρ only;
//   kAdaptive - full score.
#pragma once

#include <vector>

#include "adaptive/uncertainty/value_stats.h"
#include "core/game_tree/game_tree.h"
#include "solvers/mccfr/mccfr.h"

namespace aegis {

struct PriorityParams {
  double alpha = 1.0, beta = 0.5, gamma = 1.0, delta = 0.5;
  double eps = 1e-3;
  double lambda = 0.5;      // mixing weight of the priority distribution
  double tau = 1.0;         // sharpness of ψ
  double edge_rate = 0.05;  // EMA rate for edge priorities
  double prio_floor = 1e-6; // ε_P
};

enum class AllocationMode { kExternal, kRandom, kReach, kAdaptive };

class AdaptiveSampler final : public SamplingPolicy {
 public:
  AdaptiveSampler(const GameTree& tree, const MCCFR& solver, AllocationMode mode, PriorityParams params = {});

  void sampling_distribution(int node, int traverser, std::span<const double> p, std::span<double> q) override;
  double infoset_priority(int infoset) override;
  void on_traverser_visit(int infoset, int node, double W, double true_reach, std::span<const double> action_values,
                          double value, int64_t cost) override;
  void on_edge_backup(int node, int child_index, double priority, int64_t cost) override;
  void begin_traversal(int traverser) override;
  void end_traversal(int traverser) override;

  // Diagnostics.
  double score(int infoset) const;
  double reach_estimate(int infoset) const;
  double uncertainty(int infoset) const;
  double sensitivity(int infoset) const;
  double cost(int infoset) const;
  const std::vector<int64_t>& visits() const { return visits_; }
  AllocationMode mode() const { return mode_; }

 private:
  const GameTree& tree_;
  const MCCFR& solver_;
  AllocationMode mode_;
  PriorityParams pp_;
  int64_t traversals_[2] = {0, 0};
  std::vector<double> sumW_;
  std::vector<std::vector<RunningStats>> regret_stats_;
  std::vector<Ema> cost_;
  std::vector<int64_t> visits_;
  std::vector<Ema> edge_prio_;  // per edge (child slot)
};

}  // namespace aegis
