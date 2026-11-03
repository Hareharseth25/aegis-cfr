// Monte-Carlo CFR on a materialised GameTree (Lanctot et al. 2009).
//
// Three sampling schemes:
//
//  * External sampling (ES): for traverser i, chance and opponent actions are
//    sampled from their true distributions; all of i's actions are explored.
//    Because the sampling probability q(z) equals π_{-i}(z), the sampled
//    counterfactual regret is simply ṽ(h*a) - ṽ(h).
//
//  * Outcome sampling (OS): a single trajectory; i's actions are sampled from
//    ε-uniform + (1-ε)σ. The estimator is
//        r̃(I,a) = W*π_i(h*a, z) - W*π_i(h, z)   (a sampled),  -W*π_i(h, z) otherwise,
//        W = u_i(z)*π_{-i}(z) / q(z).
//    Averages use stochastically-weighted averaging at opponent nodes.
//
//  * Importance-sampled external sampling (IS): like ES, but chance and
//    opponent actions are drawn from an arbitrary sampling distribution q
//    supplied by a SamplingPolicy (e.g. the adaptive allocator). Correctness
//    comes from carrying the likelihood ratio
//        W(h) = π_{-i}(h) / q_{-i}(h)
//    down the trajectory and multiplying each sampled value by p(a)/q(a):
//        ṽ(h) = (p(a)/q(a)) ṽ(h*a)  at non-traverser nodes,
//        r̃(I,a) = W(h) * (ṽ(h*a) - ṽ(h))  at traverser nodes.
//    E[r̃] is the exact counterfactual regret for any q with q(a) > 0 wherever
//    p(a) > 0, so MCCFR's convergence guarantee is preserved. With q = p this
//    reduces exactly to external sampling (tested bit-for-bit).
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/game_tree/game_tree.h"
#include "core/policy/tabular_policy.h"
#include "util/rng.h"

namespace aegis {

enum class MCCFRScheme { kExternal, kOutcome, kImportance };

struct MCCFRConfig {
  MCCFRScheme scheme = MCCFRScheme::kExternal;
  bool regret_matching_plus = false;  // clip cumulative regrets at zero
  bool linear_averaging = false;      // weight average-strategy updates by t
  double os_epsilon = 0.6;            // outcome-sampling exploration
  uint64_t seed = 1;
};

// Hook used by the IS scheme to choose sampling distributions and to receive
// per-traversal statistics. All methods are called from the traversal thread.
class SamplingPolicy {
 public:
  virtual ~SamplingPolicy() = default;
  // Fill q (same size as p) with the sampling distribution at non-traverser
  // node h. Must satisfy q(a) > 0 wherever p(a) > 0.
  virtual void sampling_distribution(int node, int traverser, std::span<const double> p,
                                     std::span<double> q) = 0;
  // Priority mass contributed by a traverser information set (summed over the
  // subtree and fed back through on_edge_backup).
  virtual double infoset_priority(int infoset) { return 0.0; }
  // Statistics at a traverser infoset visit: W = π_{-i}(h)/q_{-i}(h), sampled
  // action values ṽ(h*a), node value ṽ(h), and the work done below h.
  virtual void on_traverser_visit(int infoset, int node, double W, double true_reach,
                                  std::span<const double> action_values, double value, int64_t cost) {}
  // After sampling child k at non-traverser node h: subtree priority and cost.
  virtual void on_edge_backup(int node, int child_index, double priority, int64_t cost) {}
  virtual void begin_traversal(int traverser) {}
  virtual void end_traversal(int traverser) {}
};

class MCCFR {
 public:
  MCCFR(const GameTree& tree, MCCFRConfig cfg);

  // One iteration = one traversal for each player.
  void iterate(int n = 1);
  // Batch-synchronous parallel external sampling: within a batch every thread
  // runs `traversals_per_thread` traversals per player against a frozen σ,
  // accumulating deltas privately; deltas are then summed and applied. Each
  // batch is equivalent to one MCCFR iteration with a larger sample, so the
  // estimator stays unbiased; results are deterministic for a given seed.
  void iterate_parallel(int batches, int traversals_per_thread, int threads);

  void set_sampling_policy(SamplingPolicy* s) { sampler_ = s; }

  TabularPolicy average_policy() const;
  TabularPolicy current_policy() const;
  int iteration() const { return t_; }
  int64_t nodes_touched() const { return nodes_touched_; }
  const std::vector<double>& cumulative_regret() const { return regret_; }
  const std::vector<double>& strategy_sum() const { return strategy_sum_; }
  // Number of times each infoset received a regret update (coverage).
  const std::vector<int64_t>& infoset_updates() const { return updates_; }

  // Single traversal returning the sampled regret estimate r̃(I,*) for every
  // slot (without applying it). Used to measure estimator variance.
  std::vector<double> sample_regret_estimate(int traverser);

 private:
  struct Out {
    double v;
    double prio;
    int64_t cost;
  };
  struct Buffers {
    std::vector<double>* dregret;  // nullptr -> apply directly
    std::vector<double>* dsum;
    std::vector<int64_t>* dupdates;
    Rng* rng;
    int64_t touched = 0;
  };

  void current_sigma(int infoset, double* out) const;
  Out traverse_es(int h, int i, double W, Buffers& buf);
  // Outcome sampling: returns (u_i(z)/q(z) * π_{-i}(z)... ) see .cpp
  double traverse_os(int h, int i, double my_reach, double opp_reach, double sample_reach, double& tail,
                     Buffers& buf);
  void add_regret(Buffers& buf, int slot, double r);
  void add_avg(Buffers& buf, int slot, double w);

  const GameTree& tree_;
  MCCFRConfig cfg_;
  SamplingPolicy* sampler_ = nullptr;
  Rng rng_;
  int t_ = 0;
  std::vector<double> regret_;
  std::vector<double> strategy_sum_;
  std::vector<int64_t> updates_;
  std::vector<double>* snapshot_sigma_ = nullptr;  // frozen σ during parallel batches
  int64_t nodes_touched_ = 0;
};

}  // namespace aegis
