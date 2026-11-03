// Vector-form CFR for heads-up No-Limit Hold'em turn/river subgames.
//
// Every (public node, private hand) pair is an information set, so regrets
// and average strategies are stored per decision node as contiguous
// action-major arrays R[a * n_p + i]. One iteration performs, for each player
// p (alternating updates by default):
//   1. forward pass: reach vectors of both players under σ^t;
//   2. leaf queries (depth-limited trees): batched counterfactual values;
//   3. backward pass: counterfactual values of p, with
//        r(n, h, a) = cfv_a(h) - Σ_b σ(n,h,b) cfv_b(h)
//      applied through the shared CFRConfig rule (CFR / CFR+ / DCFR / LCFR)
//      and S(n,h,a) += w_t * π_p(n,h) * σ(n,h,a).
// Turn trees parallelise over the river cards of each chance node.
//
// Re-solving gadget (Burch, Johanson & Bowling 2014; Moravčík et al. 2017):
// when enabled, the opponent o enters the subgame root through a gadget. For
// each opponent hand h the opponent may Terminate and receive the constraint
// value w(h) (a counterfactual value from the blueprint, in the same units as
// cfv computed against our root reach) or Follow into the subgame. Gadget
// regrets are updated each iteration with RM+, and the opponent's root reach
// is σ_Follow(h). MaxMargin instead lets the opponent choose a distribution
// over hands that maximises v(h) - w(h), which maximises the minimum margin.
#pragma once

#include <span>
#include <vector>

#include "solvers/common/regret_rules.h"
#include "solvers/subgame_resolver/subgame_context.h"

namespace aegis::poker {

class LeafValueProvider {
 public:
  virtual ~LeafValueProvider() = default;
  struct Query {
    int node = -1;
    const std::vector<double>* reach[2] = {nullptr, nullptr};
    std::vector<double>* cfv[2] = {nullptr, nullptr};
  };
  // Fill cfv for both players at every queried leaf (outputs pre-sized).
  virtual void evaluate(const SubgameContext& ctx, std::span<Query> queries) = 0;
};

struct GadgetConfig {
  enum class Mode { kResolve, kMaxMargin };
  bool enabled = false;
  int opponent = 1;
  Mode mode = Mode::kResolve;
  std::vector<double> constraint;  // per opponent local hand
};

struct VectorCFRConfig {
  VectorCFRConfig() = default;
  VectorCFRConfig(CFRConfig c) : cfr(c) {}  // NOLINT: implicit by design
  CFRConfig cfr = CFRConfig::dcfr();
  GadgetConfig gadget;
  LeafValueProvider* leaf = nullptr;
  // Solve only the subtree rooted at `root` (re-solving). The root reach
  // vectors default to the range priors when left empty.
  int root = 0;
  std::vector<double> root_reach[2];
};

class VectorCFR {
 public:
  VectorCFR(const SubgameContext& ctx, VectorCFRConfig cfg);

  void iterate(int n = 1);
  int iteration() const { return t_; }

  // Policies cover the solved subtree only (other decision nodes are empty).
  VectorPolicy average_policy() const;
  VectorPolicy current_policy() const;
  bool in_subtree(int node) const { return in_subtree_[node] != 0; }

  // Root counterfactual values of player p from the most recent pass.
  const std::vector<double>& root_cfv(int p) const {
    AEGIS_CHECK(p == 0 || p == 1);
    return root_cfv_[p];
  }
  // Gadget: per opponent hand probability of entering the subgame (resolve),
  // or the opponent's root distribution (maxmargin).
  std::vector<double> gadget_root_reach() const;

  // Work counter: Σ over visited nodes of the number of hands processed.
  long long hand_updates() const { return hand_updates_; }
  long long leaf_queries() const { return leaf_queries_; }

  const SubgameContext& context() const { return ctx_; }
  // Cumulative regrets / strategy sums of a decision node (action-major).
  const std::vector<double>& regrets(int node) const { return regret_[decision_of(node)]; }
  const std::vector<double>& strategy_sums(int node) const { return sum_[decision_of(node)]; }
  size_t memory_bytes() const;

 private:
  int decision_of(int node) const {
    const auto& nd = tree_.node(node);
    AEGIS_CHECK_MSG(nd.type == PNodeType::kDecision, "node " << node << " is not a decision node");
    return nd.decision_index;
  }
  void compute_sigma(int node);
  void forward(int node);
  void collect_leaves(int node, std::vector<int>& out);
  void backward(int node, int p);
  void update_gadget();

  const SubgameContext& ctx_;
  const PublicTree& tree_;
  VectorCFRConfig cfg_;
  int t_ = 0;
  // Per decision node (by decision_index): action-major arrays.
  std::vector<std::vector<double>> regret_, sum_, sigma_;
  // Per node, per player reach; per node cfv of the player being updated.
  std::vector<std::vector<double>> reach_[2];
  std::vector<std::vector<double>> cfv_;
  std::vector<std::vector<double>> leaf_cfv_[2];
  std::vector<int> leaves_;
  std::vector<uint8_t> in_subtree_;
  long long subtree_nodes_ = 0;
  std::vector<double> root_cfv_[2];
  // Gadget state.
  std::vector<double> g_regret_follow_, g_regret_term_, g_root_reach_;
  double dpos_ = 1.0, dneg_ = 1.0, avg_w_ = 1.0;
  long long hand_updates_ = 0, leaf_queries_ = 0;
};

}  // namespace aegis::poker
