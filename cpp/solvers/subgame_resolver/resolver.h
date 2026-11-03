// Safe and unsafe re-solving of poker subtrees (vector form).
//
//  resolve_subtree(ctx, base, root, opts) re-solves the subtree rooted at
//  `root` of a full game, starting from the reach vectors that `base`
//  induces there, and returns `base` with the re-solved strategy spliced in:
//
//   * kNaive    - both players enter with their base reach (ranges only).
//   * kResolve  - the opponent enters through the re-solve gadget with
//                 alternative values w(h) (default: its counterfactual
//                 best-response value against our base strategy).
//   * kMaxMargin- the opponent picks its root hand distribution to maximise
//                 v(h) - w(h).
//
//  Nested re-solving after an OFF-TREE opponent action x at decision node d:
//  re-solve the subtree below x with w(h) = max over the in-abstraction
//  actions a at d of the opponent's CBV(h, a) (best_alternative_constraint),
//  so the new action can never be worth more to the opponent than the best
//  action the blueprint already defended against.
#pragma once

#include <vector>

#include "solvers/subgame_resolver/tabular_resolving.h"
#include "solvers/subgame_resolver/vector_cfr.h"

namespace aegis::poker {

struct VectorResolveOptions {
  ResolveMethod method = ResolveMethod::kResolve;
  int us = 0;
  int iterations = 500;
  CFRConfig cfr = CFRConfig::dcfr();
  bool use_cbv = true;            // constraint = CBV (true) or base cfv (false)
  bool replace_opponent = false;  // also splice the opponent's re-solved strategy
  std::vector<double> constraint; // explicit constraint (overrides use_cbv)
  LeafValueProvider* leaf = nullptr;
};

struct VectorResolveResult {
  VectorPolicy policy;               // base with the subtree replaced
  std::vector<double> constraint;    // w(h) per opponent hand (empty for naive)
  std::vector<double> margin;        // w(h) - CBV_new(h); >= 0 means no hand gains
  double worst_violation = 0.0;      // max_h (CBV_new - w)+ weighted by nothing
  double total_violation = 0.0;      // Σ_h (CBV_new - w)+
  std::vector<double> root_reach[2];
  double seconds = 0.0;
};

VectorResolveResult resolve_subtree(const SubgameContext& ctx, const VectorPolicy& base, int root,
                                    const VectorResolveOptions& opts);

// Reach vectors of both players at `node` under `policy`.
void reach_at(const SubgameContext& ctx, const VectorPolicy& policy, int node, std::vector<double> out[2]);

// Opponent counterfactual values at `root`: best response to our part of
// `policy` (cbv = true) or following `policy` (cbv = false).
std::vector<double> opponent_constraint(const SubgameContext& ctx, const VectorPolicy& policy, int root, int opp,
                                        bool cbv);

// max over the given children of decision node `d` of the opponent's CBV.
std::vector<double> best_alternative_constraint(const SubgameContext& ctx, const VectorPolicy& policy, int d,
                                                const std::vector<int>& allowed_children, int opp);

// Copies player `player`'s (-1: both) decision nodes of subtree(root) from src.
void splice_policy(const PublicTree& t, VectorPolicy& dst, const VectorPolicy& src, int root, int player);

// Decision nodes that open street `street` (0 = turn, 1 = river).
std::vector<int> street_roots(const PublicTree& t, int street);

// Is `a` a size available in the abstraction `sizes` (facing_bet selects the
// raise list)? Fold/check/call are always in the abstraction.
bool action_in_abstraction(const ActionLabel& a, const StreetSizes& sizes, bool facing_bet);

}  // namespace aegis::poker
