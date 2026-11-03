// Subgame re-solving on a materialised game tree (Kuhn, Leduc, Coin Toss, ...).
//
// A subgame is a set of root histories forming a public state, closed in the
// sense that every information set below a root contains only histories
// below the roots. Given a blueprint σ^B, player `us` re-solves its strategy
// inside the subgame:
//
//  * kNaive (unsafe): the subgame roots are entered with the blueprint reach
//    of BOTH players (as a fixed chance distribution) and the subgame is
//    solved in isolation. Locally optimal, but the opponent can exploit it by
//    changing how often it enters the subgame with each hand.
//  * kResolve (Burch et al. 2014): for each opponent augmented information
//    set J at the roots the opponent chooses Terminate - receiving the
//    alternative value w(J) - or Follow into the subgame. Default w(J) is the
//    opponent's counterfactual best-response value (CBV) against σ^B, so the
//    re-solved strategy can never give the opponent more than it already
//    could obtain against the blueprint.
//  * kMaxMargin (Moravčík et al. 2016): the opponent chooses a distribution
//    over its root infosets to maximise v(J) - w(J); the solution maximises
//    the minimum margin w(J) - v(J).
//
// The result is σ^B with `us`'s subgame information sets replaced.
#pragma once

#include <vector>

#include "core/game_tree/game_tree.h"
#include "core/policy/tabular_policy.h"
#include "solvers/common/regret_rules.h"

namespace aegis {

enum class ResolveMethod { kNaive, kResolve, kMaxMargin };

struct TabularResolveOptions {
  ResolveMethod method = ResolveMethod::kResolve;
  int us = 1;
  int iterations = 1000;
  CFRConfig cfr = CFRConfig::cfr_plus();
  // Alternative values: counterfactual best response (true) or the
  // opponent's counterfactual value under the blueprint itself (false).
  bool use_cbv = true;
  // Also overwrite the opponent's subgame infosets with the re-solved ones
  // (naive solving produces both players' strategies).
  bool replace_opponent = false;
};

struct TabularResolveResult {
  TabularPolicy policy;                 // blueprint with the subgame replaced
  std::vector<int> subgame_infosets;    // infosets inside the subgame
  std::vector<double> constraint;       // w(J) per opponent root infoset
  std::vector<double> final_margin;     // w(J) - v_BR(J) after re-solving (>= 0 means safe)
  std::vector<double> gadget_follow;    // follow probabilities (resolve) / distribution (maxmargin)
};

// Roots: all nodes whose public state id equals `public_state`.
std::vector<int> nodes_with_public_state(const GameTree& tree, int public_state);
int public_state_id(const GameTree& tree, const std::string& key);

TabularResolveResult resolve_subgame(const GameTree& tree, const TabularPolicy& blueprint,
                                     const std::vector<int>& roots, const TabularResolveOptions& opts);

}  // namespace aegis
