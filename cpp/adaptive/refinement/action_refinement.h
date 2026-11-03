// Dynamic action-abstraction refinement for river subgames (Experiment B).
//
// A river spot is solved in an abstraction with a few first-bet sizes per
// player. The abstract solution is mapped into a REFERENCE game with a fine
// grid of sizes (pseudo-harmonic translation) where exact exploitability is
// computed. For each size x the abstraction lacks, the "translation hole"
//     hole(x) = Σ_h r_p(h) * max(0, CBV_p(h, x) - max_{a ∈ abstraction} CBV_p(h, a)) / Z
// measures how much the bettor p gains by using x against the translated
// response instead of its best in-abstraction action: the regret of the
// missing action under the current solution. Refinement adds the sizes with
// the largest holes (or as chosen by a baseline / learned controller in
// Python) and re-solves.
#pragma once

#include <array>
#include <memory>
#include <vector>

#include "solvers/subgame_resolver/action_translation.h"
#include "solvers/subgame_resolver/vector_cfr.h"

namespace aegis::poker {

struct RefinementSpot {
  TreeConfig base;                     // river board, pot, stack, raise sizes, caps
  Range r0, r1;
  std::vector<double> reference_bets;  // fine first-bet grid (both players)
  int iterations = 500;
  CFRConfig cfr = CFRConfig::dcfr();
  int threads = 1;
  TranslationMethod translation = TranslationMethod::kPseudoHarmonic;
};

struct CandidateInfo {
  int player = 0;
  int node = 0;              // reference first-bet node
  double fraction = 0.0;     // candidate bet size (pot fraction)
  double hole = 0.0;         // chips
  double hole_pct_pot = 0.0;
  double reach_mass = 0.0;   // fraction of the player's prior mass reaching the node
  double gap_low = 0.0;      // log(x / nearest smaller abstract size), or log(x / min) if none
  double gap_high = 0.0;     // log(nearest larger abstract size / x)
};

struct AbstractionEval {
  std::array<std::vector<double>, 2> bets;
  double ref_exploitability_pct = 0.0;
  double ref_nash_conv = 0.0;
  double abs_exploitability_pct = 0.0;
  double ref_ev0 = 0.0;
  int abs_nodes = 0;
  long long abs_infosets = 0;  // Σ decision nodes x hands
  double solve_seconds = 0.0;
  double eval_seconds = 0.0;
  std::vector<CandidateInfo> candidates;
};

class ActionRefiner {
 public:
  explicit ActionRefiner(RefinementSpot spot);
  ActionRefiner(const ActionRefiner&) = delete;
  ActionRefiner& operator=(const ActionRefiner&) = delete;
  AbstractionEval evaluate(const std::array<std::vector<double>, 2>& bets, bool compute_candidates = true);
  const SubgameContext& reference() const { return *ref_ctx_; }
  const PublicTree& reference_tree() const { return ref_tree_; }

 private:
  TreeConfig with_bets(const std::array<std::vector<double>, 2>& bets) const;
  RefinementSpot spot_;
  PublicTree ref_tree_;
  std::unique_ptr<SubgameContext> ref_ctx_;
};

}  // namespace aegis::poker
