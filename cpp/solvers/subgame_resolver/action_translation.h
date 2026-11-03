// Action translation: playing an abstract strategy in a finer reference game.
//
// When the opponent uses a bet size x (as a pot fraction) that the abstraction
// does not contain, it is mapped to the neighbouring abstract sizes A < x < B.
// The pseudo-harmonic mapping (Ganzfried & Sandholm, IJCAI 2013) maps to A
// with probability
//     f(x) = ((B - x)(1 + A)) / ((B - A)(1 + x)),
// which is the unique mapping that makes a simple "call or fold" opponent
// indifferent in the [0, 1]-normalised toy game; "nearest" maps to the closer
// size deterministically.
//
// translate_policy() builds a full behaviour strategy for a REFERENCE public
// tree (e.g. many bet sizes) from a policy on an ABSTRACT tree (few sizes).
// Each reference node carries a mixture of abstract nodes; randomised
// translation is resolved exactly with Bayes' rule on each player's own
// reach inside each mixture component:
//     σ_ref(n, h, a) = Σ_k w_k π_p^k(h) σ_k(h, a) / Σ_k w_k π_p^k(h),
// which is the behaviour strategy realisation-equivalent to "sample the
// translation once, then follow the abstract strategy".
#pragma once

#include "solvers/subgame_resolver/subgame_context.h"

namespace aegis::poker {

enum class TranslationMethod { kPseudoHarmonic, kNearest };

// Probability of mapping x to the smaller size A (A <= x <= B, pot fractions).
double pseudo_harmonic_low_prob(double A, double B, double x);

// Builds the reference-tree policy. Both trees must come from configs with the
// same board, pot, stack and players' hand lists (same ranges).
VectorPolicy translate_policy(const SubgameContext& ref, const SubgameContext& abs, const VectorPolicy& abs_policy,
                              TranslationMethod method = TranslationMethod::kPseudoHarmonic);

// The (abstract node, probability) distribution that a reference action at
// reference node `ref_node` maps to from abstract node `abs_node` when the
// action is not one the abstract player chose (off-tree): exposed for tests
// and for the refinement heuristics.
std::vector<std::pair<int, double>> map_offtree_action(const PublicTree& ref, int ref_child, const PublicTree& abs,
                                                       int abs_node, TranslationMethod method);

}  // namespace aegis::poker
