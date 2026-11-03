// Weighted hand ranges over the 1,326 two-card combinations.
//
// A range r_i(h) is a non-negative weight per combination, conditioned on the
// public state. Combinations that share a card with the board (or any other
// known card) are impossible and carry zero weight.
//
// Bayesian update after observing action a under an assumed policy σ_i:
//     r_i(h | a) = r_i(h) σ_i(a | h) / Σ_{h'} r_i(h') σ_i(a | h')
// This is a belief update under the assumed policy; it never reveals cards.
//
// Range notation accepted by Range::parse (comma separated, optional ":w"):
//   AA  AKs  AKo  AK  TT+  ATs+  KTo+  A9+  22-55  A2s-A5s  KTo-KQo  AsKh  random
#pragma once

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "poker/cards/cards.h"
#include "util/rng.h"

namespace aegis::poker {

class Range {
 public:
  Range() { w.fill(0.0); }
  static Range full();
  static Range parse(std::string_view text);
  static Range from_vector(std::span<const double> weights);

  double& operator[](int combo) { return w[combo]; }
  double operator[](int combo) const { return w[combo]; }

  // Zero every combination that intersects `dead` (board / known cards).
  Range& remove(CardMask dead);
  double total() const;
  int num_nonzero() const;
  Range& normalize();  // scales to total 1 (no-op on an empty range)
  Range normalized() const;

  // Posterior after an action with per-combination likelihood σ(a | h).
  // Returned range is normalised; throws if the action has zero probability.
  Range bayes_update(std::span<const double> likelihood) const;

  // Distribution of the opponent's hand given our known cards: zero the
  // combinations we block and renormalise.
  Range conditioned_on(CardMask known) const;
  // Fraction of the range's mass that `known` removes (blocker effect).
  double blocked_fraction(CardMask known) const;

  // Total weight per 13x13 hand class and the average weight (0..1) of the
  // combos that are not dead.
  std::array<double, kNumHandClasses> class_mass() const;
  std::array<double, kNumHandClasses> class_frequency(CardMask dead = 0) const;

  std::string to_string() const;

  std::array<double, kNumCombos> w;
};

// Random range generator used to create diverse training positions for the
// value network: assigns probability mass by recursive random splitting over
// combinations sorted by `strength_order` (best -> worst), so generated ranges
// are correlated with hand strength like real betting ranges (DeepStack-style).
Range random_range_by_strength(std::span<const int> strength_order, Rng& rng);

}  // namespace aegis::poker
