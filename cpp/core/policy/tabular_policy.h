// A behaviour-strategy profile σ = (σ_0, σ_1) stored as one probability per
// global action slot of a GameTree. σ_i(I, *) is a distribution over A(I).
#pragma once

#include <span>
#include <vector>

#include "core/game_tree/game_tree.h"
#include "core/information_sets/infoset_storage.h"
#include "util/rng.h"

namespace aegis {

class TabularPolicy {
 public:
  TabularPolicy() = default;
  explicit TabularPolicy(const GameTree& tree);  // uniform random

  static TabularPolicy uniform(const GameTree& tree) { return TabularPolicy(tree); }
  // Each information set gets an independent Dirichlet(concentration) draw.
  static TabularPolicy random(const GameTree& tree, Rng& rng, double concentration = 1.0);
  // Normalises per-infoset non-negative weights (e.g. strategy sums); an
  // infoset with zero total weight becomes uniform.
  static TabularPolicy from_weights(const GameTree& tree, std::span<const double> weights);

  std::span<double> at(const GameTree& tree, int infoset) { return infoset_span(tree, probs, infoset); }
  std::span<const double> at(const GameTree& tree, int infoset) const { return infoset_span(tree, probs, infoset); }

  // Throws if any infoset is not a valid distribution.
  void validate(const GameTree& tree, double tol = 1e-9) const;

  // Copies player p's infosets from `other` (used to assemble profiles).
  void set_player_from(const GameTree& tree, const TabularPolicy& other, int p);

  std::vector<double> probs;
};

}  // namespace aegis
