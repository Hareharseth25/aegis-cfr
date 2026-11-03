#include "core/policy/tabular_policy.h"

#include <cmath>

#include "util/check.h"

namespace aegis {

TabularPolicy::TabularPolicy(const GameTree& tree) : probs(tree.total_actions, 0.0) {
  for (const auto& I : tree.infosets)
    for (int a = 0; a < I.num_actions; ++a) probs[I.offset + a] = 1.0 / I.num_actions;
}

TabularPolicy TabularPolicy::random(const GameTree& tree, Rng& rng, double concentration) {
  TabularPolicy p(tree);
  for (const auto& I : tree.infosets) {
    double total = 0.0;
    for (int a = 0; a < I.num_actions; ++a) {
      const double g = rng.gamma(concentration);
      p.probs[I.offset + a] = g;
      total += g;
    }
    for (int a = 0; a < I.num_actions; ++a) p.probs[I.offset + a] /= total;
  }
  return p;
}

TabularPolicy TabularPolicy::from_weights(const GameTree& tree, std::span<const double> w) {
  AEGIS_CHECK(static_cast<int>(w.size()) == tree.total_actions);
  TabularPolicy p(tree);
  for (const auto& I : tree.infosets) {
    double total = 0.0;
    for (int a = 0; a < I.num_actions; ++a) {
      AEGIS_CHECK_MSG(w[I.offset + a] >= 0.0 || std::abs(w[I.offset + a]) < 1e-12, "negative weight");
      total += std::max(0.0, w[I.offset + a]);
    }
    if (total > 0.0)
      for (int a = 0; a < I.num_actions; ++a) p.probs[I.offset + a] = std::max(0.0, w[I.offset + a]) / total;
  }
  return p;
}

void TabularPolicy::validate(const GameTree& tree, double tol) const {
  AEGIS_CHECK(static_cast<int>(probs.size()) == tree.total_actions);
  for (const auto& I : tree.infosets) {
    double total = 0.0;
    for (int a = 0; a < I.num_actions; ++a) {
      const double x = probs[I.offset + a];
      AEGIS_CHECK_MSG(x >= -tol && std::isfinite(x), "invalid probability at " << I.key);
      total += x;
    }
    AEGIS_CHECK_MSG(std::abs(total - 1.0) < tol * I.num_actions + 1e-12,
                    "policy at '" << I.key << "' sums to " << total);
  }
}

void TabularPolicy::set_player_from(const GameTree& tree, const TabularPolicy& other, int p) {
  for (const auto& I : tree.infosets)
    if (I.player == p)
      for (int a = 0; a < I.num_actions; ++a) probs[I.offset + a] = other.probs[I.offset + a];
}

}  // namespace aegis
