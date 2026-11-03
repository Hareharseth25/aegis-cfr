// Discounted CFR (Brown & Sandholm, "Solving Imperfect-Information Games via
// Discounted Regret Minimization", AAAI 2019).
//
// On iteration t: positive accumulated regrets are multiplied by t^α/(t^α+1),
// negative ones by t^β/(t^β+1), and average-strategy contributions are
// weighted by t^γ. Recommended DCFR(α=1.5, β=0, γ=2). Linear CFR is
// DCFR(1, 1, 1). β = -∞ discards negative regret (CFR+-like floors).
#pragma once

#include "solvers/vanilla_cfr/cfr.h"

namespace aegis {

inline TabularCFR make_dcfr(const GameTree& tree, double alpha = 1.5, double beta = 0.0, double gamma = 2.0) {
  return TabularCFR(tree, CFRConfig::dcfr(alpha, beta, gamma));
}
inline TabularCFR make_linear_cfr(const GameTree& tree) { return TabularCFR(tree, CFRConfig::linear()); }

}  // namespace aegis
