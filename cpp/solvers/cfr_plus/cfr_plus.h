// CFR+ (Tammelin 2014; Tammelin, Burch, Johanson & Bowling 2015).
//   R_t^+(I,a) = max(0, R_{t-1}^+(I,a) + r_t(I,a)),
// alternating updates and linear averaging (weight max(t - d, 0)).
// Implemented by the shared TabularCFR engine via CFRConfig::cfr_plus().
#pragma once

#include "solvers/vanilla_cfr/cfr.h"

namespace aegis {

inline TabularCFR make_cfr_plus(const GameTree& tree, int averaging_delay = 0) {
  return TabularCFR(tree, CFRConfig::cfr_plus(averaging_delay));
}

}  // namespace aegis
