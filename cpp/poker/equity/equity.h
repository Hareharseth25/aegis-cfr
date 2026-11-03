// Exact and Monte-Carlo equity with card removal.
//
// Exact enumeration walks every completion of the board. Range-vs-range
// equity on a fixed 5-card board uses the O(n) sorted sweep with per-card
// running sums (inclusion-exclusion over the two blocking cards), the same
// kernel the river solver uses for showdown values.
#pragma once

#include <array>
#include <vector>

#include "poker/cards/cards.h"
#include "poker/hand_evaluator/evaluator.h"
#include "poker/ranges/range.h"
#include "util/rng.h"

namespace aegis::poker {

struct EquityResult {
  double win = 0.0, tie = 0.0, lose = 0.0;  // probabilities for player 1 (hero)
  double equity() const { return win + 0.5 * tie; }
  double std_error = 0.0;                   // Monte Carlo only
  long long samples = 0;                    // boards (exact) or samples (MC)
};

// Hand vs hand, exact enumeration over all board completions.
EquityResult equity_exact(int hero_combo, int villain_combo, const std::vector<Card>& board);

// Hand vs weighted range, exact enumeration (weights are normalised over
// compatible combos). Feasible for boards with >= 3 cards (and preflop with
// patience: 1.7M boards x range size evaluations).
EquityResult equity_vs_range_exact(int hero_combo, const Range& villain, const std::vector<Card>& board);

// Range vs range, exact, for boards with 3-5 cards. Also returns each hero
// combo's equity against the villain range (NaN where hero weight is zero or
// no compatible villain combo exists).
struct RangeEquity {
  EquityResult overall;
  std::array<double, kNumCombos> combo_equity;
  std::array<double, kNumCombos> combo_weight;  // Σ compatible villain mass x board weight
};
RangeEquity equity_range_vs_range_exact(const Range& hero, const Range& villain, const std::vector<Card>& board);

// Range vs range Monte Carlo (any board size, including preflop).
EquityResult equity_range_vs_range_mc(const Range& hero, const Range& villain, const std::vector<Card>& board,
                                      long long samples, uint64_t seed);

// Hand strength of every combo on a complete 5-card board (0 for combos that
// intersect the board). Cached per board by callers.
std::array<HandValue, kNumCombos> board_hand_values(const std::vector<Card>& board5);

}  // namespace aegis::poker
