// Poker hand evaluation for 5, 6 or 7 cards.
//
// HandValue orders hands exactly: larger is better, equal means a tie.
//   value = category << 20 | k1 << 16 | k2 << 12 | k3 << 8 | k4 << 4 | k5
// with category 0 (high card) ... 8 (straight flush) and kickers k1..k5 as ranks.
//
// evaluate() is a branch-light bit-parallel evaluator: rank multiplicities are
// computed for all 13 ranks at once from four 13-bit suit masks. It is
// verified exhaustively against evaluate_naive5() on all 2,598,960 five-card
// hands (7,462 distinct classes) and against the known category frequencies
// of all 133,784,560 seven-card hands.
#pragma once

#include <cstdint>
#include <string>

#include "poker/cards/cards.h"

namespace aegis::poker {

using HandValue = uint32_t;

enum HandCategory : int {
  kHighCard = 0,
  kPair = 1,
  kTwoPair = 2,
  kTrips = 3,
  kStraight = 4,
  kFlush = 5,
  kFullHouse = 6,
  kQuads = 7,
  kStraightFlush = 8,
};

inline int category_of(HandValue v) { return static_cast<int>(v >> 20); }
const char* category_name(int category);
std::string describe(HandValue v);  // e.g. "Two Pair, Kings and Eights"

// Fast evaluator (5-7 cards).
HandValue evaluate(const Card* cards, int n);
HandValue evaluate(CardMask cards);

// Independent, deliberately simple reference implementation for exactly five
// cards (sorting + counting). Used only for verification.
HandValue evaluate_naive5(const Card cards[5]);

}  // namespace aegis::poker
