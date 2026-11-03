#include "poker/hand_evaluator/evaluator.h"

#include <algorithm>
#include <sstream>

#include "util/check.h"

namespace aegis::poker {

namespace {

inline int top_bit(uint32_t m) { return 31 - __builtin_clz(m); }

// Packs up to `count` highest set ranks of mask m into kicker nibbles starting
// at position `slot` (0 = k1 ... 4 = k5).
inline HandValue take_top(uint32_t m, int count, int slot) {
  HandValue v = 0;
  for (int i = 0; i < count && m; ++i) {
    const int r = top_bit(m);
    m &= ~(1u << r);
    v |= static_cast<HandValue>(r) << (4 * (4 - slot - i));
  }
  return v;
}

// Highest straight in a 13-bit rank mask, returned as the rank of its top
// card (3 = five-high wheel), or -1.
inline int straight_high(uint32_t m) {
  // Shift ranks up by one and place the Ace at bit 0 as well (A-2-3-4-5).
  const uint32_t x = (m << 1) | ((m >> 12) & 1u);
  const uint32_t s = x & (x >> 1) & (x >> 2) & (x >> 3) & (x >> 4);
  if (!s) return -1;
  return top_bit(s) + 3;  // bit j means ranks j-1 .. j+3 present (shifted coords)
}

inline HandValue make(int category, HandValue kickers) {
  return (static_cast<HandValue>(category) << 20) | kickers;
}

HandValue eval_suits(const uint32_t s[4]) {
  const uint32_t any = s[0] | s[1] | s[2] | s[3];
  // Bit-sliced multiplicities.
  const uint32_t ge2 = (s[0] & s[1]) | (s[0] & s[2]) | (s[0] & s[3]) | (s[1] & s[2]) | (s[1] & s[3]) | (s[2] & s[3]);
  const uint32_t ge3 = (s[0] & s[1] & s[2]) | (s[0] & s[1] & s[3]) | (s[0] & s[2] & s[3]) | (s[1] & s[2] & s[3]);
  const uint32_t four = s[0] & s[1] & s[2] & s[3];

  // Flush / straight flush (at most one suit can hold five of seven cards).
  for (int su = 0; su < 4; ++su) {
    if (__builtin_popcount(s[su]) >= 5) {
      const int sh = straight_high(s[su]);
      if (sh >= 0) return make(kStraightFlush, static_cast<HandValue>(sh) << 16);
      // With at most 7 cards, a 5-card flush leaves at most 2 off-suit cards,
      // which is too few for quads (needs 3 off-suit) or a full house (needs
      // 3 off-suit), so the flush is the final answer.
      return make(kFlush, take_top(s[su], 5, 0));
    }
  }
  if (four) {
    const int q = top_bit(four);
    return make(kQuads, (static_cast<HandValue>(q) << 16) | take_top(any & ~(1u << q), 1, 1));
  }
  const uint32_t three = ge3 & ~four;
  const uint32_t pairs = ge2 & ~ge3;
  if (three) {
    const int t = top_bit(three);
    const uint32_t rest = (three & ~(1u << t)) | pairs;
    if (rest) {
      const int p = top_bit(rest);
      return make(kFullHouse, (static_cast<HandValue>(t) << 16) | (static_cast<HandValue>(p) << 12));
    }
  }
  const int sh = straight_high(any);
  if (sh >= 0) return make(kStraight, static_cast<HandValue>(sh) << 16);
  if (three) {
    const int t = top_bit(three);
    return make(kTrips, (static_cast<HandValue>(t) << 16) | take_top(any & ~(1u << t), 2, 1));
  }
  if (pairs) {
    const int p1 = top_bit(pairs);
    const uint32_t others = pairs & ~(1u << p1);
    if (others) {
      const int p2 = top_bit(others);
      const uint32_t kick = any & ~(1u << p1) & ~(1u << p2);
      return make(kTwoPair, (static_cast<HandValue>(p1) << 16) | (static_cast<HandValue>(p2) << 12) |
                                take_top(kick, 1, 2));
    }
    return make(kPair, (static_cast<HandValue>(p1) << 16) | take_top(any & ~(1u << p1), 3, 1));
  }
  return make(kHighCard, take_top(any, 5, 0));
}

}  // namespace

HandValue evaluate(const Card* cards, int n) {
  AEGIS_CHECK(n >= 5 && n <= 7);
  uint32_t s[4] = {0, 0, 0, 0};
  for (int i = 0; i < n; ++i) s[suit_of(cards[i])] |= 1u << rank_of(cards[i]);
  return eval_suits(s);
}

HandValue evaluate(CardMask m) {
  uint32_t s[4] = {0, 0, 0, 0};
  int n = 0;
  while (m) {
    const int c = __builtin_ctzll(m);
    m &= m - 1;
    s[suit_of(c)] |= 1u << rank_of(c);
    ++n;
  }
  AEGIS_CHECK(n >= 5 && n <= 7);
  return eval_suits(s);
}

HandValue evaluate_naive5(const Card cards[5]) {
  int ranks[5];
  bool flush = true;
  for (int i = 0; i < 5; ++i) {
    ranks[i] = rank_of(cards[i]);
    if (suit_of(cards[i]) != suit_of(cards[0])) flush = false;
  }
  std::sort(ranks, ranks + 5, std::greater<int>());
  int count[13] = {};
  for (int r : ranks) count[r]++;
  // Order ranks by (multiplicity desc, rank desc).
  std::vector<std::pair<int, int>> groups;  // (count, rank)
  for (int r = 12; r >= 0; --r)
    if (count[r]) groups.emplace_back(count[r], r);
  std::stable_sort(groups.begin(), groups.end(),
                   [](const auto& a, const auto& b) { return a.first > b.first; });
  bool straight = false;
  int high = -1;
  if (groups.size() == 5) {
    if (ranks[0] - ranks[4] == 4) {
      straight = true;
      high = ranks[0];
    } else if (ranks[0] == 12 && ranks[1] == 3 && ranks[2] == 2 && ranks[3] == 1 && ranks[4] == 0) {
      straight = true;
      high = 3;
    }
  }
  auto pack = [](int cat, std::initializer_list<int> ks) {
    HandValue v = static_cast<HandValue>(cat) << 20;
    int shift = 16;
    for (int k : ks) {
      v |= static_cast<HandValue>(k) << shift;
      shift -= 4;
    }
    return v;
  };
  if (straight && flush) return pack(kStraightFlush, {high});
  if (groups[0].first == 4) return pack(kQuads, {groups[0].second, groups[1].second});
  if (groups[0].first == 3 && groups[1].first == 2) return pack(kFullHouse, {groups[0].second, groups[1].second});
  if (flush) return pack(kFlush, {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
  if (straight) return pack(kStraight, {high});
  if (groups[0].first == 3) return pack(kTrips, {groups[0].second, groups[1].second, groups[2].second});
  if (groups[0].first == 2 && groups[1].first == 2)
    return pack(kTwoPair, {groups[0].second, groups[1].second, groups[2].second});
  if (groups[0].first == 2)
    return pack(kPair, {groups[0].second, groups[1].second, groups[2].second, groups[3].second});
  return pack(kHighCard, {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
}

const char* category_name(int c) {
  static const char* names[] = {"High Card", "Pair",       "Two Pair",  "Three of a Kind", "Straight",
                                "Flush",     "Full House", "Four of a Kind", "Straight Flush"};
  AEGIS_CHECK(c >= 0 && c <= 8);
  return names[c];
}

std::string describe(HandValue v) {
  static const char* rank_names[] = {"Twos", "Threes", "Fours", "Fives", "Sixes", "Sevens", "Eights",
                                     "Nines", "Tens", "Jacks", "Queens", "Kings", "Aces"};
  static const char* rank_single[] = {"Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
                                      "Nine", "Ten", "Jack", "Queen", "King", "Ace"};
  const int c = category_of(v);
  const int k1 = (v >> 16) & 15, k2 = (v >> 12) & 15;
  std::ostringstream os;
  os << category_name(c);
  switch (c) {
    case kPair:
    case kTrips:
    case kQuads:
      os << ", " << rank_names[k1];
      break;
    case kTwoPair:
      os << ", " << rank_names[k1] << " and " << rank_names[k2];
      break;
    case kFullHouse:
      os << ", " << rank_names[k1] << " full of " << rank_names[k2];
      break;
    case kStraight:
    case kStraightFlush:
    case kFlush:
    case kHighCard:
      os << ", " << rank_single[k1] << "-high";
      break;
    default:
      break;
  }
  return os.str();
}

}  // namespace aegis::poker
