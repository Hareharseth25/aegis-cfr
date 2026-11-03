// Cards, card masks and the 1,326 two-card combinations of a 52-card deck.
//
// Card id = rank * 4 + suit, rank 0..12 = 2..A, suit 0..3 = c, d, h, s.
// Combination index for cards a < b: b(b-1)/2 + a, giving 0..1325.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aegis::poker {

using Card = int;
using CardMask = uint64_t;

inline constexpr int kNumCards = 52;
inline constexpr int kNumCombos = 1326;  // C(52, 2)
inline constexpr int kNumHandClasses = 169;

constexpr int rank_of(Card c) { return c >> 2; }
constexpr int suit_of(Card c) { return c & 3; }
constexpr Card make_card(int rank, int suit) { return rank * 4 + suit; }
constexpr CardMask card_bit(Card c) { return CardMask{1} << c; }

Card parse_card(std::string_view s);              // "As", "td", "2c"
std::string card_to_string(Card c);               // "As"
std::vector<Card> parse_cards(std::string_view s);  // "AsKh8c", "As Kh 8c", "As,Kh"
std::string cards_to_string(const std::vector<Card>& cs);
CardMask mask_of(const std::vector<Card>& cs);
int popcount(CardMask m);

char rank_char(int rank);
int rank_from_char(char c);  // -1 if invalid

constexpr int combo_index(Card a, Card b) {
  if (a > b) {
    const Card t = a;
    a = b;
    b = t;
  }
  return b * (b - 1) / 2 + a;
}

struct ComboTable {
  std::array<std::array<Card, 2>, kNumCombos> cards;  // cards[i] = {low, high}
  std::array<CardMask, kNumCombos> mask;
  std::array<int, kNumCombos> hand_class;  // 0..168
};
const ComboTable& combos();

inline CardMask combo_mask(int i) { return combos().mask[i]; }
std::string combo_to_string(int i);  // high card first, e.g. "AsKh"
int parse_combo(std::string_view s);  // "AsKh" -> index

// The 169 strategically distinct starting-hand classes laid out on the usual
// 13x13 grid: row/col 0 = Ace. Diagonal = pairs, above = suited, below = offsuit.
std::string hand_class_name(int cls);  // "AA", "AKs", "AKo"
int hand_class_index(std::string_view name);  // -1 if invalid
int hand_class_row(int cls);
int hand_class_col(int cls);
int hand_class_combos(int cls);  // 6, 4 or 12

}  // namespace aegis::poker
