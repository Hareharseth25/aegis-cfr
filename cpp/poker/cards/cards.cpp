#include "poker/cards/cards.h"

#include <cctype>

#include "util/check.h"

namespace aegis::poker {

namespace {
constexpr char kRanks[] = "23456789TJQKA";
constexpr char kSuits[] = "cdhs";
}  // namespace

char rank_char(int r) { return kRanks[r]; }

int rank_from_char(char c) {
  c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  for (int r = 0; r < 13; ++r)
    if (kRanks[r] == c) return r;
  return -1;
}

static int suit_from_char(char c) {
  c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  for (int s = 0; s < 4; ++s)
    if (kSuits[s] == c) return s;
  return -1;
}

Card parse_card(std::string_view s) {
  AEGIS_CHECK_MSG(s.size() == 2, "bad card '" << s << "'");
  const int r = rank_from_char(s[0]);
  const int su = suit_from_char(s[1]);
  AEGIS_CHECK_MSG(r >= 0 && su >= 0, "bad card '" << s << "'");
  return make_card(r, su);
}

std::string card_to_string(Card c) {
  AEGIS_CHECK(c >= 0 && c < kNumCards);
  return {kRanks[rank_of(c)], kSuits[suit_of(c)]};
}

std::vector<Card> parse_cards(std::string_view s) {
  std::vector<Card> out;
  std::string compact;
  for (char ch : s)
    if (!std::isspace(static_cast<unsigned char>(ch)) && ch != ',' && ch != '[' && ch != ']') compact += ch;
  AEGIS_CHECK_MSG(compact.size() % 2 == 0, "bad card list '" << s << "'");
  CardMask seen = 0;
  for (size_t i = 0; i < compact.size(); i += 2) {
    const Card c = parse_card(std::string_view(compact).substr(i, 2));
    AEGIS_CHECK_MSG(!(seen & card_bit(c)), "duplicate card " << card_to_string(c));
    seen |= card_bit(c);
    out.push_back(c);
  }
  return out;
}

std::string cards_to_string(const std::vector<Card>& cs) {
  std::string s;
  for (Card c : cs) s += card_to_string(c);
  return s;
}

CardMask mask_of(const std::vector<Card>& cs) {
  CardMask m = 0;
  for (Card c : cs) m |= card_bit(c);
  return m;
}

int popcount(CardMask m) { return __builtin_popcountll(m); }

// Hand-class grid: index = row * 13 + col with row/col 0 = Ace (rank 12).
int hand_class_row(int cls) { return cls / 13; }
int hand_class_col(int cls) { return cls % 13; }

static int class_of_cards(Card a, Card b) {
  int ra = rank_of(a), rb = rank_of(b);
  if (ra < rb) std::swap(ra, rb);  // ra = higher rank
  const int hi = 12 - ra, lo = 12 - rb;  // grid coordinates (0 = Ace)
  if (ra == rb) return hi * 13 + hi;
  if (suit_of(a) == suit_of(b)) return hi * 13 + lo;  // suited: above diagonal
  return lo * 13 + hi;                                // offsuit: below diagonal
}

std::string hand_class_name(int cls) {
  AEGIS_CHECK(cls >= 0 && cls < kNumHandClasses);
  const int row = cls / 13, col = cls % 13;
  const int r_row = 12 - row, r_col = 12 - col;
  if (row == col) return {kRanks[r_row], kRanks[r_row]};
  if (row < col) return {kRanks[r_row], kRanks[r_col], 's'};
  return {kRanks[r_col], kRanks[r_row], 'o'};
}

int hand_class_index(std::string_view name) {
  if (name.size() < 2 || name.size() > 3) return -1;
  int r1 = rank_from_char(name[0]), r2 = rank_from_char(name[1]);
  if (r1 < 0 || r2 < 0) return -1;
  if (r1 < r2) std::swap(r1, r2);
  const int hi = 12 - r1, lo = 12 - r2;
  if (r1 == r2) return name.size() == 2 ? hi * 13 + hi : -1;
  if (name.size() != 3) return -1;
  const char t = static_cast<char>(std::tolower(static_cast<unsigned char>(name[2])));
  if (t == 's') return hi * 13 + lo;
  if (t == 'o') return lo * 13 + hi;
  return -1;
}

int hand_class_combos(int cls) {
  const int row = cls / 13, col = cls % 13;
  if (row == col) return 6;
  return row < col ? 4 : 12;
}

const ComboTable& combos() {
  static const ComboTable table = [] {
    ComboTable t{};
    for (Card b = 1; b < kNumCards; ++b)
      for (Card a = 0; a < b; ++a) {
        const int i = combo_index(a, b);
        t.cards[i] = {a, b};
        t.mask[i] = card_bit(a) | card_bit(b);
        t.hand_class[i] = class_of_cards(a, b);
      }
    return t;
  }();
  return table;
}

std::string combo_to_string(int i) {
  const auto& c = combos().cards[i];
  return card_to_string(c[1]) + card_to_string(c[0]);
}

int parse_combo(std::string_view s) {
  const auto cs = parse_cards(s);
  AEGIS_CHECK_MSG(cs.size() == 2, "combo needs two cards: '" << s << "'");
  return combo_index(cs[0], cs[1]);
}

}  // namespace aegis::poker
