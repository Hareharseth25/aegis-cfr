#include "poker/equity/equity.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "util/check.h"

namespace aegis::poker {

namespace {

std::vector<Card> live_cards(CardMask dead) {
  std::vector<Card> out;
  for (Card c = 0; c < kNumCards; ++c)
    if (!(dead & card_bit(c))) out.push_back(c);
  return out;
}

// Calls fn(mask_of_added_cards) for every k-subset of `deck`.
template <typename Fn>
void for_each_subset(const std::vector<Card>& deck, int k, Fn&& fn) {
  const int n = static_cast<int>(deck.size());
  if (k == 0) {
    fn(CardMask{0});
    return;
  }
  std::vector<int> idx(k);
  std::iota(idx.begin(), idx.end(), 0);
  while (true) {
    CardMask m = 0;
    for (int i : idx) m |= card_bit(deck[i]);
    fn(m);
    int i = k - 1;
    while (i >= 0 && idx[i] == n - k + i) --i;
    if (i < 0) break;
    ++idx[i];
    for (int j = i + 1; j < k; ++j) idx[j] = idx[j - 1] + 1;
  }
}

void check_board(const std::vector<Card>& board) {
  AEGIS_CHECK_MSG(board.size() <= 5, "board has more than 5 cards");
  AEGIS_CHECK_MSG(popcount(mask_of(board)) == static_cast<int>(board.size()), "duplicate board card");
}

}  // namespace

std::array<HandValue, kNumCombos> board_hand_values(const std::vector<Card>& board5) {
  AEGIS_CHECK(board5.size() == 5);
  const CardMask bm = mask_of(board5);
  std::array<HandValue, kNumCombos> v{};
  const auto& C = combos();
  for (int i = 0; i < kNumCombos; ++i) v[i] = (C.mask[i] & bm) ? 0 : evaluate(bm | C.mask[i]);
  return v;
}

EquityResult equity_exact(int hero, int villain, const std::vector<Card>& board) {
  check_board(board);
  const auto& C = combos();
  const CardMask bm = mask_of(board);
  AEGIS_CHECK_MSG(!(C.mask[hero] & C.mask[villain]) && !(C.mask[hero] & bm) && !(C.mask[villain] & bm),
                  "hands overlap each other or the board");
  const auto deck = live_cards(bm | C.mask[hero] | C.mask[villain]);
  EquityResult r;
  long long w = 0, t = 0, l = 0;
  for_each_subset(deck, 5 - static_cast<int>(board.size()), [&](CardMask add) {
    const CardMask full = bm | add;
    const HandValue a = evaluate(full | C.mask[hero]);
    const HandValue b = evaluate(full | C.mask[villain]);
    if (a > b) ++w;
    else if (a == b) ++t;
    else ++l;
  });
  const double n = static_cast<double>(w + t + l);
  r.win = w / n;
  r.tie = t / n;
  r.lose = l / n;
  r.samples = w + t + l;
  return r;
}

EquityResult equity_vs_range_exact(int hero, const Range& villain, const std::vector<Card>& board) {
  check_board(board);
  const auto& C = combos();
  const CardMask bm = mask_of(board);
  AEGIS_CHECK_MSG(!(C.mask[hero] & bm), "hero hand overlaps the board");
  std::vector<int> vs;
  std::vector<double> vw;
  for (int i = 0; i < kNumCombos; ++i)
    if (villain[i] > 0.0 && !(C.mask[i] & (bm | C.mask[hero]))) {
      vs.push_back(i);
      vw.push_back(villain[i]);
    }
  AEGIS_CHECK_MSG(!vs.empty(), "villain range has no combos compatible with hero and board");
  // Each (villain combo, completion) pair is equally likely given the combo,
  // and every villain combo leaves the same number of completions, so we can
  // weight each completion by the villain combo weight directly.
  const auto deck = live_cards(bm | C.mask[hero]);
  double W = 0, T = 0, L = 0;
  long long boards = 0;
  const int k = 5 - static_cast<int>(board.size());
  for_each_subset(deck, k, [&](CardMask add) {
    ++boards;
    const CardMask full = bm | add;
    const HandValue a = evaluate(full | C.mask[hero]);
    for (size_t j = 0; j < vs.size(); ++j) {
      if (C.mask[vs[j]] & add) continue;
      const HandValue b = evaluate(full | C.mask[vs[j]]);
      if (a > b) W += vw[j];
      else if (a == b) T += vw[j];
      else L += vw[j];
    }
  });
  const double n = W + T + L;
  EquityResult r;
  r.win = W / n;
  r.tie = T / n;
  r.lose = L / n;
  r.samples = boards;
  return r;
}

RangeEquity equity_range_vs_range_exact(const Range& hero, const Range& villain, const std::vector<Card>& board) {
  check_board(board);
  AEGIS_CHECK_MSG(board.size() >= 3, "exact range-vs-range needs a flop, turn or river board (use MC preflop)");
  const auto& C = combos();
  const CardMask bm = mask_of(board);
  RangeEquity out;
  std::array<double, kNumCombos> win{}, tie{}, tot{};
  const auto deck = live_cards(bm);
  const int k = 5 - static_cast<int>(board.size());
  std::vector<int> order(kNumCombos);
  for_each_subset(deck, k, [&](CardMask add) {
    const CardMask full = bm | add;
    std::array<HandValue, kNumCombos> val{};
    int n = 0;
    for (int i = 0; i < kNumCombos; ++i) {
      if (C.mask[i] & full) continue;
      val[i] = evaluate(full | C.mask[i]);
      order[n++] = i;
    }
    std::sort(order.begin(), order.begin() + n, [&](int a, int b) { return val[a] < val[b]; });
    // Sweep ascending: villain mass strictly below (wins) and equal (ties),
    // minus combos that share a card with the hero combo.
    double below = 0.0;
    double below_card[kNumCards] = {};
    int g = 0;
    double total_v = 0.0, total_card[kNumCards] = {};
    for (int j = 0; j < n; ++j) {
      const int i = order[j];
      total_v += villain[i];
      total_card[C.cards[i][0]] += villain[i];
      total_card[C.cards[i][1]] += villain[i];
    }
    while (g < n) {
      int e = g;
      while (e < n && val[order[e]] == val[order[g]]) ++e;
      double eq = 0.0, eq_card[kNumCards] = {};
      for (int j = g; j < e; ++j) {
        const int i = order[j];
        eq += villain[i];
        eq_card[C.cards[i][0]] += villain[i];
        eq_card[C.cards[i][1]] += villain[i];
      }
      for (int j = g; j < e; ++j) {
        const int i = order[j];
        if (hero[i] <= 0.0) continue;
        const Card a = C.cards[i][0], b = C.cards[i][1];
        const double wv = below - below_card[a] - below_card[b];
        // Tie group contains hero combo itself (same cards) - remove it once
        // (it was subtracted twice via both cards and must be added back once).
        const double tv = eq - eq_card[a] - eq_card[b] + villain[i];
        const double all = total_v - total_card[a] - total_card[b] + villain[i];
        win[i] += wv;
        tie[i] += tv;
        tot[i] += all;
      }
      below += eq;
      for (int c = 0; c < kNumCards; ++c) below_card[c] += eq_card[c];
      g = e;
    }
  });
  double W = 0, T = 0, Ttot = 0;
  for (int i = 0; i < kNumCombos; ++i) {
    out.combo_weight[i] = tot[i];
    if (hero[i] > 0.0 && tot[i] > 0.0) {
      out.combo_equity[i] = (win[i] + 0.5 * tie[i]) / tot[i];
      W += hero[i] * win[i];
      T += hero[i] * tie[i];
      Ttot += hero[i] * tot[i];
    } else {
      out.combo_equity[i] = std::numeric_limits<double>::quiet_NaN();
    }
  }
  AEGIS_CHECK_MSG(Ttot > 0.0, "ranges have no compatible combinations");
  out.overall.win = W / Ttot;
  out.overall.tie = T / Ttot;
  out.overall.lose = 1.0 - out.overall.win - out.overall.tie;
  out.overall.samples = 1;
  return out;
}

EquityResult equity_range_vs_range_mc(const Range& hero, const Range& villain, const std::vector<Card>& board,
                                      long long samples, uint64_t seed) {
  check_board(board);
  const auto& C = combos();
  const CardMask bm = mask_of(board);
  std::vector<int> hs, vs;
  std::vector<double> hw, vw;
  for (int i = 0; i < kNumCombos; ++i) {
    if (C.mask[i] & bm) continue;
    if (hero[i] > 0) {
      hs.push_back(i);
      hw.push_back(hero[i]);
    }
    if (villain[i] > 0) {
      vs.push_back(i);
      vw.push_back(villain[i]);
    }
  }
  AEGIS_CHECK_MSG(!hs.empty() && !vs.empty(), "empty range");
  Rng rng(seed);
  // Rejection sampling of compatible (hero, villain) pairs gives the correct
  // joint distribution P(h, v) prop. to r_h(h) r_v(v) [disjoint].
  double sum = 0, sumsq = 0;
  long long w = 0, t = 0;
  long long done = 0, attempts = 0;
  while (done < samples) {
    AEGIS_CHECK_MSG(++attempts < samples * 1000 + 1000000, "ranges are (almost) incompatible");
    const int h = hs[rng.sample(hw)];
    const int v = vs[rng.sample(vw)];
    if (C.mask[h] & C.mask[v]) continue;
    CardMask used = bm | C.mask[h] | C.mask[v];
    CardMask full = bm;
    for (int k = static_cast<int>(board.size()); k < 5; ++k) {
      Card c;
      do {
        c = static_cast<Card>(rng.below(kNumCards));
      } while (used & card_bit(c));
      used |= card_bit(c);
      full |= card_bit(c);
    }
    const HandValue a = evaluate(full | C.mask[h]);
    const HandValue b = evaluate(full | C.mask[v]);
    const double x = a > b ? 1.0 : (a == b ? 0.5 : 0.0);
    w += a > b;
    t += a == b;
    sum += x;
    sumsq += x * x;
    ++done;
  }
  EquityResult r;
  r.win = static_cast<double>(w) / samples;
  r.tie = static_cast<double>(t) / samples;
  r.lose = 1.0 - r.win - r.tie;
  const double mean = sum / samples;
  r.std_error = std::sqrt(std::max(0.0, sumsq / samples - mean * mean) / samples);
  r.samples = samples;
  return r;
}

}  // namespace aegis::poker
