#include "value_net/data_gen.h"

#include <algorithm>

#include "poker/hand_evaluator/evaluator.h"
#include "poker/ranges/range.h"
#include "solvers/subgame_resolver/leaf_values.h"
#include "util/check.h"
#include "util/rng.h"
#include "util/thread_pool.h"
#include "value_net/mlp.h"

namespace aegis::nn {

using namespace poker;

namespace {

std::vector<Card> sample_board(const std::vector<Card>& turn, Rng& rng) {
  std::vector<Card> b = turn;
  CardMask used = mask_of(b);
  while (b.size() < 5) {
    const Card c = static_cast<Card>(rng.below(kNumCards));
    if (used & card_bit(c)) continue;
    used |= card_bit(c);
    b.push_back(c);
  }
  return b;
}

Range sample_range(const std::vector<Card>& board, Rng& rng, double uniform_mix_prob) {
  const CardMask bm = mask_of(board);
  std::vector<int> order;
  std::vector<HandValue> val(kNumCombos, 0);
  for (int i = 0; i < kNumCombos; ++i) {
    if (combos().mask[i] & bm) continue;
    val[i] = evaluate(bm | combos().mask[i]);
    order.push_back(i);
  }
  // Strongest first; random tie order among equal strengths.
  rng.next();
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return val[a] > val[b]; });
  Range r = random_range_by_strength(order, rng);
  if (rng.uniform() < uniform_mix_prob) {
    const double lam = rng.uniform();
    for (int i : order) r[i] = (1.0 - lam) * r[i] + lam;
  }
  // Floor keeps every live hand reachable (well-defined average strategies).
  for (int i : order) r[i] = std::max(r[i], 1e-6);
  return r;
}

}  // namespace

RiverSamples generate_river_samples(const RiverSampleConfig& cfg, int n) {
  AEGIS_CHECK(n > 0);
  AEGIS_CHECK_MSG(!cfg.pot_stack.empty(), "need at least one (pot, stack) option");
  AEGIS_CHECK(cfg.turn_board.size() <= 4);
  RiverSamples out;
  out.n = n;
  out.features.assign(static_cast<size_t>(n) * kInputSize, 0.0f);
  out.targets.assign(static_cast<size_t>(n) * kOutputSize, 0.0f);
  out.mask.assign(static_cast<size_t>(n) * kOutputSize, 0.0f);
  out.boards.assign(static_cast<size_t>(n) * 5, -1);
  out.exploitability_pct_pot.assign(n, 0.0);
  out.pot.assign(n, 0.0);
  out.stack.assign(n, 0.0);
  ThreadPool pool(std::max(1, cfg.threads));
  pool.parallel_for(n, [&](int k) {
    uint64_t s = cfg.seed * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(k) * 0xBF58476D1CE4E5B9ULL + 1;
    Rng rng(splitmix64(s));
    const auto board = sample_board(cfg.turn_board, rng);
    const auto [pot, stack] = cfg.pot_stack[rng.below(cfg.pot_stack.size())];
    const Range r0 = sample_range(board, rng, cfg.uniform_mix_prob);
    const Range r1 = sample_range(board, rng, cfg.uniform_mix_prob);
    const auto res = solve_river_values(board, r0.w.data(), r1.w.data(), pot, stack, cfg.river_sizes, cfg.iterations,
                                        cfg.cfr);
    build_features(board, r0.w.data(), r1.w.data(), pot, stack, &out.features[static_cast<size_t>(k) * kInputSize]);
    const CardMask bm = mask_of(board);
    double mass[2] = {0.0, 0.0};
    for (int i = 0; i < kNumCombos; ++i)
      if (!(combos().mask[i] & bm)) {
        mass[0] += r0[i];
        mass[1] += r1[i];
      }
    float* t = &out.targets[static_cast<size_t>(k) * kOutputSize];
    float* m = &out.mask[static_cast<size_t>(k) * kOutputSize];
    for (int p = 0; p < 2; ++p)
      for (int i = 0; i < kNumCombos; ++i) {
        if (combos().mask[i] & bm) continue;
        t[p * kNumCombos + i] = static_cast<float>(res.cfv[p][i] / (pot * mass[1 - p]));
        m[p * kNumCombos + i] = 1.0f;
      }
    for (int j = 0; j < 5; ++j) out.boards[static_cast<size_t>(k) * 5 + j] = board[j];
    out.exploitability_pct_pot[k] = res.exploitability_pct_pot;
    out.pot[k] = pot;
    out.stack[k] = stack;
  });
  return out;
}

}  // namespace aegis::nn
