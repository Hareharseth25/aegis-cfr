#include "value_net/mlp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "poker/hand_evaluator/evaluator.h"

#include "util/check.h"

#if defined(AEGIS_HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace aegis::nn {

namespace {

// Hand strengths of all combos on a 5-card board, cached per board (thread-local).
struct BoardInfo {
  std::vector<int> order;                 // live combos sorted by strength (ascending)
  std::vector<poker::HandValue> value;    // per combo (0 = dead)
};

const BoardInfo& board_info(poker::CardMask bm) {
  thread_local std::unordered_map<poker::CardMask, BoardInfo> cache;
  auto it = cache.find(bm);
  if (it != cache.end()) return it->second;
  if (cache.size() > 4096) cache.clear();
  BoardInfo b;
  b.value.assign(poker::kNumCombos, 0);
  const auto& C = poker::combos();
  for (int i = 0; i < poker::kNumCombos; ++i) {
    if (C.mask[i] & bm) continue;
    b.value[i] = poker::evaluate(bm | C.mask[i]);
    b.order.push_back(i);
  }
  std::stable_sort(b.order.begin(), b.order.end(), [&](int x, int y) { return b.value[x] < b.value[y]; });
  return cache.emplace(bm, std::move(b)).first->second;
}

// e(h) = (W - L) / (W + T + L) over the opponent's compatible mass.
void checkdown_values(const BoardInfo& b, const float* opp, float* out) {
  const auto& C = poker::combos();
  const auto& ord = b.order;
  const int n = static_cast<int>(ord.size());
  std::vector<double> below(poker::kNumCombos, 0.0), above(poker::kNumCombos, 0.0);
  double tot = 0.0, card_tot[poker::kNumCards] = {};
  for (int i : ord) {
    tot += opp[i];
    card_tot[C.cards[i][0]] += opp[i];
    card_tot[C.cards[i][1]] += opp[i];
  }
  {
    double run = 0.0, card[poker::kNumCards] = {};
    int j = 0;
    for (int k = 0; k < n; ++k) {
      const int h = ord[k];
      while (j < n && b.value[ord[j]] < b.value[h]) {
        const int o = ord[j++];
        run += opp[o];
        card[C.cards[o][0]] += opp[o];
        card[C.cards[o][1]] += opp[o];
      }
      below[h] = run - card[C.cards[h][0]] - card[C.cards[h][1]];
    }
  }
  {
    double run = 0.0, card[poker::kNumCards] = {};
    int j = n - 1;
    for (int k = n - 1; k >= 0; --k) {
      const int h = ord[k];
      while (j >= 0 && b.value[ord[j]] > b.value[h]) {
        const int o = ord[j--];
        run += opp[o];
        card[C.cards[o][0]] += opp[o];
        card[C.cards[o][1]] += opp[o];
      }
      above[h] = run - card[C.cards[h][0]] - card[C.cards[h][1]];
    }
  }
  for (int h : ord) {
    const double compat = tot - card_tot[C.cards[h][0]] - card_tot[C.cards[h][1]] + opp[h];
    out[h] = compat > 1e-12 ? static_cast<float>((std::max(0.0, below[h]) - std::max(0.0, above[h])) / compat) : 0.0f;
  }
}

}  // namespace

void build_features(const std::vector<poker::Card>& board, const double* r0, const double* r1, double pot,
                    double stack, float* out) {
  std::memset(out, 0, sizeof(float) * kInputSize);
  for (poker::Card c : board) out[c] = 1.0f;
  const poker::CardMask bm = poker::mask_of(board);
  const auto& C = poker::combos();
  const double* rs[2] = {r0, r1};
  for (int p = 0; p < 2; ++p) {
    double total = 0.0;
    for (int i = 0; i < poker::kNumCombos; ++i)
      if (!(C.mask[i] & bm)) total += rs[p][i];
    float* dst = out + kBoardFeatures + p * kRangeFeatures;
    if (total > 0.0)
      for (int i = 0; i < poker::kNumCombos; ++i)
        dst[i] = (C.mask[i] & bm) ? 0.0f : static_cast<float>(rs[p][i] / total);
  }
  if (board.size() == 5) {
    const BoardInfo& bi = board_info(bm);
    for (int p = 0; p < 2; ++p) {
      const float* opp = out + kBoardFeatures + (1 - p) * kRangeFeatures;
      checkdown_values(bi, opp, out + kBoardFeatures + 2 * kRangeFeatures + p * kEquityFeatures);
    }
  }
  float* sc = out + kBoardFeatures + 2 * kRangeFeatures + 2 * kEquityFeatures;
  sc[0] = static_cast<float>(pot / (pot + 2.0 * stack));
  sc[1] = static_cast<float>(std::log1p(stack / pot));
}

long long MLP::parameters() const {
  long long n = 0;
  for (const auto& s : shapes_) n += static_cast<long long>(s[0]) * s[1] + s[0];
  return n;
}

MLP MLP::from_arrays(const std::vector<std::vector<float>>& weights, const std::vector<std::vector<float>>& biases,
                     const std::vector<std::array<int, 2>>& shapes) {
  AEGIS_CHECK(weights.size() == biases.size() && weights.size() == shapes.size() && !shapes.empty());
  MLP m;
  for (size_t l = 0; l < shapes.size(); ++l) {
    AEGIS_CHECK(static_cast<long long>(weights[l].size()) == static_cast<long long>(shapes[l][0]) * shapes[l][1]);
    AEGIS_CHECK(static_cast<int>(biases[l].size()) == shapes[l][0]);
    if (l > 0) AEGIS_CHECK_MSG(shapes[l][1] == shapes[l - 1][0], "layer shapes do not chain");
  }
  m.w_ = weights;
  m.b_ = biases;
  m.shapes_ = shapes;
  return m;
}

MLP MLP::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  AEGIS_CHECK_MSG(f.good(), "cannot open value network '" << path << "'");
  char magic[8];
  f.read(magic, 8);
  AEGIS_CHECK_MSG(std::memcmp(magic, "AEGISMLP", 8) == 0, "not an AEGIS MLP file: " << path);
  uint32_t version = 0, layers = 0;
  f.read(reinterpret_cast<char*>(&version), 4);
  f.read(reinterpret_cast<char*>(&layers), 4);
  AEGIS_CHECK_MSG(version == 1 && layers > 0 && layers < 64, "unsupported MLP file");
  std::vector<std::vector<float>> W(layers), B(layers);
  std::vector<std::array<int, 2>> S(layers);
  for (uint32_t l = 0; l < layers; ++l) {
    uint32_t in = 0, out = 0;
    f.read(reinterpret_cast<char*>(&in), 4);
    f.read(reinterpret_cast<char*>(&out), 4);
    S[l] = {static_cast<int>(out), static_cast<int>(in)};
    W[l].resize(static_cast<size_t>(in) * out);
    B[l].resize(out);
    f.read(reinterpret_cast<char*>(W[l].data()), static_cast<std::streamsize>(W[l].size() * 4));
    f.read(reinterpret_cast<char*>(B[l].data()), static_cast<std::streamsize>(B[l].size() * 4));
    AEGIS_CHECK_MSG(f.good(), "truncated MLP file");
  }
  return from_arrays(W, B, S);
}

void MLP::save(const std::string& path) const {
  std::ofstream f(path, std::ios::binary);
  AEGIS_CHECK_MSG(f.good(), "cannot write " << path);
  f.write("AEGISMLP", 8);
  const uint32_t version = 1, layers = static_cast<uint32_t>(shapes_.size());
  f.write(reinterpret_cast<const char*>(&version), 4);
  f.write(reinterpret_cast<const char*>(&layers), 4);
  for (size_t l = 0; l < shapes_.size(); ++l) {
    const uint32_t in = shapes_[l][1], out = shapes_[l][0];
    f.write(reinterpret_cast<const char*>(&in), 4);
    f.write(reinterpret_cast<const char*>(&out), 4);
    f.write(reinterpret_cast<const char*>(w_[l].data()), static_cast<std::streamsize>(w_[l].size() * 4));
    f.write(reinterpret_cast<const char*>(b_[l].data()), static_cast<std::streamsize>(b_[l].size() * 4));
  }
}

std::vector<float> MLP::forward(const std::vector<float>& x, int batch) const {
  AEGIS_CHECK(!shapes_.empty());
  AEGIS_CHECK(static_cast<long long>(x.size()) == static_cast<long long>(batch) * input_size());
  std::vector<float> cur = x, next;
  for (size_t l = 0; l < shapes_.size(); ++l) {
    const int out = shapes_[l][0], in = shapes_[l][1];
    next.assign(static_cast<size_t>(batch) * out, 0.0f);
    // next = cur * Wᵀ + b
    for (int r = 0; r < batch; ++r) std::memcpy(&next[static_cast<size_t>(r) * out], b_[l].data(), sizeof(float) * out);
#if defined(AEGIS_HAVE_ACCELERATE)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, out, in, 1.0f, cur.data(), in, w_[l].data(), in, 1.0f,
                next.data(), out);
#else
    for (int r = 0; r < batch; ++r) {
      const float* xr = &cur[static_cast<size_t>(r) * in];
      float* yr = &next[static_cast<size_t>(r) * out];
      for (int o = 0; o < out; ++o) {
        const float* wr = &w_[l][static_cast<size_t>(o) * in];
        float s = 0.0f;
        for (int k = 0; k < in; ++k) s += wr[k] * xr[k];
        yr[o] += s;
      }
    }
#endif
    if (l + 1 < shapes_.size())
      for (auto& v : next) v = v > 0.0f ? v : 0.0f;
    cur.swap(next);
  }
  return cur;
}

void zero_sum_project(const double* r0n, const double* r1n, float* v0, float* v1) {
  double e = 0.0, m0 = 0.0, m1 = 0.0;
  for (int i = 0; i < poker::kNumCombos; ++i) {
    e += r0n[i] * v0[i] + r1n[i] * v1[i];
    m0 += r0n[i];
    m1 += r1n[i];
  }
  if (m0 <= 0.0 || m1 <= 0.0) return;
  const float d0 = static_cast<float>(0.5 * e / m0), d1 = static_cast<float>(0.5 * e / m1);
  for (int i = 0; i < poker::kNumCombos; ++i) {
    v0[i] -= d0;
    v1[i] -= d1;
  }
}

}  // namespace aegis::nn
