// Counterfactual value network: feature layout and batched MLP inference.
//
// The network maps a river public belief state to per-hand counterfactual
// values of BOTH players (DeepStack-style), not a single scalar:
//
//   input  x = [ board one-hot (52) | r̂_0 (1326) | r̂_1 (1326) |
//                e_0 (1326) | e_1 (1326) | pot share | log(1+SPR) ]
//   output y = [ v_0 (1326) | v_1 (1326) ]
//
// r̂_p are the players' ranges normalised to sum 1 over live combos; e_p(h) is
// hand h's check-down value against the opponent's range, (P(win) - P(lose))
// over the compatible opponent mass with exact card removal - the value the
// hand would have if both players checked down. It is computed analytically
// (O(n) sweep, strengths cached per board), so the network only has to learn
// how betting changes values, not hand strength from scratch. v_p(h)
// is the counterfactual value of hand h normalised by the pot and the
// opponent's range mass: cfv_p(h) = v_p(h) * pot * Σ r_o. The (public) betting
// history before the river is summarised by pot and stack: in this
// formulation the public belief state (board, ranges, pot, stacks) is a
// sufficient statistic for the river subgame's values, so explicit history
// features would be redundant.
//
// Weights are trained in PyTorch (python/training/train_value_net.py) and
// exported to a simple binary file read here. Inference uses one SGEMM per
// layer over the whole batch (Apple Accelerate when available).
#pragma once

#include <array>
#include <string>
#include <vector>

#include "poker/cards/cards.h"

namespace aegis::nn {

inline constexpr int kBoardFeatures = 52;
inline constexpr int kRangeFeatures = poker::kNumCombos;
inline constexpr int kEquityFeatures = poker::kNumCombos;
inline constexpr int kScalarFeatures = 2;
inline constexpr int kInputSize =
    kBoardFeatures + 2 * kRangeFeatures + 2 * kEquityFeatures + kScalarFeatures;  // 5358
inline constexpr int kOutputSize = 2 * poker::kNumCombos;                                  // 2652

// Writes one feature row. r0/r1 are 1326-vectors (any scale; normalised here).
void build_features(const std::vector<poker::Card>& board, const double* r0, const double* r1, double pot,
                    double stack, float* out);

class MLP {
 public:
  MLP() = default;
  static MLP load(const std::string& path);
  // From raw arrays: weights[l] is row-major (out x in), biases[l] has `out`.
  static MLP from_arrays(const std::vector<std::vector<float>>& weights, const std::vector<std::vector<float>>& biases,
                         const std::vector<std::array<int, 2>>& shapes);
  void save(const std::string& path) const;

  int input_size() const { return shapes_.empty() ? 0 : shapes_.front()[1]; }
  int output_size() const { return shapes_.empty() ? 0 : shapes_.back()[0]; }
  int num_layers() const { return static_cast<int>(shapes_.size()); }
  long long parameters() const;

  // x: batch x input (row-major); returns batch x output. ReLU between layers.
  std::vector<float> forward(const std::vector<float>& x, int batch) const;

 private:
  std::vector<std::vector<float>> w_, b_;
  std::vector<std::array<int, 2>> shapes_;  // (out, in)
};

// Zero-sum projection: shifts both players' normalised values so that
// Σ r̂_0 v_0 + Σ r̂_1 v_1 = 0 (DeepStack's output correction).
void zero_sum_project(const double* r0n, const double* r1n, float* v0, float* v1);

}  // namespace aegis::nn
