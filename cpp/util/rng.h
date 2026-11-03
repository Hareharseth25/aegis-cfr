// Deterministic, seedable random number generation (xoshiro256** seeded by
// SplitMix64). Every sampled experiment records its seed so results can be
// reproduced bit-for-bit.
#pragma once

#include <cmath>
#include <cstdint>
#include <span>

#include "util/check.h"

namespace aegis {

inline uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

class Rng {
 public:
  using result_type = uint64_t;
  explicit Rng(uint64_t seed = 0x5EEDULL) { reseed(seed); }

  void reseed(uint64_t seed) {
    uint64_t x = seed;
    for (auto& s : s_) s = splitmix64(x);
  }

  static constexpr uint64_t min() { return 0; }
  static constexpr uint64_t max() { return ~0ULL; }

  uint64_t operator()() { return next(); }

  uint64_t next() {
    const uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  // Uniform double in [0, 1) with 53 bits of randomness.
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

  // Uniform integer in [0, n).
  uint64_t below(uint64_t n) {
    AEGIS_CHECK(n > 0);
    // Lemire's nearly-divisionless method.
    __uint128_t m = static_cast<__uint128_t>(next()) * n;
    uint64_t l = static_cast<uint64_t>(m);
    if (l < n) {
      const uint64_t t = (0 - n) % n;
      while (l < t) {
        m = static_cast<__uint128_t>(next()) * n;
        l = static_cast<uint64_t>(m);
      }
    }
    return static_cast<uint64_t>(m >> 64);
  }

  // Samples an index from an (unnormalised, non-negative) weight vector.
  int sample(std::span<const double> weights) {
    double total = 0.0;
    for (double w : weights) total += w;
    AEGIS_CHECK_MSG(total > 0.0, "cannot sample from an all-zero distribution");
    double u = uniform() * total;
    int last_positive = -1;
    for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
      if (weights[i] <= 0.0) continue;
      last_positive = i;
      if (u < weights[i]) return i;
      u -= weights[i];
    }
    return last_positive;  // guards against floating-point round-off
  }

  // Standard gamma variate (Marsaglia-Tsang), used for Dirichlet sampling.
  double gamma(double shape);

 private:
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t s_[4];
};

inline double Rng::gamma(double shape) {
  AEGIS_CHECK(shape > 0.0);
  if (shape < 1.0) {
    const double u = uniform();
    return gamma(shape + 1.0) * std::pow(u <= 0.0 ? 1e-300 : u, 1.0 / shape);
  }
  const double d = shape - 1.0 / 3.0;
  const double c = 1.0 / std::sqrt(9.0 * d);
  while (true) {
    double x, v;
    do {
      // Box-Muller normal.
      double u1 = uniform();
      double u2 = uniform();
      if (u1 <= 0.0) u1 = 1e-300;
      x = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
      v = 1.0 + c * x;
    } while (v <= 0.0);
    v = v * v * v;
    const double u = uniform();
    if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
    if (std::log(u <= 0.0 ? 1e-300 : u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
  }
}

}  // namespace aegis
