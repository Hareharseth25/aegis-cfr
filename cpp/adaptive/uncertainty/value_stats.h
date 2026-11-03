// Running statistics used to quantify uncertainty of sampled quantities.
#pragma once

#include <cmath>
#include <cstdint>

namespace aegis {

// Welford's numerically stable running mean / variance.
struct RunningStats {
  int64_t n = 0;
  double mean = 0.0;
  double m2 = 0.0;
  void add(double x) {
    ++n;
    const double d = x - mean;
    mean += d / static_cast<double>(n);
    m2 += d * (x - mean);
  }
  double variance() const { return n > 1 ? m2 / static_cast<double>(n - 1) : 0.0; }
  double std_error() const { return n > 1 ? std::sqrt(variance() / static_cast<double>(n)) : 0.0; }
};

// Exponential moving average with an "unset" state.
struct Ema {
  double value = 0.0;
  bool set = false;
  void add(double x, double rate) {
    if (!set) {
      value = x;
      set = true;
    } else {
      value += rate * (x - value);
    }
  }
};

}  // namespace aegis
