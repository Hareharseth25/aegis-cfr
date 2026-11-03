#pragma once

#include <chrono>

namespace aegis {

class Stopwatch {
 public:
  Stopwatch() : start_(Clock::now()) {}
  void reset() { start_ = Clock::now(); }
  double seconds() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }
  double millis() const { return seconds() * 1e3; }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

}  // namespace aegis
