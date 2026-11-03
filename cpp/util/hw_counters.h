// Hardware performance counters (cycles, instructions, cache misses, branch
// misses) for profiling solver kernels.
//
//  * Linux: perf_event_open (no root needed when perf_event_paranoid <= 2).
//  * macOS (Apple Silicon): the private kperf framework. The kernel only grants
//    counter access to root, so run benchmarks with `sudo` to enable it.
//  * Otherwise: counters report available() == false and only wall-clock and
//    rusage numbers are recorded. Benchmarks print which backend was used so a
//    reader never mistakes a missing counter for a zero.
#pragma once

#include <cstdint>
#include <string>

namespace aegis {

struct HwSample {
  double wall_seconds = 0.0;
  double user_cpu_seconds = 0.0;
  uint64_t cycles = 0;
  uint64_t instructions = 0;
  uint64_t cache_misses = 0;
  uint64_t branch_misses = 0;
  long max_rss_kb = 0;
  bool counters_valid = false;
  double ipc() const { return cycles ? static_cast<double>(instructions) / cycles : 0.0; }
};

class HwCounters {
 public:
  HwCounters();
  ~HwCounters();
  HwCounters(const HwCounters&) = delete;
  HwCounters& operator=(const HwCounters&) = delete;

  bool available() const { return available_; }
  const std::string& backend() const { return backend_; }
  const std::string& reason() const { return reason_; }

  void start();
  HwSample stop();

 private:
  bool available_ = false;
  std::string backend_ = "none";
  std::string reason_;
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace aegis
