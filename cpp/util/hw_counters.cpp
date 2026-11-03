#include "util/hw_counters.h"

#include <sys/resource.h>

#include <chrono>
#include <cstring>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace aegis {

namespace {
double user_cpu_now() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6;
}
long max_rss_kb_now() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return ru.ru_maxrss / 1024;  // bytes on macOS
#else
  return ru.ru_maxrss;  // kilobytes on Linux
#endif
}
}  // namespace

struct HwCounters::Impl {
  std::chrono::steady_clock::time_point t0;
  double cpu0 = 0.0;
#if defined(__linux__)
  int fds[4] = {-1, -1, -1, -1};
  uint64_t base[4] = {0, 0, 0, 0};
#elif defined(__APPLE__)
  void* lib = nullptr;
  int (*kpc_force_all_ctrs_set)(int) = nullptr;
  int (*kpc_set_counting)(uint32_t) = nullptr;
  int (*kpc_set_thread_counting)(uint32_t) = nullptr;
  uint32_t (*kpc_get_counter_count)(uint32_t) = nullptr;
  int (*kpc_get_thread_counters)(int, unsigned int, uint64_t*) = nullptr;
  uint64_t base[32] = {};
  unsigned int count = 0;
#endif
};

HwCounters::HwCounters() : impl_(new Impl) {
#if defined(__linux__)
  const uint64_t configs[4] = {PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
                               PERF_COUNT_HW_CACHE_MISSES, PERF_COUNT_HW_BRANCH_MISSES};
  bool ok = true;
  for (int i = 0; i < 4; ++i) {
    perf_event_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = configs[i];
    attr.disabled = 0;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    impl_->fds[i] = static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0));
    if (impl_->fds[i] < 0) ok = false;
  }
  available_ = ok;
  backend_ = ok ? "linux-perf_event" : "none";
  if (!ok) reason_ = "perf_event_open failed (check /proc/sys/kernel/perf_event_paranoid)";
#elif defined(__APPLE__)
  impl_->lib = dlopen("/System/Library/PrivateFrameworks/kperf.framework/kperf", RTLD_LAZY);
  if (!impl_->lib) {
    reason_ = "kperf.framework not found";
    return;
  }
  impl_->kpc_force_all_ctrs_set =
      reinterpret_cast<int (*)(int)>(dlsym(impl_->lib, "kpc_force_all_ctrs_set"));
  impl_->kpc_set_counting = reinterpret_cast<int (*)(uint32_t)>(dlsym(impl_->lib, "kpc_set_counting"));
  impl_->kpc_set_thread_counting =
      reinterpret_cast<int (*)(uint32_t)>(dlsym(impl_->lib, "kpc_set_thread_counting"));
  impl_->kpc_get_counter_count =
      reinterpret_cast<uint32_t (*)(uint32_t)>(dlsym(impl_->lib, "kpc_get_counter_count"));
  impl_->kpc_get_thread_counters = reinterpret_cast<int (*)(int, unsigned int, uint64_t*)>(
      dlsym(impl_->lib, "kpc_get_thread_counters"));
  if (!impl_->kpc_force_all_ctrs_set || !impl_->kpc_set_counting || !impl_->kpc_set_thread_counting ||
      !impl_->kpc_get_counter_count || !impl_->kpc_get_thread_counters) {
    reason_ = "kperf symbols missing";
    return;
  }
  constexpr uint32_t kFixedMask = 1u;  // KPC_CLASS_FIXED_MASK: cycles, instructions
  if (impl_->kpc_force_all_ctrs_set(1) != 0) {
    reason_ = "kperf requires root on macOS (run the benchmark with sudo); counters disabled";
    return;
  }
  if (impl_->kpc_set_counting(kFixedMask) != 0 || impl_->kpc_set_thread_counting(kFixedMask) != 0) {
    reason_ = "kpc_set_counting failed";
    return;
  }
  impl_->count = impl_->kpc_get_counter_count(kFixedMask);
  if (impl_->count < 2 || impl_->count > 32) {
    reason_ = "unexpected fixed counter count";
    return;
  }
  available_ = true;
  backend_ = "macos-kperf(fixed: cycles,instructions)";
#else
  reason_ = "no hardware counter backend for this platform";
#endif
}

HwCounters::~HwCounters() {
#if defined(__linux__)
  for (int fd : impl_->fds)
    if (fd >= 0) close(fd);
#endif
  delete impl_;
}

void HwCounters::start() {
  impl_->t0 = std::chrono::steady_clock::now();
  impl_->cpu0 = user_cpu_now();
  if (!available_) return;
#if defined(__linux__)
  for (int i = 0; i < 4; ++i) {
    uint64_t v = 0;
    if (read(impl_->fds[i], &v, sizeof(v)) != sizeof(v)) v = 0;
    impl_->base[i] = v;
  }
#elif defined(__APPLE__)
  impl_->kpc_get_thread_counters(0, impl_->count, impl_->base);
#endif
}

HwSample HwCounters::stop() {
  HwSample s;
  s.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - impl_->t0).count();
  s.user_cpu_seconds = user_cpu_now() - impl_->cpu0;
  s.max_rss_kb = max_rss_kb_now();
  if (!available_) return s;
#if defined(__linux__)
  uint64_t v[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    if (read(impl_->fds[i], &v[i], sizeof(uint64_t)) != sizeof(uint64_t)) v[i] = impl_->base[i];
  }
  s.cycles = v[0] - impl_->base[0];
  s.instructions = v[1] - impl_->base[1];
  s.cache_misses = v[2] - impl_->base[2];
  s.branch_misses = v[3] - impl_->base[3];
  s.counters_valid = true;
#elif defined(__APPLE__)
  uint64_t now[32] = {};
  impl_->kpc_get_thread_counters(0, impl_->count, now);
  s.cycles = now[0] - impl_->base[0];
  s.instructions = now[1] - impl_->base[1];
  s.counters_valid = true;
#endif
  return s;
}

}  // namespace aegis
