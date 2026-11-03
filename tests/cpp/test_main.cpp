#include <chrono>
#include <cstring>
#include <exception>

#include "test_framework.h"

int main(int argc, char** argv) {
  bool only_slow = false, exclude_slow = false;
  std::string filter;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--only-slow")) only_slow = true;
    else if (!std::strcmp(argv[i], "--exclude-slow")) exclude_slow = true;
    else filter = argv[i];
  }
  int run = 0, failed_tests = 0;
  for (const auto& tc : aegis_test::registry()) {
    if (only_slow && !tc.slow) continue;
    if (exclude_slow && tc.slow) continue;
    if (!filter.empty() && tc.name.find(filter) == std::string::npos) continue;
    const int before = aegis_test::failures();
    const auto t0 = std::chrono::steady_clock::now();
    try {
      tc.fn();
    } catch (const std::exception& e) {
      aegis_test::report_failure(tc.name.c_str(), 0, std::string("uncaught exception: ") + e.what());
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const bool ok = aegis_test::failures() == before;
    if (!ok) ++failed_tests;
    ++run;
    std::cout << (ok ? "[ OK ] " : "[FAIL] ") << tc.name << " (" << static_cast<long>(ms) << " ms)\n";
  }
  std::cout << run - failed_tests << "/" << run << " tests passed\n";
  return failed_tests == 0 ? 0 : 1;
}
