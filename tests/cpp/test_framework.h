// Minimal self-contained test framework (no external dependencies).
//
//   AEGIS_TEST(name) { ... }        regular test
//   AEGIS_SLOW_TEST(name) { ... }   exhaustive/long test (run with --only-slow)
//   EXPECT_TRUE(x), EXPECT_NEAR(a, b, tol), EXPECT_EQ(a, b)
#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace aegis_test {

struct TestCase {
  std::string name;
  std::function<void()> fn;
  bool slow;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn, bool slow) { registry().push_back({name, fn, slow}); }
};

inline int& failures() {
  static int f = 0;
  return f;
}

inline void report_failure(const char* file, int line, const std::string& msg) {
  ++failures();
  std::cerr << "    FAIL " << file << ":" << line << ": " << msg << "\n";
}

}  // namespace aegis_test

#define AEGIS_TEST_CAT_(a, b) a##b
#define AEGIS_TEST_CAT(a, b) AEGIS_TEST_CAT_(a, b)
#define AEGIS_TEST_IMPL_(name, slow)                                                       \
  static void name();                                                                      \
  static ::aegis_test::Registrar AEGIS_TEST_CAT(registrar_, name)(#name, name, slow);      \
  static void name()
#define AEGIS_TEST(name) AEGIS_TEST_IMPL_(name, false)
#define AEGIS_SLOW_TEST(name) AEGIS_TEST_IMPL_(name, true)

#define EXPECT_TRUE(x)                                                                 \
  do {                                                                                 \
    if (!(x)) ::aegis_test::report_failure(__FILE__, __LINE__, "expected true: " #x); \
  } while (0)

#define EXPECT_EQ(a, b)                                                                     \
  do {                                                                                      \
    const auto va_ = (a);                                                                   \
    const auto vb_ = (b);                                                                   \
    if (!(va_ == vb_)) {                                                                    \
      std::ostringstream os_;                                                               \
      os_ << #a " == " #b " (" << va_ << " vs " << vb_ << ")";                              \
      ::aegis_test::report_failure(__FILE__, __LINE__, os_.str());                          \
    }                                                                                       \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                                               \
  do {                                                                                       \
    const double va_ = (a);                                                                  \
    const double vb_ = (b);                                                                  \
    if (!(std::abs(va_ - vb_) <= (tol))) {                                                   \
      std::ostringstream os_;                                                                \
      os_.precision(12);                                                                     \
      os_ << #a " ~ " #b " (" << va_ << " vs " << vb_ << ", tol " << (tol) << ")";           \
      ::aegis_test::report_failure(__FILE__, __LINE__, os_.str());                           \
    }                                                                                        \
  } while (0)

#define EXPECT_THROWS(stmt)                                                                  \
  do {                                                                                       \
    bool threw_ = false;                                                                     \
    try {                                                                                    \
      stmt;                                                                                  \
    } catch (...) {                                                                          \
      threw_ = true;                                                                         \
    }                                                                                        \
    if (!threw_) ::aegis_test::report_failure(__FILE__, __LINE__, "expected exception: " #stmt); \
  } while (0)
