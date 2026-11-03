// Runtime checks that stay enabled in release builds. A solver that silently
// continues after an invariant violation produces numbers nobody can trust.
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace aegis {

class AegisError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

namespace detail {
[[noreturn]] inline void check_failed(const char* expr, const char* file, int line,
                                      const std::string& msg) {
  std::ostringstream os;
  os << "AEGIS check failed: (" << expr << ") at " << file << ":" << line;
  if (!msg.empty()) os << " - " << msg;
  throw AegisError(os.str());
}
}  // namespace detail

}  // namespace aegis

#define AEGIS_CHECK(cond)                                                     \
  do {                                                                        \
    if (!(cond)) ::aegis::detail::check_failed(#cond, __FILE__, __LINE__, ""); \
  } while (0)

#define AEGIS_CHECK_MSG(cond, msg)                                    \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::ostringstream aegis_os_;                                   \
      aegis_os_ << msg;                                               \
      ::aegis::detail::check_failed(#cond, __FILE__, __LINE__,        \
                                    aegis_os_.str());                 \
    }                                                                 \
  } while (0)
