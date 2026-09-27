#pragma once
// A ~60-line test harness so the project has zero third-party dependencies.
// Each test binary links testlib_main.cc, which runs every registered TEST and
// exits non-zero on any failure. Usage: ./test_x [substring-filter]

#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace testlib {

struct Case {
  const char* name;
  std::function<void()> fn;
};

std::vector<Case>& registry();

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void fail(const char* file, int line, const std::string& msg) {
  std::ostringstream os;
  os << file << ":" << line << ": " << msg;
  throw Failure(os.str());
}

}  // namespace testlib

#define TL_CAT2(a, b) a##b
#define TL_CAT(a, b) TL_CAT2(a, b)

#define TEST(name)                                                         \
  static void name();                                                      \
  static ::testlib::Registrar TL_CAT(tl_reg_, name)(#name, name);          \
  static void name()

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) ::testlib::fail(__FILE__, __LINE__, "CHECK(" #cond ")");  \
  } while (0)

#define CHECK_EQ(a, b)                                                     \
  do {                                                                     \
    const auto& tl_a = (a);                                                \
    const auto& tl_b = (b);                                                \
    if (!(tl_a == tl_b)) {                                                 \
      std::ostringstream tl_os;                                            \
      tl_os << "CHECK_EQ(" #a ", " #b ") got " << tl_a << " vs " << tl_b;  \
      ::testlib::fail(__FILE__, __LINE__, tl_os.str());                    \
    }                                                                      \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                              \
  do {                                                                     \
    const double tl_a = (a), tl_b = (b);                                   \
    if (!(std::fabs(tl_a - tl_b) <= (tol))) {                              \
      std::ostringstream tl_os;                                            \
      tl_os << "CHECK_NEAR(" #a ", " #b ") got " << tl_a << " vs " << tl_b; \
      ::testlib::fail(__FILE__, __LINE__, tl_os.str());                    \
    }                                                                      \
  } while (0)

#define CHECK_THROWS(stmt, ExType)                                         \
  do {                                                                     \
    bool tl_caught = false;                                                \
    try {                                                                  \
      stmt;                                                                \
    } catch (const ExType&) {                                              \
      tl_caught = true;                                                    \
    }                                                                      \
    if (!tl_caught)                                                        \
      ::testlib::fail(__FILE__, __LINE__, "expected " #ExType " from: " #stmt); \
  } while (0)
