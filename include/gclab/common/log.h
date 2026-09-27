#pragma once
// Minimal structured logger: one line per event, logfmt style.
//
//   ts_us=1732350000123456 level=INFO event=world.run nranks=4 rank=-1
//
// logfmt is greppable and trivially parsed from Python (shlex/regex), which is
// all the analysis scripts need. Level comes from GCLAB_LOG_LEVEL
// (DEBUG|INFO|WARN|ERROR, default WARN) -- the same idea as NCCL_DEBUG.

#include <initializer_list>
#include <string>
#include <utility>

namespace gclab::log {

enum class Level { Debug = 0, Info = 1, Warn = 2, Error = 3 };

using Field = std::pair<const char*, std::string>;

bool enabled(Level level);
void emit(Level level, const char* event, std::initializer_list<Field> fields);

template <typename T>
std::string str(const T& v) {
  return std::to_string(v);
}
inline std::string str(const std::string& v) { return v; }
inline std::string str(const char* v) { return v; }

}  // namespace gclab::log

#define GCLAB_LOG(level, event, ...)                                  \
  do {                                                                \
    if (::gclab::log::enabled(::gclab::log::Level::level)) {          \
      ::gclab::log::emit(::gclab::log::Level::level, event, __VA_ARGS__); \
    }                                                                 \
  } while (0)
