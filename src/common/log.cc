#include "gclab/common/log.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace gclab::log {
namespace {

Level parse_level() {
  const char* env = std::getenv("GCLAB_LOG_LEVEL");
  if (env == nullptr) return Level::Warn;
  if (std::strcmp(env, "DEBUG") == 0) return Level::Debug;
  if (std::strcmp(env, "INFO") == 0) return Level::Info;
  if (std::strcmp(env, "ERROR") == 0) return Level::Error;
  return Level::Warn;
}

const char* name(Level l) {
  switch (l) {
    case Level::Debug: return "DEBUG";
    case Level::Info: return "INFO";
    case Level::Warn: return "WARN";
    case Level::Error: return "ERROR";
  }
  return "?";
}

}  // namespace

bool enabled(Level level) {
  static const Level threshold = parse_level();
  return level >= threshold;
}

void emit(Level level, const char* event, std::initializer_list<Field> fields) {
  using namespace std::chrono;
  const auto ts = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  std::string line = "ts_us=" + std::to_string(ts) + " level=" + name(level) + " event=" + event;
  for (const auto& [k, v] : fields) {
    line += ' ';
    line += k;
    line += '=';
    line += v;
  }
  line += '\n';
  // One write per line under a mutex so lines from different rank threads
  // never interleave.
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  std::fputs(line.c_str(), stderr);
}

}  // namespace gclab::log
