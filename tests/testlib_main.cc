#include <cstdio>
#include <cstring>
#include <exception>

#include "gclab/common/error.h"
#include "testlib.h"

namespace testlib {
std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}
}  // namespace testlib

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int passed = 0, failed = 0, todo = 0;
  for (const auto& c : testlib::registry()) {
    if (filter && std::strstr(c.name, filter) == nullptr) continue;
    try {
      c.fn();
      std::printf("[ PASS ] %s\n", c.name);
      ++passed;
    } catch (const gclab::NotImplemented& e) {
      std::printf("[ TODO ] %s -- %s\n", c.name, e.what());
      ++todo;
    } catch (const std::exception& e) {
      std::printf("[ FAIL ] %s -- %s\n", c.name, e.what());
      ++failed;
    }
  }
  std::printf("\n%d passed, %d failed, %d todo\n", passed, failed, todo);
  return (failed == 0 && todo == 0) ? 0 : 1;
}
