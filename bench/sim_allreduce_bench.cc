// M2 simulated AllReduce benchmark skeleton.
#include <cstdio>

#include "gclab/common/error.h"

int main() {
  // TODO(M2): sweep algorithm, topology, rank count, message size, and chunk
  // size. Emit the schema documented in analysis/schema.py and always set
  // mode=SIMULATED.
  try {
    throw gclab::NotImplemented("sim_allreduce_bench");
  } catch (const gclab::NotImplemented& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
}
