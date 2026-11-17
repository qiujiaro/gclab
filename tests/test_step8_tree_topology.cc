// M2/T9 and T10: tree AllReduce and topology contention. CPU_ONLY.
#include "gclab/common/error.h"
#include "testlib.h"

TEST(tree_and_topology_todo) {
  // TODO(M2/T9,T10): test both tree algorithms for non-power-of-two worlds,
  // validate routes, and show that flows sharing a PCIe uplink contend while
  // independent NVLink-like routes do not.
  throw gclab::NotImplemented("M2 tree and topology tests");
}
