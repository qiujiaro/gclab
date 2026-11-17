// M2/T7: Ring ReduceScatter, AllGather and AllReduce. CPU_ONLY.
#include "gclab/common/error.h"
#include "testlib.h"

TEST(ring_collectives_todo) {
  // TODO(M2/T7): test p=1..16, Sum/Max, zero and odd counts, count<p,
  // unmodified inputs, mismatched collective order, and the expected
  // 2*(p-1)/p per-rank AllReduce traffic.
  throw gclab::NotImplemented("M2 ring collective tests");
}
