// M2/T6: non-blocking Transport requests. CPU_ONLY.
#include "gclab/common/error.h"
#include "testlib.h"

TEST(async_send_recv_todo) {
  // TODO(M2/T6): cover send-first, recv-first, multiple outstanding requests,
  // zero-byte messages, buffer lifetime, tag mismatch, timeout, world abort,
  // foreign requests, and repeated wait.
  throw gclab::NotImplemented("M2 async Transport tests");
}
