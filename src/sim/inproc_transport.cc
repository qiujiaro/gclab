#include "gclab/sim/inproc_transport.h"

#include "gclab/common/error.h"
#include "gclab/common/log.h"

namespace gclab::sim {

InProcTransport::InProcTransport(World& world, int rank) : world_(world), rank_(rank) {}

void InProcTransport::send(int dst, const void* buf, std::size_t bytes, int tag) {
  // TODO(step 3):
  //   1. Validate dst (0 <= dst < size()). Is sending to yourself legal?
  //      Decide, document it in a comment, and make the tests agree.
  //   2. Apply the send-side timing rule from inproc_transport.h.
  //   3. Build a Message: copy `bytes` bytes from `buf` into payload (eager
  //      semantics: caller may overwrite buf as soon as we return), set tag
  //      and send_start_us.
  //   4. world_.channel(rank_, dst).push(std::move(msg)).
  //   5. Optional: GCLAB_LOG(Debug, "p2p.send", {...}) with rank, dst, tag,
  //      bytes, sim time.
  (void)dst;
  (void)buf;
  (void)bytes;
  (void)tag;
  throw NotImplemented("InProcTransport::send");
}

void InProcTransport::recv(int src, void* buf, std::size_t bytes, int tag) {
  // TODO(step 3):
  //   1. Validate src.
  //   2. Message m = world_.channel(src, rank_).pop(tag, world_.config().recv_timeout);
  //   3. If m.payload.size() != bytes, throw CommError with BOTH sizes, the
  //      peer and the tag in the message. (Think: what would real NCCL do
  //      here?)
  //   4. Copy the payload into buf.
  //   5. Apply the recv-side timing rule from inproc_transport.h.
  (void)src;
  (void)buf;
  (void)bytes;
  (void)tag;
  throw NotImplemented("InProcTransport::recv");
}

}  // namespace gclab::sim
