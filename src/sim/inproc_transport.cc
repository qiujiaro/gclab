#include "gclab/sim/inproc_transport.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

#include "gclab/common/error.h"
#include "gclab/common/log.h"
#include "gclab/sim/cost_model.h"

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
  if (dst < 0 || dst >= size()) {
    throw CommError("bad send peer " + std::to_string(dst) + " from rank " +
                    std::to_string(rank_));
  }
  if (bytes != 0 && buf == nullptr) throw CommError("null send buffer with non-zero size");
  // Self-send is legal: eager buffering makes a subsequent self-recv safe.
  const double start_us = std::max(clock_us_, tx_free_us_);
  const double finish_us = start_us + wire_time_us(world_.config().link, bytes);
  tx_free_us_ = finish_us;
  clock_us_ = finish_us;

  Message msg;
  msg.tag = tag;
  msg.send_start_us = start_us;
  msg.payload.resize(bytes);
  if (bytes != 0) std::memcpy(msg.payload.data(), buf, bytes);
  world_.channel(rank_, dst).push(std::move(msg));
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
  if (src < 0 || src >= size()) {
    throw CommError("bad recv peer " + std::to_string(src) + " at rank " +
                    std::to_string(rank_));
  }
  if (bytes != 0 && buf == nullptr) throw CommError("null recv buffer with non-zero size");

  Message msg = world_.channel(src, rank_).pop(tag, world_.config().recv_timeout);
  if (msg.payload.size() != bytes) {
    throw CommError("recv size mismatch from peer " + std::to_string(src) + " for tag " +
                    std::to_string(tag) + ": expected " + std::to_string(bytes) +
                    " bytes, got " + std::to_string(msg.payload.size()));
  }
  if (bytes != 0) std::memcpy(buf, msg.payload.data(), bytes);

  const double first_byte_us = msg.send_start_us + world_.config().link.alpha_us;
  const double begin_us = std::max(first_byte_us, rx_free_us_);
  const double finish_us = begin_us + wire_time_us(world_.config().link, bytes);
  rx_free_us_ = finish_us;
  clock_us_ = std::max(clock_us_, finish_us);
}

}  // namespace gclab::sim
