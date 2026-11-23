#include "gclab/sim/inproc_transport.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

#include "gclab/common/error.h"
#include "gclab/common/log.h"
#include "gclab/sim/cost_model.h"

namespace gclab::sim {

namespace {

class SendRequest final : public Request {
 public:
  SendRequest(InProcTransport* owner_value, double finish_us_value)
      : owner(owner_value), finish_us(finish_us_value) {}

  InProcTransport* owner;
  double finish_us;
  bool waited = false;
};

class RecvRequest final : public Request {
 public:
  RecvRequest(InProcTransport* owner_value, int src_value, void* buffer_value,
              std::size_t bytes_value, int tag_value)
      : owner(owner_value),
        src(src_value),
        buffer(buffer_value),
        bytes(bytes_value),
        tag(tag_value) {}

  InProcTransport* owner;
  int src;
  void* buffer;
  std::size_t bytes;
  int tag;
  bool waited = false;
};

}  // namespace

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

RequestPtr InProcTransport::isend(int dst, const void* buf, std::size_t bytes, int tag) {
  // Reserve the outgoing port and publish an eager copy immediately. Waiting
  // on the returned request will advance the caller's clock to finish_us.
  if (dst < 0 || dst >= size()) {
    throw CommError("bad send peer " + std::to_string(dst) + " from rank " +
                    std::to_string(rank_));
  }
  if (bytes != 0 && buf == nullptr) {
    throw CommError("null send buffer with non-zero size");
  }

  const double start_us = std::max(clock_us_, tx_free_us_);
  const double finish_us = start_us + wire_time_us(world_.config().link, bytes);
  tx_free_us_ = finish_us;

  Message msg;
  msg.tag = tag;
  msg.send_start_us = start_us;
  msg.payload.resize(bytes);
  if (bytes != 0) std::memcpy(msg.payload.data(), buf, bytes);
  world_.channel(rank_, dst).push(std::move(msg));

  return std::make_unique<SendRequest>(this, finish_us);
}

RequestPtr InProcTransport::irecv(int src, void* buf, std::size_t bytes, int tag) {
  // TODO(M2/T6): create an in-process receive request that completes only when
  // a matching message is available, or fails on timeout/world abort.
  (void)src;
  (void)buf;
  (void)bytes;
  (void)tag;
  throw NotImplemented("InProcTransport::irecv");
}

void InProcTransport::wait(Request& request) {
  // TODO(M2/T6): reject requests created by another transport, reject a
  // second wait, complete the operation, and advance the simulated clock.
  (void)request;
  throw NotImplemented("InProcTransport::wait");
}

void InProcTransport::wait_all(std::vector<RequestPtr>& requests) {
  // TODO(M2/T6): wait in input order and define how already-completed requests
  // are handled when a later request fails.
  (void)requests;
  throw NotImplemented("InProcTransport::wait_all");
}

}  // namespace gclab::sim
