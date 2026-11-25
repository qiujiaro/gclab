#include "gclab/sim/inproc_transport.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

#include "gclab/common/error.h"
#include "gclab/common/log.h"
#include "gclab/sim/cost_model.h"
#include "gclab/sim/topology.h"

namespace gclab::sim {

namespace {

struct ScheduledSend {
  double start_us = 0.0;
  double finish_us = 0.0;
  double alpha_us = 0.0;
  double wire_us = 0.0;
};

ScheduledSend schedule_send(World& world, int src, int dst, double earliest_start_us,
                            std::size_t bytes) {
  const WorldConfig& config = world.config();
  LinkParams link = config.link;
  const std::vector<int>* shared_resources = nullptr;
  if (config.topology) {
    const LinkPath& path = config.topology->path(src, dst);
    link = path.link;
    shared_resources = &path.shared_resources;
  }

  const double wire_us = wire_time_us(link, bytes);
  double start_us = earliest_start_us;
  if (shared_resources != nullptr) {
    start_us = world.reserve_shared_resources(*shared_resources, start_us, wire_us);
  }
  return {start_us, start_us + wire_us, link.alpha_us, wire_us};
}

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
  const ScheduledSend timing =
      schedule_send(world_, rank_, dst, std::max(clock_us_, tx_free_us_), bytes);
  tx_free_us_ = timing.finish_us;
  clock_us_ = timing.finish_us;

  Message msg;
  msg.tag = tag;
  msg.send_start_us = timing.start_us;
  msg.alpha_us = timing.alpha_us;
  msg.wire_time_us = timing.wire_us;
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

  const double first_byte_us = msg.send_start_us + msg.alpha_us;
  const double begin_us = std::max(first_byte_us, rx_free_us_);
  const double finish_us = begin_us + msg.wire_time_us;
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

  const ScheduledSend timing =
      schedule_send(world_, rank_, dst, std::max(clock_us_, tx_free_us_), bytes);
  tx_free_us_ = timing.finish_us;

  Message msg;
  msg.tag = tag;
  msg.send_start_us = timing.start_us;
  msg.alpha_us = timing.alpha_us;
  msg.wire_time_us = timing.wire_us;
  msg.payload.resize(bytes);
  if (bytes != 0) std::memcpy(msg.payload.data(), buf, bytes);
  world_.channel(rank_, dst).push(std::move(msg));

  return std::make_unique<SendRequest>(this, timing.finish_us);
}

RequestPtr InProcTransport::irecv(int src, void* buf, std::size_t bytes, int tag) {
  if (src < 0 || src >= size()) {
    throw CommError(
        "bad recv peer " + std::to_string(src) +
        " at rank " + std::to_string(rank_));
  }

  if (bytes != 0 && buf == nullptr) {
    throw CommError("null recv buffer with non-zero size");
  }

  return std::make_unique<RecvRequest>(
      this, src, buf, bytes, tag);
}

void InProcTransport::wait(Request& request) {
  // SendRequest
  if (auto* send_request =
          dynamic_cast<SendRequest*>(&request)) {
    if (send_request->owner != this) {
      throw CommError(
          "send request belongs to another transport");
    }

    if (send_request->waited) {
      throw CommError(
          "send request has already been waited");
    }

    clock_us_ =
        std::max(clock_us_, send_request->finish_us);

    send_request->waited = true;
    return;
  }

  // RecvRequest
  if (auto* recv_request =
          dynamic_cast<RecvRequest*>(&request)) {
    if (recv_request->owner != this) {
      throw CommError(
          "receive request belongs to another transport");
    }

    if (recv_request->waited) {
      throw CommError(
          "receive request has already been waited");
    }

    Message msg =
        world_.channel(recv_request->src, rank_)
            .pop(
                recv_request->tag,
                world_.config().recv_timeout);

    if (msg.payload.size() != recv_request->bytes) {
      throw CommError(
          "recv size mismatch from peer " +
          std::to_string(recv_request->src) +
          " for tag " +
          std::to_string(recv_request->tag) +
          ": expected " +
          std::to_string(recv_request->bytes) +
          " bytes, got " +
          std::to_string(msg.payload.size()));
    }

    if (recv_request->bytes != 0) {
      std::memcpy(
          recv_request->buffer,
          msg.payload.data(),
          recv_request->bytes);
    }

    const double first_byte_us = msg.send_start_us + msg.alpha_us;

    const double begin_us =
        std::max(first_byte_us, rx_free_us_);

    const double finish_us = begin_us + msg.wire_time_us;

    rx_free_us_ = finish_us;
    clock_us_ = std::max(clock_us_, finish_us);

    recv_request->waited = true;
    return;
  }

  throw CommError("unknown request type");
}

void InProcTransport::wait_all(
    std::vector<RequestPtr>& requests) {
  for (auto& request : requests) {
    if (!request) {
      throw CommError("null request in wait_all");
    }

    wait(*request);
  }
}

}  // namespace gclab::sim
