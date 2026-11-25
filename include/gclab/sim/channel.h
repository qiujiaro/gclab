#pragma once
// A one-directional, unbounded, tag-matched message queue between two ranks.
//
// One Channel exists per ordered pair (src -> dst). Exactly one thread pushes
// (rank src) and exactly one thread pops (rank dst), but write it as if any
// number could: it is guarded by a mutex + condition variable.
//
// Matching rule (same as MPI): pop(tag) returns the OLDEST message with that
// tag. Messages with the same tag never overtake each other; messages with
// different tags may.
//
// Production contrast: NCCL has no tags and no matching at all. Each
// connection is a fixed ring buffer whose slots are consumed strictly in order,
// and correctness depends on every rank issuing collectives in the same order.
// We keep tags here so that mistakes show up as clear errors instead of
// silent corruption.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>

namespace gclab::sim {

struct Message {
  int tag = 0;
  std::vector<std::byte> payload;
  // Simulated time at which the SENDER started injecting this message.
  // The receiver uses it to compute when the bytes finish arriving.
  double send_start_us = 0.0;
  // Route timing captured by the sender. Keeping it with the message lets the
  // receiver honor heterogeneous topology paths without recomputing a route.
  double alpha_us = 0.0;
  double wire_time_us = 0.0;
};

class Channel {
 public:
  Channel() = default;
  Channel(const Channel&) = delete;
  Channel& operator=(const Channel&) = delete;

  // Enqueue a message and wake any waiting receiver. Never blocks for long
  // (eager / buffered semantics: the queue is unbounded).
  void push(Message msg);

  // Block until a message with `tag` is available, then remove and return it.
  // Throws TimeoutError if `timeout` elapses first.
  // Throws AbortedError if abort() is called while waiting (or was already
  // called before pop was entered).
  Message pop(int tag, std::chrono::milliseconds timeout);

  // Wake all waiters and make every current and future pop() throw
  // AbortedError. Idempotent. (Provided.)
  void abort();

  // Number of queued messages (for tests / debugging). (Provided.)
  std::size_t pending() const;

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Message> queue_;
  bool aborted_ = false;
};

}  // namespace gclab::sim
