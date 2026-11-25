#pragma once
// InProcTransport: Transport over in-process Channels with a simulated clock.
//
// Timing model: "single-port, full-duplex" alpha-beta.
//   - Each rank has ONE outgoing port and ONE incoming port. They work at the
//     same time (full duplex), but each handles one message at a time.
//   - Let w = wire_time_us(link, n) = n / beta.
//
//   send(n bytes) at local clock c:
//       start         = max(c, tx_free_)
//       tx_free_      = start + w
//       clock_        = start + w        (eager send blocks until injected)
//       msg.send_start_us = start
//
//   recv(msg of n bytes):
//       first_byte    = msg.send_start_us + alpha
//       begin         = max(first_byte, rx_free_)   (incoming port busy?)
//       finish        = begin + w
//       rx_free_      = finish
//       clock_        = max(clock_, finish)
//
// Consequences you will verify in tests:
//   p2p, idle network:          receiver done at alpha + w
//   root sends to p-1 peers:    p-1 sends serialize on root's tx port
//   p-1 peers send to root:     p-1 recvs serialize on root's rx port
// Without the rx port, a linear Reduce would look as fast as one message --
// a classic way a naive simulator lies.
//
// Module 2 optionally replaces the homogeneous link with topology routes and
// serializes flows that name the same shared resource. Still not modelled:
// reduction compute cost (the gamma term), protocol overheads (LL vs Simple).

#include "gclab/sim/transport.h"
#include "gclab/sim/world.h"

namespace gclab::sim {

class InProcTransport final : public Transport {
 public:
  InProcTransport(World& world, int rank);

  int rank() const override { return rank_; }
  int size() const override { return world_.config().nranks; }

  void send(int dst, const void* buf, std::size_t bytes, int tag) override;
  void recv(int src, void* buf, std::size_t bytes, int tag) override;
  RequestPtr isend(int dst, const void* buf, std::size_t bytes, int tag) override;
  RequestPtr irecv(int src, void* buf, std::size_t bytes, int tag) override;
  void wait(Request& request) override;
  void wait_all(std::vector<RequestPtr>& requests) override;

  double now_us() const override { return clock_us_; }

 private:
  World& world_;
  int rank_;
  double clock_us_ = 0.0;
  double tx_free_us_ = 0.0;
  double rx_free_us_ = 0.0;
};

}  // namespace gclab::sim
