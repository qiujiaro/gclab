#pragma once
// Alpha-beta (Hockney) cost model for one point-to-point link.
//
//   T(n) = alpha + n / beta
//
//   alpha : per-message latency (software + wire startup), microseconds
//   beta  : sustained bandwidth, GB/s  (1 GB/s = 1e9 bytes/s, *decimal* GB,
//           the convention nccl-tests uses for algbw/busbw)
//
// Everything this file computes is SIMULATED TIME. It says nothing about how
// fast your Mac runs the simulator.

#include <cstddef>

namespace gclab::sim {

struct LinkParams {
  double alpha_us = 5.0;     // latency per message
  double beta_GBps = 25.0;   // bandwidth
};

// Serialization time: how long the payload occupies the link (n / beta), in us.
// This is the part of the cost that blocks the sender's port and the
// receiver's port.
double wire_time_us(const LinkParams& link, std::size_t bytes);

// Full point-to-point time for an idle link: alpha + n / beta, in us.
double transfer_time_us(const LinkParams& link, std::size_t bytes);

}  // namespace gclab::sim
