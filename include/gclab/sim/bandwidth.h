#pragma once
// nccl-tests-compatible algorithm and bus bandwidth helpers.

#include <cstddef>

namespace gclab::sim {

enum class CollectiveKind {
  AllReduce,
  AllGather,
  ReduceScatter,
  Broadcast,
  Reduce,
};

// Decimal GB/s. elapsed_us must be finite and positive.
double algorithm_bandwidth_GBps(std::size_t message_bytes, double elapsed_us);

// Multiplicative nccl-tests correction from algbw to busbw.
double bus_bandwidth_factor(CollectiveKind collective, int nranks);

double bus_bandwidth_GBps(CollectiveKind collective, int nranks,
                          std::size_t message_bytes, double elapsed_us);

}  // namespace gclab::sim
