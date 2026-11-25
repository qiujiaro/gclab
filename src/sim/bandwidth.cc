#include "gclab/sim/bandwidth.h"

#include <cmath>

#include "gclab/common/error.h"

namespace gclab::sim {

double algorithm_bandwidth_GBps(std::size_t message_bytes, double elapsed_us) {
  if (!std::isfinite(elapsed_us) || elapsed_us <= 0.0) {
    throw CommError("elapsed_us must be finite and positive");
  }
  // 1 GB/s = 1000 bytes/us when GB uses decimal units, as nccl-tests does.
  return static_cast<double>(message_bytes) / elapsed_us / 1000.0;
}

double bus_bandwidth_factor(CollectiveKind collective, int nranks) {
  if (nranks <= 0) throw CommError("nranks must be positive");
  const double p = static_cast<double>(nranks);
  switch (collective) {
    case CollectiveKind::AllReduce:
      return 2.0 * (p - 1.0) / p;
    case CollectiveKind::AllGather:
    case CollectiveKind::ReduceScatter:
      return (p - 1.0) / p;
    case CollectiveKind::Broadcast:
    case CollectiveKind::Reduce:
      return 1.0;
  }
  throw CommError("unknown collective kind");
}

double bus_bandwidth_GBps(CollectiveKind collective, int nranks,
                          std::size_t message_bytes, double elapsed_us) {
  return algorithm_bandwidth_GBps(message_bytes, elapsed_us) *
         bus_bandwidth_factor(collective, nranks);
}

}  // namespace gclab::sim
