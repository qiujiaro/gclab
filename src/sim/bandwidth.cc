#include "gclab/sim/bandwidth.h"

#include "gclab/common/error.h"

namespace gclab::sim {

double algorithm_bandwidth_GBps(std::size_t message_bytes, double elapsed_us) {
  // TODO(M2/T11): validate elapsed_us and convert bytes/us to decimal GB/s.
  (void)message_bytes;
  (void)elapsed_us;
  throw NotImplemented("algorithm_bandwidth_GBps");
}

double bus_bandwidth_factor(CollectiveKind collective, int nranks) {
  // TODO(M2/T11): match nccl-tests factors for every CollectiveKind, including
  // the nranks == 1 edge case.
  (void)collective;
  (void)nranks;
  throw NotImplemented("bus_bandwidth_factor");
}

double bus_bandwidth_GBps(CollectiveKind collective, int nranks,
                          std::size_t message_bytes, double elapsed_us) {
  // TODO(M2/T11): compose algorithm_bandwidth_GBps and
  // bus_bandwidth_factor without duplicating validation.
  (void)collective;
  (void)nranks;
  (void)message_bytes;
  (void)elapsed_us;
  throw NotImplemented("bus_bandwidth_GBps");
}

}  // namespace gclab::sim
