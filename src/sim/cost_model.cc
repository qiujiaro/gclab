#include "gclab/sim/cost_model.h"

#include "gclab/common/error.h"

namespace gclab::sim {

double wire_time_us(const LinkParams& link, std::size_t bytes) {
  // TODO(step 1): return n / beta in MICROSECONDS.
  //   Watch the units: beta is in GB/s (1e9 bytes per second).
  //   How many bytes per microsecond is 1 GB/s

  return static_cast<double>(bytes) /1000.0/link.beta_GBps;
}

double transfer_time_us(const LinkParams& link, std::size_t bytes) {
  // TODO(step 1): alpha + n / beta. Reuse wire_time_us.
  return link.alpha_us+wire_time_us(link,bytes);
}

}  // namespace gclab::sim
