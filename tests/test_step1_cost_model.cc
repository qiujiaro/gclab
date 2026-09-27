// Step 1: alpha-beta cost model. CPU_ONLY.
#include "gclab/sim/cost_model.h"
#include "testlib.h"

using gclab::sim::LinkParams;
using gclab::sim::transfer_time_us;
using gclab::sim::wire_time_us;

TEST(wire_time_units) {
  LinkParams link{5.0, 10.0};  // 5 us, 10 GB/s
  // 1e6 bytes at 10 GB/s = 1e-4 s = 100 us
  CHECK_NEAR(wire_time_us(link, 1'000'000), 100.0, 1e-9);
  // 1 GB/s means 1000 bytes per microsecond
  LinkParams one{0.0, 1.0};
  CHECK_NEAR(wire_time_us(one, 1000), 1.0, 1e-12);
}

TEST(zero_bytes_costs_only_latency) {
  LinkParams link{5.0, 10.0};
  CHECK_NEAR(wire_time_us(link, 0), 0.0, 1e-12);
  CHECK_NEAR(transfer_time_us(link, 0), 5.0, 1e-12);
}

TEST(transfer_is_alpha_plus_wire) {
  LinkParams link{5.0, 10.0};
  CHECK_NEAR(transfer_time_us(link, 1'000'000), 105.0, 1e-9);
}

TEST(crossover_point) {
  // n_half = alpha * beta: the message size where latency and bandwidth terms
  // are equal. Below it you are latency-bound, above it bandwidth-bound.
  LinkParams link{5.0, 25.0};
  const double n_half_bytes = link.alpha_us * link.beta_GBps * 1e3;  // 125 KB
  CHECK_NEAR(wire_time_us(link, static_cast<std::size_t>(n_half_bytes)), link.alpha_us, 1e-9);
}
