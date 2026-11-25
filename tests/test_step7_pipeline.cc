// M2/T8 and T11: chunk pipelining and bandwidth formulas. CPU_ONLY.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "gclab/common/error.h"
#include "gclab/sim/bandwidth.h"
#include "gclab/sim/collectives.h"
#include "gclab/sim/inproc_transport.h"
#include "gclab/sim/world.h"
#include "testlib.h"

using namespace std::chrono_literals;
using namespace gclab::sim;

namespace {

WorldConfig cfg(int nranks) {
  WorldConfig config;
  config.nranks = nranks;
  config.link = {5.0, 10.0};
  config.recv_timeout = 2000ms;
  return config;
}

float input(int rank, std::size_t index) {
  return static_cast<float>((rank + 1) * 100 + (index % 31));
}

float expected_sum(int nranks, std::size_t index) {
  float result = 0.0f;
  for (int rank = 0; rank < nranks; ++rank) result += input(rank, index);
  return result;
}

void check_chunked_ring(int nranks, std::size_t count, std::size_t chunk_bytes) {
  World world(cfg(nranks));
  world.run([&](Transport& tp) {
    std::vector<float> send(count), recv(count, -1.0f);
    for (std::size_t i = 0; i < count; ++i) send[i] = input(tp.rank(), i);
    const std::vector<float> send_copy = send;

    allreduce_ring(tp, send.data(), recv.data(), count, ReduceOp::Sum, chunk_bytes);

    CHECK(send == send_copy);
    for (std::size_t i = 0; i < count; ++i) {
      CHECK_EQ(recv[i], expected_sum(nranks, i));
    }
  });
}

}  // namespace

TEST(chunk_pipeline_ring_matches_unchunked_results) {
  for (int nranks : {1, 2, 3, 5, 8, 16}) {
    for (std::size_t count : {
             std::size_t{0}, std::size_t{1}, std::size_t{3},
             std::size_t{7}, std::size_t{17}, std::size_t{65}}) {
      for (std::size_t chunk_bytes : {
               sizeof(float), 2 * sizeof(float), 4 * sizeof(float),
               16 * sizeof(float), 1024 * sizeof(float)}) {
        check_chunked_ring(nranks, count, chunk_bytes);
      }
    }
  }
}

TEST(chunk_pipeline_handles_blocks_smaller_than_world) {
  check_chunked_ring(8, 3, sizeof(float));
}

TEST(chunk_pipeline_preserves_unpipelined_ring_cost_for_one_port_model) {
  constexpr int kRanks = 4;
  constexpr std::size_t kCount = 250'000;  // 1 MB total.

  auto run = [](std::size_t chunk_bytes) {
    World world(cfg(kRanks));
    return world.run([&](Transport& tp) {
      std::vector<float> send(kCount, 1.0f), recv(kCount);
      allreduce_ring(tp, send.data(), recv.data(), kCount, ReduceOp::Sum, chunk_bytes);
    }).makespan_us;
  };

  const double unchunked_us = run(0);
  const double chunked_us = run(4096);
  // Chunks are posted back-to-back on one TX/RX port, so they stream through
  // each ring step without adding a latency bubble between adjacent chunks.
  CHECK_NEAR(chunked_us, unchunked_us, 1e-9);
}

TEST(chunk_pipeline_double_tree_matches_unchunked_results) {
  constexpr int kRanks = 7;
  constexpr std::size_t kCount = 37;
  World world(cfg(kRanks));
  world.run([](Transport& tp) {
    std::vector<float> send(kCount), recv(kCount, -1.0f);
    for (std::size_t i = 0; i < kCount; ++i) send[i] = input(tp.rank(), i);

    allreduce_double_binary_tree(tp, send.data(), recv.data(), kCount,
                                 ReduceOp::Sum, 3 * sizeof(float));

    for (std::size_t i = 0; i < kCount; ++i) {
      CHECK_EQ(recv[i], expected_sum(kRanks, i));
    }
  });
}

TEST(chunk_pipeline_rejects_non_float_aligned_sizes) {
  World world(cfg(2));
  InProcTransport rank0(world, 0);
  std::vector<float> send(2, 1.0f), recv(2);
  for (std::size_t bad_size : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                               std::size_t{5}, std::size_t{6}}) {
    CHECK_THROWS(
        allreduce_ring(rank0, send.data(), recv.data(), send.size(),
                       ReduceOp::Sum, bad_size),
        gclab::CommError);
    CHECK_THROWS(
        allreduce_double_binary_tree(rank0, send.data(), recv.data(), send.size(),
                                     ReduceOp::Sum, bad_size),
        gclab::CommError);
  }
}

TEST(bandwidth_algorithm_uses_decimal_gigabytes_per_second) {
  CHECK_NEAR(algorithm_bandwidth_GBps(1'000'000, 100.0), 10.0, 1e-12);
  CHECK_NEAR(algorithm_bandwidth_GBps(1000, 1.0), 1.0, 1e-12);
  CHECK_NEAR(algorithm_bandwidth_GBps(0, 100.0), 0.0, 1e-12);
}

TEST(bandwidth_algorithm_rejects_invalid_elapsed_time) {
  CHECK_THROWS(algorithm_bandwidth_GBps(1000, 0.0), gclab::CommError);
  CHECK_THROWS(algorithm_bandwidth_GBps(1000, -1.0), gclab::CommError);
  CHECK_THROWS(algorithm_bandwidth_GBps(
                   1000, std::numeric_limits<double>::infinity()),
               gclab::CommError);
  CHECK_THROWS(algorithm_bandwidth_GBps(
                   1000, std::numeric_limits<double>::quiet_NaN()),
               gclab::CommError);
}

TEST(bandwidth_bus_factors_match_nccl_tests_formulas) {
  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::AllReduce, 4), 1.5, 1e-12);
  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::AllGather, 4), 0.75, 1e-12);
  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::ReduceScatter, 4), 0.75, 1e-12);
  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::Broadcast, 4), 1.0, 1e-12);
  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::Reduce, 4), 1.0, 1e-12);

  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::AllReduce, 1), 0.0, 1e-12);
  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::AllGather, 1), 0.0, 1e-12);
  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::ReduceScatter, 1), 0.0, 1e-12);
  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::Broadcast, 1), 1.0, 1e-12);
  CHECK_NEAR(bus_bandwidth_factor(CollectiveKind::Reduce, 1), 1.0, 1e-12);
}

TEST(bandwidth_bus_factor_rejects_invalid_inputs) {
  CHECK_THROWS(bus_bandwidth_factor(CollectiveKind::AllReduce, 0),
               gclab::CommError);
  CHECK_THROWS(bus_bandwidth_factor(CollectiveKind::AllReduce, -1),
               gclab::CommError);
  CHECK_THROWS(bus_bandwidth_factor(static_cast<CollectiveKind>(999), 4),
               gclab::CommError);
}

TEST(bandwidth_bus_value_composes_algorithm_bandwidth_and_factor) {
  CHECK_NEAR(bus_bandwidth_GBps(CollectiveKind::AllReduce, 4, 1'000'000, 100.0),
             15.0, 1e-12);
  CHECK_NEAR(bus_bandwidth_GBps(CollectiveKind::AllGather, 4, 1'000'000, 100.0),
             7.5, 1e-12);
  CHECK_NEAR(
      bus_bandwidth_GBps(CollectiveKind::ReduceScatter, 4, 1'000'000, 100.0),
      7.5, 1e-12);
}
