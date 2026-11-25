// M2/T9 and T10: tree AllReduce and topology contention. CPU_ONLY.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "gclab/common/error.h"
#include "gclab/sim/collectives.h"
#include "gclab/sim/inproc_transport.h"
#include "gclab/sim/topology.h"
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

float expected(int nranks, std::size_t index, ReduceOp op) {
  float result = input(0, index);
  for (int rank = 1; rank < nranks; ++rank) {
    const float value = input(rank, index);
    result = op == ReduceOp::Sum ? result + value : std::max(result, value);
  }
  return result;
}

void check_tree(int nranks, std::size_t count, ReduceOp op, bool double_tree) {
  World world(cfg(nranks));
  world.run([&](Transport& tp) {
    std::vector<float> send(count), recv(count, -1.0f);
    for (std::size_t i = 0; i < count; ++i) send[i] = input(tp.rank(), i);
    const std::vector<float> send_copy = send;

    if (double_tree) {
      allreduce_double_binary_tree(tp, send.data(), recv.data(), count, op, 0);
    } else {
      allreduce_tree(tp, send.data(), recv.data(), count, op);
    }

    CHECK(send == send_copy);
    for (std::size_t i = 0; i < count; ++i) {
      CHECK_EQ(recv[i], expected(nranks, i, op));
    }
  });
}

}  // namespace

TEST(allreduce_tree_correct_for_power_of_two_and_other_worlds) {
  for (int nranks = 1; nranks <= 16; ++nranks) {
    for (std::size_t count : {
             std::size_t{0}, std::size_t{1}, std::size_t{3},
             std::size_t{7}, std::size_t{17}, std::size_t{32}}) {
      for (ReduceOp op : {ReduceOp::Sum, ReduceOp::Max}) {
        check_tree(nranks, count, op, false);
      }
    }
  }
}

TEST(allreduce_tree_supports_in_place_buffers) {
  constexpr int kRanks = 5;
  constexpr std::size_t kCount = 13;
  World world(cfg(kRanks));
  world.run([](Transport& tp) {
    std::vector<float> buffer(kCount);
    for (std::size_t i = 0; i < kCount; ++i) buffer[i] = input(tp.rank(), i);

    allreduce_tree(tp, buffer.data(), buffer.data(), kCount, ReduceOp::Sum);

    for (std::size_t i = 0; i < kCount; ++i) {
      CHECK_EQ(buffer[i], expected(kRanks, i, ReduceOp::Sum));
    }
  });
}

TEST(allreduce_tree_validates_buffers) {
  World world(cfg(2));
  InProcTransport rank0(world, 0);
  float send = 1.0f, recv = 0.0f;
  CHECK_THROWS(allreduce_tree(rank0, nullptr, &recv, 1, ReduceOp::Sum),
               gclab::CommError);
  CHECK_THROWS(allreduce_tree(rank0, &send, nullptr, 1, ReduceOp::Sum),
               gclab::CommError);
  allreduce_tree(rank0, nullptr, nullptr, 0, ReduceOp::Sum);
}

TEST(allreduce_tree_cost_is_logarithmic_for_power_of_two_world) {
  constexpr int kRanks = 8;
  constexpr std::size_t kCount = 250'000;  // 1 MB at 10 GB/s.
  World world(cfg(kRanks));
  const RunResult result = world.run([](Transport& tp) {
    std::vector<float> send(kCount, 1.0f), recv(kCount);
    allreduce_tree(tp, send.data(), recv.data(), kCount, ReduceOp::Sum);
  });
  CHECK_NEAR(result.makespan_us, 2.0 * 3.0 * (5.0 + 100.0), 1e-9);
}

TEST(double_binary_tree_correct_for_power_of_two_and_other_worlds) {
  for (int nranks = 1; nranks <= 16; ++nranks) {
    for (std::size_t count : {
             std::size_t{0}, std::size_t{1}, std::size_t{3},
             std::size_t{7}, std::size_t{17}, std::size_t{32}}) {
      for (ReduceOp op : {ReduceOp::Sum, ReduceOp::Max}) {
        check_tree(nranks, count, op, true);
      }
    }
  }
}

TEST(double_binary_tree_supports_in_place_buffers) {
  constexpr int kRanks = 6;
  constexpr std::size_t kCount = 13;
  World world(cfg(kRanks));
  world.run([](Transport& tp) {
    std::vector<float> buffer(kCount);
    for (std::size_t i = 0; i < kCount; ++i) buffer[i] = input(tp.rank(), i);

    allreduce_double_binary_tree(tp, buffer.data(), buffer.data(), kCount,
                                 ReduceOp::Max, 0);

    for (std::size_t i = 0; i < kCount; ++i) {
      CHECK_EQ(buffer[i], expected(kRanks, i, ReduceOp::Max));
    }
  });
}

TEST(double_binary_tree_validates_buffers_and_supports_chunking) {
  World world(cfg(2));
  InProcTransport rank0(world, 0);
  float send = 1.0f, recv = 0.0f;
  CHECK_THROWS(
      allreduce_double_binary_tree(rank0, nullptr, &recv, 1, ReduceOp::Sum, 0),
      gclab::CommError);
  CHECK_THROWS(
      allreduce_double_binary_tree(rank0, &send, nullptr, 1, ReduceOp::Sum, 0),
      gclab::CommError);
  allreduce_double_binary_tree(rank0, nullptr, nullptr, 0, ReduceOp::Sum, 0);

  World single_world(cfg(1));
  InProcTransport only_rank(single_world, 0);
  allreduce_double_binary_tree(only_rank, &send, &recv, 1, ReduceOp::Sum,
                               sizeof(float));
  CHECK_EQ(recv, send);
}

TEST(topology_validates_rank_paths_links_and_resources) {
  CHECK_THROWS(Topology(0), gclab::CommError);
  CHECK_THROWS(Topology(-1), gclab::CommError);

  Topology topology(2);
  topology.set_path(0, 1, {{2.0, 20.0}, {3, 1}});
  const LinkPath& configured = topology.path(0, 1);
  CHECK_NEAR(configured.link.alpha_us, 2.0, 1e-12);
  CHECK_NEAR(configured.link.beta_GBps, 20.0, 1e-12);
  CHECK_EQ(configured.shared_resources.size(), std::size_t{2});
  CHECK_EQ(configured.shared_resources[0], 1);
  CHECK_EQ(configured.shared_resources[1], 3);

  CHECK_THROWS(topology.path(-1, 0), gclab::CommError);
  CHECK_THROWS(topology.path(0, 2), gclab::CommError);
  CHECK_THROWS(topology.set_path(0, 1, {{-1.0, 10.0}, {}}), gclab::CommError);
  CHECK_THROWS(topology.set_path(0, 1, {{1.0, 0.0}, {}}), gclab::CommError);
  CHECK_THROWS(topology.set_path(
                   0, 1,
                   {{std::numeric_limits<double>::quiet_NaN(), 10.0}, {}}),
               gclab::CommError);
  CHECK_THROWS(topology.set_path(0, 1, {{1.0, 10.0}, {-1}}), gclab::CommError);
  CHECK_THROWS(topology.set_path(0, 1, {{1.0, 10.0}, {2, 2}}),
               gclab::CommError);
}

TEST(topology_factories_build_expected_routes) {
  const Topology full = Topology::fully_connected(3, {2.0, 20.0});
  CHECK_NEAR(full.path(0, 2).link.alpha_us, 2.0, 1e-12);
  CHECK_NEAR(full.path(0, 2).link.beta_GBps, 20.0, 1e-12);
  CHECK(full.path(0, 2).shared_resources.empty());
  CHECK_NEAR(full.path(1, 1).link.alpha_us, 0.0, 1e-12);
  CHECK(full.path(1, 1).shared_resources.empty());

  const Topology pcie = Topology::pcie_switch(3, {1.0, 32.0}, {2.0, 16.0});
  CHECK_NEAR(pcie.path(0, 2).link.alpha_us, 4.0, 1e-12);
  CHECK_NEAR(pcie.path(0, 2).link.beta_GBps, 16.0, 1e-12);
  CHECK_EQ(pcie.path(0, 2).shared_resources.size(), std::size_t{1});
  CHECK_EQ(pcie.path(0, 2).shared_resources[0], 0);
  CHECK_EQ(pcie.path(2, 0).shared_resources[0], 0);
  CHECK(pcie.path(0, 0).shared_resources.empty());

  const Topology nvlink = Topology::nvlink_mesh(3, {1.0, 100.0});
  CHECK_NEAR(nvlink.path(2, 0).link.alpha_us, 1.0, 1e-12);
  CHECK_NEAR(nvlink.path(2, 0).link.beta_GBps, 100.0, 1e-12);
  CHECK(nvlink.path(2, 0).shared_resources.empty());
}

TEST(topology_route_overrides_homogeneous_world_link) {
  WorldConfig config = cfg(2);
  config.link = {99.0, 1.0};  // Must be ignored when topology is present.
  config.topology =
      std::make_shared<Topology>(Topology::fully_connected(2, LinkParams{2.0, 20.0}));
  World world(config);
  const RunResult result = world.run([](Transport& tp) {
    std::vector<std::byte> buffer(1'000'000);
    if (tp.rank() == 0) {
      tp.send(1, buffer.data(), buffer.size(), 700);
    } else {
      tp.recv(0, buffer.data(), buffer.size(), 700);
    }
  });
  // 1 MB / 20 GB/s = 50 us, plus 2 us path latency at the receiver.
  CHECK_NEAR(result.makespan_us, 52.0, 1e-9);
}

TEST(topology_pcie_shared_uplink_contends_but_nvlink_routes_do_not) {
  auto run_two_flows = [](std::shared_ptr<const Topology> topology) {
    WorldConfig config = cfg(4);
    config.topology = std::move(topology);
    World world(config);
    return world.run([](Transport& tp) {
      std::vector<std::byte> buffer(1'000'000);
      if (tp.rank() % 2 == 0) {
        tp.send(tp.rank() + 1, buffer.data(), buffer.size(), 710);
      } else {
        tp.recv(tp.rank() - 1, buffer.data(), buffer.size(), 710);
      }
    }).makespan_us;
  };

  const double pcie_us = run_two_flows(std::make_shared<Topology>(
      Topology::pcie_switch(4, LinkParams{0.0, 20.0}, LinkParams{5.0, 10.0})));
  const double nvlink_us = run_two_flows(std::make_shared<Topology>(
      Topology::nvlink_mesh(4, LinkParams{5.0, 10.0})));

  // Each transfer occupies 100 us of wire time. The PCIe paths share resource
  // 0 and serialize (200 us + 5 us latency); NVLink paths finish concurrently.
  CHECK_NEAR(pcie_us, 205.0, 1e-9);
  CHECK_NEAR(nvlink_us, 105.0, 1e-9);
}

TEST(topology_shared_resource_clock_resets_between_successful_runs) {
  WorldConfig config = cfg(4);
  config.topology = std::make_shared<Topology>(
      Topology::pcie_switch(4, LinkParams{0.0, 20.0}, LinkParams{5.0, 10.0}));
  World world(config);
  auto run = [&] {
    return world.run([](Transport& tp) {
      std::vector<std::byte> buffer(1'000'000);
      if (tp.rank() % 2 == 0) {
        tp.send(tp.rank() + 1, buffer.data(), buffer.size(), 720);
      } else {
        tp.recv(tp.rank() - 1, buffer.data(), buffer.size(), 720);
      }
    }).makespan_us;
  };
  CHECK_NEAR(run(), 205.0, 1e-9);
  CHECK_NEAR(run(), 205.0, 1e-9);
}

TEST(topology_world_rejects_rank_count_mismatch) {
  WorldConfig config = cfg(2);
  config.topology =
      std::make_shared<Topology>(Topology::fully_connected(3, LinkParams{1.0, 10.0}));
  CHECK_THROWS((void)World{config}, gclab::CommError);
}
