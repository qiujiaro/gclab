// M2/T7: Ring ReduceScatter, AllGather and AllReduce. CPU_ONLY.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

#include "gclab/common/error.h"
#include "gclab/sim/collectives.h"
#include "gclab/sim/inproc_transport.h"
#include "gclab/sim/world.h"
#include "testlib.h"

using namespace std::chrono_literals;
using namespace gclab::sim;

namespace {

void check_partition(std::size_t total_count, int nranks,
                     const std::vector<std::size_t>& expected_counts) {
  CHECK_EQ(expected_counts.size(), static_cast<std::size_t>(nranks));

  std::size_t next_offset = 0;
  for (int rank = 0; rank < nranks; ++rank) {
    const BlockRange block = block_for_rank(total_count, nranks, rank);
    CHECK_EQ(block.offset, next_offset);
    CHECK_EQ(block.count, expected_counts[static_cast<std::size_t>(rank)]);
    next_offset += block.count;
  }
  CHECK_EQ(next_offset, total_count);
}

WorldConfig world_cfg(int nranks) {
  WorldConfig config;
  config.nranks = nranks;
  config.link = {5.0, 10.0};
  config.recv_timeout = 2000ms;
  return config;
}

float gathered_value(int owner, std::size_t global_index) {
  return static_cast<float>((owner + 1) * 1000 + global_index);
}

void check_allgather(int nranks, std::size_t count) {
  std::vector<float> expected(count);
  for (int owner = 0; owner < nranks; ++owner) {
    const BlockRange block = block_for_rank(count, nranks, owner);
    for (std::size_t i = 0; i < block.count; ++i) {
      expected[block.offset + i] = gathered_value(owner, block.offset + i);
    }
  }

  World world(world_cfg(nranks));
  world.run([&](Transport& tp) {
    const BlockRange local = block_for_rank(count, nranks, tp.rank());
    std::vector<float> send(local.count);
    for (std::size_t i = 0; i < local.count; ++i) {
      send[i] = gathered_value(tp.rank(), local.offset + i);
    }
    const std::vector<float> send_copy = send;
    std::vector<float> recv(count, -1.0f);

    allgather_ring(tp, send.data(), recv.data(), count, 0);
    CHECK(send == send_copy);
    CHECK(recv == expected);
  });
}

float reduce_input(int rank, std::size_t global_index) {
  return static_cast<float>((rank + 1) * 100 + (global_index % 31));
}

float reduced_value(int nranks, std::size_t global_index, ReduceOp op) {
  float result = reduce_input(0, global_index);
  for (int rank = 1; rank < nranks; ++rank) {
    const float value = reduce_input(rank, global_index);
    result = op == ReduceOp::Sum ? result + value : std::max(result, value);
  }
  return result;
}

void check_reduce_scatter(int nranks, std::size_t count, ReduceOp op) {
  World world(world_cfg(nranks));
  world.run([&](Transport& tp) {
    std::vector<float> send(count);
    for (std::size_t i = 0; i < count; ++i) {
      send[i] = reduce_input(tp.rank(), i);
    }
    const std::vector<float> send_copy = send;
    const BlockRange local = block_for_rank(count, nranks, tp.rank());
    std::vector<float> recv(local.count, -1.0f);

    reduce_scatter_ring(tp, send.data(), recv.data(), count, op, 0);

    CHECK(send == send_copy);
    for (std::size_t i = 0; i < local.count; ++i) {
      CHECK_EQ(recv[i], reduced_value(nranks, local.offset + i, op));
    }
  });
}

void check_allreduce(int nranks, std::size_t count, ReduceOp op) {
  World world(world_cfg(nranks));
  world.run([&](Transport& tp) {
    std::vector<float> send(count);
    for (std::size_t i = 0; i < count; ++i) {
      send[i] = reduce_input(tp.rank(), i);
    }
    const std::vector<float> send_copy = send;
    std::vector<float> recv(count, -1.0f);

    allreduce_ring(tp, send.data(), recv.data(), count, op, 0);

    CHECK(send == send_copy);
    for (std::size_t i = 0; i < count; ++i) {
      CHECK_EQ(recv[i], reduced_value(nranks, i, op));
    }
  });
}

}  // namespace

TEST(block_for_rank_even_partition) {
  check_partition(8, 4, {2, 2, 2, 2});
}

TEST(block_for_rank_uneven_partition_gives_remainder_to_first_ranks) {
  check_partition(10, 3, {4, 3, 3});
  check_partition(11, 4, {3, 3, 3, 2});
}

TEST(block_for_rank_count_smaller_than_world) {
  check_partition(3, 5, {1, 1, 1, 0, 0});
}

TEST(block_for_rank_zero_count) {
  check_partition(0, 4, {0, 0, 0, 0});
}

TEST(block_for_rank_partitions_are_contiguous_and_complete) {
  for (std::size_t total_count : {
           std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3},
           std::size_t{7}, std::size_t{16}, std::size_t{31}}) {
    for (int nranks = 1; nranks <= 16; ++nranks) {
      std::size_t next_offset = 0;
      const std::size_t base = total_count / static_cast<std::size_t>(nranks);
      const std::size_t remainder = total_count % static_cast<std::size_t>(nranks);

      for (int rank = 0; rank < nranks; ++rank) {
        const BlockRange block = block_for_rank(total_count, nranks, rank);
        const std::size_t expected_count =
            base + (static_cast<std::size_t>(rank) < remainder ? 1 : 0);
        CHECK_EQ(block.offset, next_offset);
        CHECK_EQ(block.count, expected_count);
        next_offset += block.count;
      }
      CHECK_EQ(next_offset, total_count);
    }
  }
}

TEST(block_for_rank_rejects_invalid_inputs) {
  // Check invalid ranks before nranks==0 so a broken implementation reports a
  // normal test failure instead of reaching an integer division by zero.
  CHECK_THROWS(block_for_rank(10, 3, -1), gclab::CommError);
  CHECK_THROWS(block_for_rank(10, 3, 3), gclab::CommError);
  CHECK_THROWS(block_for_rank(10, 0, 0), gclab::CommError);
  CHECK_THROWS(block_for_rank(10, -1, 0), gclab::CommError);
}

TEST(allgather_ring_correct_for_even_and_uneven_counts) {
  for (int nranks = 1; nranks <= 16; ++nranks) {
    for (std::size_t count : {
             std::size_t{0}, std::size_t{1}, std::size_t{3},
             std::size_t{7}, std::size_t{17}, std::size_t{32}}) {
      check_allgather(nranks, count);
    }
  }
}

TEST(allgather_ring_supports_in_place_local_block) {
  constexpr int kRanks = 5;
  constexpr std::size_t kCount = 13;
  World world(world_cfg(kRanks));
  world.run([](Transport& tp) {
    const BlockRange local = block_for_rank(kCount, kRanks, tp.rank());
    std::vector<float> recv(kCount, -1.0f);
    for (std::size_t i = 0; i < local.count; ++i) {
      recv[local.offset + i] = gathered_value(tp.rank(), local.offset + i);
    }

    allgather_ring(tp, recv.data() + local.offset, recv.data(), kCount, 0);

    for (int owner = 0; owner < kRanks; ++owner) {
      const BlockRange block = block_for_rank(kCount, kRanks, owner);
      for (std::size_t i = 0; i < block.count; ++i) {
        CHECK_EQ(recv[block.offset + i], gathered_value(owner, block.offset + i));
      }
    }
  });
}

TEST(allgather_ring_validates_buffers_and_supports_chunking) {
  World world(world_cfg(2));
  InProcTransport rank0(world, 0);
  float send = 1.0f;
  std::vector<float> recv(2);

  CHECK_THROWS(allgather_ring(rank0, nullptr, recv.data(), 2, 0), gclab::CommError);
  CHECK_THROWS(allgather_ring(rank0, &send, nullptr, 2, 0), gclab::CommError);
  allgather_ring(rank0, nullptr, nullptr, 0, 0);

  World single_world(world_cfg(1));
  InProcTransport only_rank(single_world, 0);
  float single_recv = 0.0f;
  allgather_ring(only_rank, &send, &single_recv, 1, sizeof(float));
  CHECK_EQ(single_recv, send);
}

TEST(reduce_scatter_ring_correct_for_sum_and_max) {
  for (int nranks = 1; nranks <= 16; ++nranks) {
    for (std::size_t count : {
             std::size_t{0}, std::size_t{1}, std::size_t{3},
             std::size_t{7}, std::size_t{17}, std::size_t{32}}) {
      for (ReduceOp op : {ReduceOp::Sum, ReduceOp::Max}) {
        check_reduce_scatter(nranks, count, op);
      }
    }
  }
}

TEST(reduce_scatter_ring_validates_buffers_and_supports_chunking) {
  World world(world_cfg(2));
  InProcTransport rank0(world, 0);
  std::vector<float> send(2, 1.0f);
  float recv = 0.0f;

  CHECK_THROWS(reduce_scatter_ring(rank0, nullptr, &recv, 2, ReduceOp::Sum, 0),
               gclab::CommError);
  CHECK_THROWS(reduce_scatter_ring(rank0, send.data(), nullptr, 2, ReduceOp::Sum, 0),
               gclab::CommError);
  reduce_scatter_ring(rank0, nullptr, nullptr, 0, ReduceOp::Sum, 0);

  World single_world(world_cfg(1));
  InProcTransport only_rank(single_world, 0);
  reduce_scatter_ring(only_rank, send.data(), &recv, 1, ReduceOp::Sum, sizeof(float));
  CHECK_EQ(recv, send.front());
}

TEST(allreduce_ring_correct_for_sum_and_max) {
  for (int nranks = 1; nranks <= 16; ++nranks) {
    for (std::size_t count : {
             std::size_t{0}, std::size_t{1}, std::size_t{3},
             std::size_t{7}, std::size_t{17}, std::size_t{32}}) {
      for (ReduceOp op : {ReduceOp::Sum, ReduceOp::Max}) {
        check_allreduce(nranks, count, op);
      }
    }
  }
}

TEST(allreduce_ring_supports_in_place_buffers) {
  constexpr int kRanks = 5;
  constexpr std::size_t kCount = 13;
  World world(world_cfg(kRanks));
  world.run([](Transport& tp) {
    std::vector<float> buffer(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
      buffer[i] = reduce_input(tp.rank(), i);
    }

    allreduce_ring(tp, buffer.data(), buffer.data(), kCount, ReduceOp::Sum, 0);

    for (std::size_t i = 0; i < kCount; ++i) {
      CHECK_EQ(buffer[i], reduced_value(kRanks, i, ReduceOp::Sum));
    }
  });
}

TEST(allreduce_ring_validates_buffers_and_supports_chunking) {
  World world(world_cfg(2));
  InProcTransport rank0(world, 0);
  std::vector<float> send(2, 1.0f), recv(2);

  CHECK_THROWS(allreduce_ring(rank0, nullptr, recv.data(), 2, ReduceOp::Sum, 0),
               gclab::CommError);
  CHECK_THROWS(allreduce_ring(rank0, send.data(), nullptr, 2, ReduceOp::Sum, 0),
               gclab::CommError);
  allreduce_ring(rank0, nullptr, nullptr, 0, ReduceOp::Sum, 0);

  World single_world(world_cfg(1));
  InProcTransport only_rank(single_world, 0);
  float single_recv = 0.0f;
  allreduce_ring(only_rank, send.data(), &single_recv, 1, ReduceOp::Sum, sizeof(float));
  CHECK_EQ(single_recv, send.front());
}

TEST(allreduce_ring_cost_matches_two_ring_phases) {
  constexpr int kRanks = 4;
  constexpr std::size_t kCount = 250'000;  // 1 MB total, 250 KB per block.
  World world(world_cfg(kRanks));
  const RunResult result = world.run([](Transport& tp) {
    std::vector<float> send(kCount, 1.0f), recv(kCount);
    allreduce_ring(tp, send.data(), recv.data(), kCount, ReduceOp::Sum, 0);
  });

  // Two phases, each with p-1 steps. Every step pays 5 us latency plus
  // 250 KB / 10 GB/s = 25 us wire time.
  CHECK_NEAR(result.makespan_us, 2.0 * (kRanks - 1) * (5.0 + 25.0), 1e-9);
}

TEST(mismatched_ring_collectives_fail_bounded) {
  WorldConfig config = world_cfg(2);
  config.recv_timeout = 200ms;
  World world(config);
  const auto start = std::chrono::steady_clock::now();
  bool failed = false;
  try {
    world.run([](Transport& tp) {
      std::vector<float> full(4, static_cast<float>(tp.rank() + 1));
      if (tp.rank() == 0) {
        std::vector<float> shard(2);
        reduce_scatter_ring(tp, full.data(), shard.data(), full.size(), ReduceOp::Sum, 0);
      } else {
        std::vector<float> gathered(4);
        allgather_ring(tp, full.data(), gathered.data(), gathered.size(), 0);
      }
    });
  } catch (const std::exception&) {
    failed = true;
  }
  CHECK(failed);
  CHECK(std::chrono::steady_clock::now() - start < 2000ms);
}
