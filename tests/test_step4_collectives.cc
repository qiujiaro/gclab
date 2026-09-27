// Step 4/5: Broadcast and Reduce correctness + simulated cost. CPU_ONLY.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gclab/common/error.h"
#include "gclab/sim/collectives.h"
#include "gclab/sim/world.h"
#include "testlib.h"

using namespace std::chrono_literals;
using namespace gclab::sim;

static WorldConfig cfg(int n) {
  WorldConfig c;
  c.nranks = n;
  c.link = {5.0, 10.0};
  c.recv_timeout = 2000ms;
  return c;
}

// Deterministic per-rank input so failures are reproducible.
static float input(int rank, std::size_t i) { return static_cast<float>((rank + 1) * 1000 + (i % 97)); }

using BcastFn = void (*)(Transport&, void*, std::size_t, int);

static void check_broadcast(BcastFn fn, int p, std::size_t bytes, int root) {
  World w(cfg(p));
  w.run([&](Transport& tp) {
    std::vector<std::uint8_t> buf(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
      buf[i] = tp.rank() == root ? static_cast<std::uint8_t>(i * 7 + root) : 0xEE;
    }
    fn(tp, buf.data(), bytes, root);
    for (std::size_t i = 0; i < bytes; ++i) {
      if (buf[i] != static_cast<std::uint8_t>(i * 7 + root)) {
        throw std::runtime_error("rank " + std::to_string(tp.rank()) + " wrong byte " +
                                 std::to_string(i));
      }
    }
  });
}

TEST(broadcast_linear_correct) {
  for (int p : {1, 2, 3, 5, 8})
    for (int root = 0; root < p; ++root) check_broadcast(broadcast_linear, p, 4096, root);
}

TEST(broadcast_binomial_correct_all_sizes_and_roots) {
  for (int p = 1; p <= 9; ++p)
    for (int root = 0; root < p; ++root)
      for (std::size_t bytes : {std::size_t{0}, std::size_t{1}, std::size_t{4096}})
        check_broadcast(broadcast_binomial, p, bytes, root);
}

static double bcast_makespan(BcastFn fn, int p, std::size_t bytes) {
  World w(cfg(p));
  return w.run([&](Transport& tp) {
            std::vector<std::uint8_t> buf(bytes);
            fn(tp, buf.data(), bytes, 0);
          }).makespan_us;
}

TEST(broadcast_linear_cost) {
  // alpha + (p-1) * n/beta, with n/beta = 100 us for 1 MB at 10 GB/s
  CHECK_NEAR(bcast_makespan(broadcast_linear, 8, 1'000'000), 5.0 + 7 * 100.0, 1e-6);
}

TEST(broadcast_binomial_cost_power_of_two) {
  for (int p : {2, 4, 8, 16}) {
    const double expect = std::log2(p) * (5.0 + 100.0);
    CHECK_NEAR(bcast_makespan(broadcast_binomial, p, 1'000'000), expect, 1e-6);
  }
}

TEST(broadcast_binomial_cost_bound_non_power_of_two) {
  for (int p : {3, 5, 6, 7, 9}) {
    const double bound = std::ceil(std::log2(p)) * (5.0 + 100.0);
    CHECK(bcast_makespan(broadcast_binomial, p, 1'000'000) <= bound + 1e-6);
  }
}

using ReduceFn = void (*)(Transport&, const float*, float*, std::size_t, ReduceOp, int);

static void check_reduce(ReduceFn fn, int p, std::size_t count, ReduceOp op, int root) {
  World w(cfg(p));
  w.run([&](Transport& tp) {
    std::vector<float> send(count), recv(count, -1.0f);
    for (std::size_t i = 0; i < count; ++i) send[i] = input(tp.rank(), i);
    const std::vector<float> send_copy = send;
    fn(tp, send.data(), tp.rank() == root ? recv.data() : nullptr, count, op, root);
    if (send != send_copy) throw std::runtime_error("sendbuf was modified");
    if (tp.rank() != root) return;
    for (std::size_t i = 0; i < count; ++i) {
      float expect = op == ReduceOp::Sum ? 0.0f : input(0, i);
      for (int r = 0; r < p; ++r)
        expect = op == ReduceOp::Sum ? expect + input(r, i) : std::max(expect, input(r, i));
      // Inputs are small integers, so float sums are exact in any order.
      if (recv[i] != expect) throw std::runtime_error("wrong reduce result at " + std::to_string(i));
    }
  });
}

TEST(reduce_linear_correct) {
  for (int p : {1, 2, 3, 4, 7})
    for (int root = 0; root < p; ++root)
      for (ReduceOp op : {ReduceOp::Sum, ReduceOp::Max}) check_reduce(reduce_linear, p, 1000, op, root);
}

TEST(reduce_linear_cost) {
  // p-1 messages serialize on root's incoming port: alpha + (p-1) * n/beta.
  World w(cfg(8));
  constexpr std::size_t kCount = 250'000;  // 1 MB of floats
  auto res = w.run([&](Transport& tp) {
    std::vector<float> send(kCount, 1.0f), recv(kCount);
    reduce_linear(tp, send.data(), recv.data(), kCount, ReduceOp::Sum, 0);
  });
  CHECK_NEAR(res.final_clock_us[0], 5.0 + 7 * 100.0, 1e-6);
}

TEST(reduce_binomial_correct_p1_stretch) {
  for (int p = 1; p <= 9; ++p)
    for (int root = 0; root < p; ++root) check_reduce(reduce_binomial, p, 333, ReduceOp::Sum, root);
}

TEST(mismatched_collective_order_fails_bounded) {
  // Rank 0 calls broadcast, rank 1 calls reduce. Must fail, never hang.
  WorldConfig c = cfg(2);
  c.recv_timeout = 200ms;
  World w(c);
  const auto t0 = std::chrono::steady_clock::now();
  bool failed = false;
  try {
    w.run([](Transport& tp) {
      std::vector<float> a(16, 1.0f), b(16);
      if (tp.rank() == 0) broadcast_linear(tp, a.data(), a.size() * sizeof(float), 1);
      else reduce_linear(tp, a.data(), nullptr, a.size(), ReduceOp::Sum, 0);
    });
  } catch (const gclab::NotImplemented&) {
    throw;
  } catch (const std::exception&) {
    failed = true;
  }
  CHECK(failed);
  CHECK(std::chrono::steady_clock::now() - t0 < 2000ms);
}
