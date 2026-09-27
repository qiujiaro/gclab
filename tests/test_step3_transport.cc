// Step 3: InProcTransport send/recv + simulated clock + failure propagation.
// CPU_ONLY.
#include <chrono>
#include <cstring>
#include <vector>

#include "gclab/common/error.h"
#include "gclab/sim/world.h"
#include "testlib.h"

using namespace std::chrono_literals;
using namespace gclab::sim;

static WorldConfig cfg(int n) {
  WorldConfig c;
  c.nranks = n;
  c.link = {5.0, 10.0};   // alpha 5 us, 10 GB/s -> 1 MB = 100 us of wire time
  c.recv_timeout = 500ms;
  return c;
}
constexpr std::size_t kMB = 1'000'000;

TEST(p2p_data_and_clock) {
  World w(cfg(2));
  auto res = w.run([](Transport& tp) {
    std::vector<char> buf(kMB);
    if (tp.rank() == 0) {
      for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<char>(i * 31);
      tp.send(1, buf.data(), buf.size(), 0);
      std::memset(buf.data(), 0, buf.size());  // eager: safe to overwrite now
    } else {
      tp.recv(0, buf.data(), buf.size(), 0);
      for (std::size_t i = 0; i < buf.size(); ++i) {
        if (buf[i] != static_cast<char>(i * 31)) throw std::runtime_error("corrupt payload");
      }
    }
  });
  CHECK_NEAR(res.final_clock_us[0], 100.0, 1e-6);  // sender: wire time only
  CHECK_NEAR(res.final_clock_us[1], 105.0, 1e-6);  // receiver: alpha + wire
}

TEST(ping_pong_clock) {
  World w(cfg(2));
  auto res = w.run([](Transport& tp) {
    std::vector<char> buf(kMB);
    if (tp.rank() == 0) {
      tp.send(1, buf.data(), buf.size(), 0);
      tp.recv(1, buf.data(), buf.size(), 0);
    } else {
      tp.recv(0, buf.data(), buf.size(), 0);
      tp.send(0, buf.data(), buf.size(), 0);
    }
  });
  CHECK_NEAR(res.final_clock_us[0], 210.0, 1e-6);  // 2 * (alpha + wire)
}

TEST(sender_port_serializes) {
  // Rank 0 sends 1 MB to ranks 1, 2, 3 in that order.
  World w(cfg(4));
  auto res = w.run([](Transport& tp) {
    std::vector<char> buf(kMB);
    if (tp.rank() == 0) {
      for (int d = 1; d < 4; ++d) tp.send(d, buf.data(), buf.size(), 0);
    } else {
      tp.recv(0, buf.data(), buf.size(), 0);
    }
  });
  CHECK_NEAR(res.final_clock_us[0], 300.0, 1e-6);
  CHECK_NEAR(res.final_clock_us[1], 105.0, 1e-6);
  CHECK_NEAR(res.final_clock_us[2], 205.0, 1e-6);
  CHECK_NEAR(res.final_clock_us[3], 305.0, 1e-6);
}

TEST(receiver_port_serializes) {
  // Ranks 1..3 all send 1 MB to rank 0 at time 0.
  World w(cfg(4));
  auto res = w.run([](Transport& tp) {
    std::vector<char> buf(kMB);
    if (tp.rank() == 0) {
      for (int s = 1; s < 4; ++s) tp.recv(s, buf.data(), buf.size(), 0);
    } else {
      tp.send(0, buf.data(), buf.size(), 0);
    }
  });
  CHECK_NEAR(res.final_clock_us[0], 305.0, 1e-6);  // alpha + 3 * wire, not 105
}

TEST(size_mismatch_is_detected) {
  World w(cfg(2));
  CHECK_THROWS(w.run([](Transport& tp) {
                 std::vector<char> buf(1024);
                 if (tp.rank() == 0) tp.send(1, buf.data(), 1024, 0);
                 else tp.recv(0, buf.data(), 512, 0);
               }),
               gclab::CommError);
}

TEST(bad_peer_is_detected) {
  World w(cfg(2));
  CHECK_THROWS(w.run([](Transport& tp) {
                 char c = 0;
                 if (tp.rank() == 0) tp.send(7, &c, 1, 0);
               }),
               gclab::CommError);
}

TEST(deadlock_becomes_bounded_timeout) {
  // Both ranks recv first: a classic deadlock. Must end in TimeoutError,
  // within roughly one recv_timeout, never hang.
  World w(cfg(2));
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_THROWS(w.run([](Transport& tp) {
                 char c = 0;
                 tp.recv(1 - tp.rank(), &c, 1, 0);
                 tp.send(1 - tp.rank(), &c, 1, 0);
               }),
               gclab::TimeoutError);
  CHECK(std::chrono::steady_clock::now() - t0 < 3000ms);
}

TEST(one_rank_failure_aborts_peers_quickly) {
  // Rank 0 throws immediately; rank 1 is blocked in recv with a long timeout.
  // The world must abort rank 1 and rethrow rank 0's original error.
  WorldConfig c = cfg(2);
  c.recv_timeout = 10'000ms;
  World w(c);
  const auto t0 = std::chrono::steady_clock::now();
  bool got_root_cause = false;
  try {
    w.run([](Transport& tp) {
      if (tp.rank() == 0) throw std::runtime_error("rank0 boom");
      char ch = 0;
      tp.recv(0, &ch, 1, 0);
    });
  } catch (const std::runtime_error& e) {
    got_root_cause = std::string(e.what()) == "rank0 boom";
  }
  CHECK(got_root_cause);
  CHECK(std::chrono::steady_clock::now() - t0 < 2000ms);
}
