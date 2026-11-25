// M2/T6: non-blocking Transport requests. CPU_ONLY.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "gclab/common/error.h"
#include "gclab/sim/inproc_transport.h"
#include "gclab/sim/world.h"
#include "testlib.h"

using namespace std::chrono_literals;
using namespace gclab::sim;

namespace {

WorldConfig cfg(int nranks = 2) {
  WorldConfig config;
  config.nranks = nranks;
  config.link = {5.0, 10.0};
  config.recv_timeout = 200ms;
  return config;
}

}  // namespace

TEST(async_send_recv_payload_and_timing) {
  World world(cfg());
  world.run([](Transport& tp) {
    constexpr std::size_t kBytes = 1'000'000;
    if (tp.rank() == 0) {
      std::vector<std::uint8_t> send(kBytes, 0x5A);
      auto request = tp.isend(1, send.data(), send.size(), 10);

      // Posting is non-blocking in simulated time. Waiting observes the
      // 1 MB / 10 GB/s = 100 us send completion.
      CHECK_NEAR(tp.now_us(), 0.0, 1e-9);
      tp.wait(*request);
      CHECK_NEAR(tp.now_us(), 100.0, 1e-9);
    } else {
      std::vector<std::uint8_t> recv(kBytes);
      auto request = tp.irecv(0, recv.data(), recv.size(), 10);

      CHECK_NEAR(tp.now_us(), 0.0, 1e-9);
      tp.wait(*request);
      CHECK_EQ(recv.front(), static_cast<std::uint8_t>(0x5A));
      CHECK_EQ(recv.back(), static_cast<std::uint8_t>(0x5A));
      // Receiver pays the 5 us link latency in addition to wire time.
      CHECK_NEAR(tp.now_us(), 105.0, 1e-9);
    }
  });
}

TEST(async_receive_can_be_posted_before_send) {
  World world(cfg());
  std::atomic<bool> receive_posted{false};
  world.run([&](Transport& tp) {
    std::uint8_t value = 0;
    if (tp.rank() == 0) {
      const auto deadline = std::chrono::steady_clock::now() + 1000ms;
      while (!receive_posted.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("irecv blocked instead of returning a request");
        }
        std::this_thread::yield();
      }
      value = 42;
      auto request = tp.isend(1, &value, sizeof(value), 15);
      tp.wait(*request);
    } else {
      auto request = tp.irecv(0, &value, sizeof(value), 15);
      receive_posted.store(true, std::memory_order_release);
      tp.wait(*request);
      CHECK_EQ(value, static_cast<std::uint8_t>(42));
    }
  });
}

TEST(multiple_async_sends_serialize_the_tx_port) {
  World world(cfg());
  world.run([](Transport& tp) {
    constexpr std::size_t kBytes = 1'000'000;
    if (tp.rank() == 0) {
      std::vector<std::uint8_t> first_payload(kBytes, 1);
      std::vector<std::uint8_t> second_payload(kBytes, 2);
      auto first = tp.isend(1, first_payload.data(), first_payload.size(), 20);
      auto second = tp.isend(1, second_payload.data(), second_payload.size(), 20);

      CHECK_NEAR(tp.now_us(), 0.0, 1e-9);
      tp.wait(*first);
      CHECK_NEAR(tp.now_us(), 100.0, 1e-9);
      tp.wait(*second);
      CHECK_NEAR(tp.now_us(), 200.0, 1e-9);
    } else {
      std::vector<std::uint8_t> first_payload(kBytes);
      std::vector<std::uint8_t> second_payload(kBytes);
      auto first = tp.irecv(0, first_payload.data(), first_payload.size(), 20);
      auto second = tp.irecv(0, second_payload.data(), second_payload.size(), 20);

      tp.wait(*first);
      tp.wait(*second);
      CHECK_EQ(first_payload.front(), static_cast<std::uint8_t>(1));
      CHECK_EQ(second_payload.front(), static_cast<std::uint8_t>(2));
      CHECK_NEAR(tp.now_us(), 205.0, 1e-9);
    }
  });
}

TEST(async_zero_byte_message) {
  World world(cfg());
  world.run([](Transport& tp) {
    if (tp.rank() == 0) {
      auto request = tp.isend(1, nullptr, 0, 30);
      tp.wait(*request);
      CHECK_NEAR(tp.now_us(), 0.0, 1e-9);
    } else {
      auto request = tp.irecv(0, nullptr, 0, 30);
      tp.wait(*request);
      CHECK_NEAR(tp.now_us(), 5.0, 1e-9);
    }
  });
}

TEST(async_rejects_bad_peers_and_null_buffers) {
  World world(cfg());
  world.run([](Transport& tp) {
    std::uint8_t byte = 0;
    CHECK_THROWS(tp.isend(-1, &byte, 1, 40), gclab::CommError);
    CHECK_THROWS(tp.isend(tp.size(), &byte, 1, 40), gclab::CommError);
    CHECK_THROWS(tp.irecv(-1, &byte, 1, 40), gclab::CommError);
    CHECK_THROWS(tp.irecv(tp.size(), &byte, 1, 40), gclab::CommError);
    CHECK_THROWS(tp.isend(0, nullptr, 1, 40), gclab::CommError);
    CHECK_THROWS(tp.irecv(0, nullptr, 1, 40), gclab::CommError);
  });
}

TEST(async_rejects_repeated_wait) {
  World world(cfg());
  world.run([](Transport& tp) {
    if (tp.rank() == 0) {
      std::uint8_t send = 7;
      auto request = tp.isend(1, &send, sizeof(send), 50);
      tp.wait(*request);
      CHECK_THROWS(tp.wait(*request), gclab::CommError);
    } else {
      std::uint8_t recv = 0;
      auto request = tp.irecv(0, &recv, sizeof(recv), 50);
      tp.wait(*request);
      CHECK_EQ(recv, static_cast<std::uint8_t>(7));
      CHECK_THROWS(tp.wait(*request), gclab::CommError);
    }
  });
}

TEST(async_rejects_request_from_another_transport) {
  World world(cfg());
  InProcTransport rank0(world, 0);
  InProcTransport rank1(world, 1);
  std::uint8_t payload = 9;
  auto request = rank0.isend(1, &payload, sizeof(payload), 60);

  CHECK_THROWS(rank1.wait(*request), gclab::CommError);
  // A rejected foreign wait must not consume the request.
  rank0.wait(*request);
}

TEST(async_receive_size_mismatch_fails) {
  World world(cfg());
  bool failed = false;
  try {
    world.run([](Transport& tp) {
      if (tp.rank() == 0) {
        std::uint32_t send = 123;
        auto request = tp.isend(1, &send, sizeof(send), 70);
        tp.wait(*request);
      } else {
        std::uint64_t recv = 0;
        auto request = tp.irecv(0, &recv, sizeof(recv), 70);
        tp.wait(*request);
      }
    });
  } catch (const gclab::CommError&) {
    failed = true;
  }
  CHECK(failed);
}

TEST(async_tag_mismatch_times_out_bounded) {
  World world(cfg());
  const auto start = std::chrono::steady_clock::now();
  bool failed = false;
  try {
    world.run([](Transport& tp) {
      std::uint8_t value = 0;
      if (tp.rank() == 0) {
        auto request = tp.isend(1, &value, sizeof(value), 80);
        tp.wait(*request);
      } else {
        auto request = tp.irecv(0, &value, sizeof(value), 81);
        tp.wait(*request);
      }
    });
  } catch (const gclab::TimeoutError&) {
    failed = true;
  }
  CHECK(failed);
  CHECK(std::chrono::steady_clock::now() - start < 2000ms);
}

TEST(async_receive_wakes_when_world_is_aborted) {
  World world(cfg());
  std::atomic<bool> receive_posted{false};
  const auto start = std::chrono::steady_clock::now();
  bool failed = false;
  try {
    world.run([&](Transport& tp) {
      if (tp.rank() == 0) {
        const auto deadline = std::chrono::steady_clock::now() + 1000ms;
        while (!receive_posted.load(std::memory_order_acquire)) {
          if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("receive request was never posted");
          }
          std::this_thread::yield();
        }
        throw std::runtime_error("injected rank failure");
      }

      std::uint8_t recv = 0;
      auto request = tp.irecv(0, &recv, sizeof(recv), 85);
      receive_posted.store(true, std::memory_order_release);
      tp.wait(*request);
    });
  } catch (const std::runtime_error&) {
    failed = true;
  }
  CHECK(failed);
  CHECK(std::chrono::steady_clock::now() - start < 2000ms);
}

TEST(wait_all_completes_every_request) {
  World world(cfg());
  world.run([](Transport& tp) {
    if (tp.rank() == 0) {
      std::uint8_t first_payload = 11;
      std::uint8_t second_payload = 22;
      std::vector<RequestPtr> requests;
      requests.push_back(tp.isend(1, &first_payload, sizeof(first_payload), 90));
      requests.push_back(tp.isend(1, &second_payload, sizeof(second_payload), 91));
      tp.wait_all(requests);
    } else {
      std::uint8_t first_payload = 0;
      std::uint8_t second_payload = 0;
      std::vector<RequestPtr> requests;
      requests.push_back(tp.irecv(0, &first_payload, sizeof(first_payload), 90));
      requests.push_back(tp.irecv(0, &second_payload, sizeof(second_payload), 91));
      tp.wait_all(requests);
      CHECK_EQ(first_payload, static_cast<std::uint8_t>(11));
      CHECK_EQ(second_payload, static_cast<std::uint8_t>(22));
    }
  });
}

TEST(wait_all_rejects_null_request) {
  World world(cfg(1));
  world.run([](Transport& tp) {
    std::vector<RequestPtr> requests;
    requests.push_back(nullptr);
    CHECK_THROWS(tp.wait_all(requests), gclab::CommError);
  });
}

TEST(wait_all_accepts_empty_request_list) {
  World world(cfg(1));
  world.run([](Transport& tp) {
    std::vector<RequestPtr> requests;
    tp.wait_all(requests);
    CHECK_NEAR(tp.now_us(), 0.0, 1e-9);
  });
}

TEST(wait_rejects_unknown_request_type) {
  class UnknownRequest final : public Request {};

  World world(cfg(1));
  world.run([](Transport& tp) {
    UnknownRequest request;
    CHECK_THROWS(tp.wait(request), gclab::CommError);
  });
}
