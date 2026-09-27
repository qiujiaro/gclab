// Step 2: thread-safe tag-matched channel. CPU_ONLY.
// Run under ThreadSanitizer too:  cmake -B build-tsan -DGCLAB_SANITIZER=thread
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <thread>

#include "gclab/common/error.h"
#include "gclab/sim/channel.h"
#include "testlib.h"

using namespace std::chrono_literals;
using gclab::sim::Channel;
using gclab::sim::Message;

static Message make(int tag, int value) {
  Message m;
  m.tag = tag;
  m.payload.resize(sizeof(int));
  std::memcpy(m.payload.data(), &value, sizeof(int));
  return m;
}

// std::thread that forwards an exception to join() instead of calling
// std::terminate. Exceptions escaping a thread body kill the whole process.
class SafeThread {
 public:
  template <typename F>
  explicit SafeThread(F f)
      : t_([this, f] {
          try {
            f();
          } catch (...) {
            err_ = std::current_exception();
          }
        }) {}
  void join() {
    t_.join();
    if (err_) std::rethrow_exception(err_);
  }
  ~SafeThread() {
    if (t_.joinable()) t_.join();
  }

 private:
  std::exception_ptr err_;
  std::thread t_;  // declared last: starts after err_ is constructed
};

static int value_of(const Message& m) {
  int v = 0;
  std::memcpy(&v, m.payload.data(), sizeof(int));
  return v;
}

TEST(fifo_within_tag) {
  Channel ch;
  for (int i = 0; i < 5; ++i) ch.push(make(7, i));
  for (int i = 0; i < 5; ++i) CHECK_EQ(value_of(ch.pop(7, 100ms)), i);
  CHECK_EQ(ch.pending(), 0u);
}

TEST(tags_can_overtake) {
  Channel ch;
  ch.push(make(1, 10));
  ch.push(make(2, 20));
  CHECK_EQ(value_of(ch.pop(2, 100ms)), 20);  // skips the older tag-1 message
  CHECK_EQ(value_of(ch.pop(1, 100ms)), 10);
}

TEST(pop_times_out_and_is_bounded) {
  Channel ch;
  ch.push(make(1, 10));  // wrong tag must not satisfy the pop
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_THROWS(ch.pop(2, 50ms), gclab::TimeoutError);
  const auto waited = std::chrono::steady_clock::now() - t0;
  CHECK(waited >= 45ms);
  CHECK(waited < 1000ms);
  CHECK_EQ(ch.pending(), 1u);
}

TEST(blocking_pop_wakes_on_push) {
  Channel ch;
  SafeThread producer([&] {
    std::this_thread::sleep_for(20ms);
    ch.push(make(3, 42));
  });
  CHECK_EQ(value_of(ch.pop(3, 2000ms)), 42);
  producer.join();
}

TEST(abort_wakes_blocked_pop) {
  Channel ch;
  SafeThread killer([&] {
    std::this_thread::sleep_for(20ms);
    ch.abort();
  });
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_THROWS(ch.pop(3, 5000ms), gclab::AbortedError);
  CHECK(std::chrono::steady_clock::now() - t0 < 1000ms);  // did not wait for the timeout
  killer.join();
  CHECK_THROWS(ch.pop(3, 10ms), gclab::AbortedError);     // sticky
}

TEST(many_messages_concurrent) {
  Channel ch;
  constexpr int kN = 20000;
  SafeThread producer([&] {
    for (int i = 0; i < kN; ++i) ch.push(make(i % 2, i));
  });
  // Consume tag 0 and tag 1 alternately; each stream must stay ordered.
  int next0 = 0, next1 = 1;
  for (int i = 0; i < kN; ++i) {
    if (i % 2 == 0) {
      CHECK_EQ(value_of(ch.pop(0, 2000ms)), next0);
      next0 += 2;
    } else {
      CHECK_EQ(value_of(ch.pop(1, 2000ms)), next1);
      next1 += 2;
    }
  }
  producer.join();
}
