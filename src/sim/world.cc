#include "gclab/sim/world.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <mutex>
#include <string>
#include <thread>

#include "gclab/common/error.h"
#include "gclab/common/log.h"
#include "gclab/sim/inproc_transport.h"

namespace gclab::sim {

World::World(WorldConfig cfg) : cfg_(cfg) {
  if (cfg_.nranks < 1) throw CommError("nranks must be >= 1");
  const int n = cfg_.nranks;
  channels_.reserve(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n * n; ++i) channels_.push_back(std::make_unique<Channel>());
}

Channel& World::channel(int src, int dst) {
  const int n = cfg_.nranks;
  if (src < 0 || src >= n || dst < 0 || dst >= n) {
    throw CommError("bad channel " + std::to_string(src) + "->" + std::to_string(dst));
  }
  return *channels_[static_cast<std::size_t>(src) * n + dst];
}

void World::abort() {
  for (auto& ch : channels_) ch->abort();
}

RunResult World::run(const std::function<void(Transport&)>& fn) {
  const int n = cfg_.nranks;
  RunResult result;
  result.final_clock_us.assign(n, 0.0);

  std::mutex err_mu;
  std::exception_ptr first_error;

  GCLAB_LOG(Info, "world.run.begin", {{"nranks", log::str(n)}});
  const auto t0 = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;
  threads.reserve(n);
  for (int r = 0; r < n; ++r) {
    threads.emplace_back([&, r] {
      InProcTransport tp(*this, r);
      try {
        fn(tp);
      } catch (const AbortedError&) {
        // Secondary failure caused by another rank's abort; not the root cause.
      } catch (...) {
        bool is_first = false;
        {
          std::lock_guard<std::mutex> lock(err_mu);
          if (!first_error) {
            first_error = std::current_exception();
            is_first = true;
          }
        }
        if (is_first) {
          GCLAB_LOG(Error, "rank.failed", {{"rank", log::str(r)}});
          abort();  // wake every blocked peer
        }
      }
      result.final_clock_us[r] = tp.now_us();
    });
  }
  for (auto& t : threads) t.join();

  const auto t1 = std::chrono::steady_clock::now();
  result.wall_time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  for (double c : result.final_clock_us) result.makespan_us = std::max(result.makespan_us, c);
  GCLAB_LOG(Info, "world.run.end",
            {{"makespan_sim_us", log::str(result.makespan_us)},
             {"wall_us", log::str(result.wall_time_us)}});

  if (first_error) std::rethrow_exception(first_error);
  return result;
}

}  // namespace gclab::sim
