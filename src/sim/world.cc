#include "gclab/sim/world.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <mutex>
#include <string>
#include <thread>

#include "gclab/common/error.h"
#include "gclab/common/log.h"
#include "gclab/sim/inproc_transport.h"
#include "gclab/sim/topology.h"

namespace gclab::sim {

World::World(WorldConfig cfg) : cfg_(cfg) {
  if (cfg_.nranks < 1) throw CommError("nranks must be >= 1");
  if (cfg_.topology && cfg_.topology->nranks() != cfg_.nranks) {
    throw CommError("topology rank count does not match world rank count");
  }
  const auto n = static_cast<std::size_t>(cfg_.nranks);
  channels_.reserve(n * n);
  for (std::size_t i = 0; i < n * n; ++i) {
    channels_.push_back(std::make_unique<Channel>());
  }
}

Channel& World::channel(int src, int dst) {
  const int n = cfg_.nranks;
  if (src < 0 || src >= n || dst < 0 || dst >= n) {
    throw CommError("bad channel " + std::to_string(src) + "->" + std::to_string(dst));
  }
  return *channels_[static_cast<std::size_t>(src) * n + dst];
}

double World::reserve_shared_resources(const std::vector<int>& resources,
                                       double earliest_start_us, double duration_us) {
  if (!std::isfinite(earliest_start_us) || earliest_start_us < 0.0 ||
      !std::isfinite(duration_us) || duration_us < 0.0) {
    throw CommError("invalid shared-resource reservation time");
  }
  if (resources.empty()) return earliest_start_us;

  std::lock_guard<std::mutex> lock(shared_resource_mu_);
  double start_us = earliest_start_us;
  for (int resource : resources) {
    if (resource < 0) throw CommError("shared resource id must be non-negative");
    start_us = std::max(start_us, shared_resource_free_us_[resource]);
  }
  const double finish_us = start_us + duration_us;
  if (!std::isfinite(finish_us)) throw CommError("shared-resource finish time overflow");
  for (int resource : resources) shared_resource_free_us_[resource] = finish_us;
  return start_us;
}

void World::abort() {
  for (auto& ch : channels_) ch->abort();
}

RunResult World::run(const std::function<void(Transport&)>& fn) {
  const int n = cfg_.nranks;
  {
    std::lock_guard<std::mutex> lock(shared_resource_mu_);
    shared_resource_free_us_.clear();
  }
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
