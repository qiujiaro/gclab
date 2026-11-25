#pragma once
// World: N simulated ranks, each running on its own std::thread, connected by
// an N x N grid of Channels. Owns startup, failure propagation and teardown.
//
// Roughly the job of `torchrun` + ncclCommInitRank + ncclCommAbort, collapsed
// into one process.

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gclab/sim/channel.h"
#include "gclab/sim/cost_model.h"
#include "gclab/sim/transport.h"

namespace gclab::sim {
class Topology;
}

namespace gclab::sim {

struct WorldConfig {
  int nranks = 2;
  LinkParams link{};  // homogeneous fully-connected network in Module 1
  // Optional heterogeneous route and shared-link model. When null, Module-1's
  // homogeneous `link` remains in effect.
  std::shared_ptr<const Topology> topology;
  std::chrono::milliseconds recv_timeout{2000};
};

struct RunResult {
  std::vector<double> final_clock_us;  // simulated clock of each rank at exit
  double makespan_us = 0.0;            // max over ranks
  double wall_time_us = 0.0;           // REAL host time for the whole run
};

class World {
 public:
  explicit World(WorldConfig cfg);

  // Run `fn` once per rank, each on its own thread, with that rank's
  // Transport. Blocks until every thread has exited.
  //
  // Failure handling: if any rank throws, the world is aborted so blocked
  // peers wake up with AbortedError instead of hanging, and after all threads
  // are joined the FIRST (root-cause) exception is rethrown.
  //
  // A World can be run again after a SUCCESSFUL run. After a failed run its
  // channels stay aborted; create a fresh World.
  RunResult run(const std::function<void(Transport&)>& fn);

  const WorldConfig& config() const { return cfg_; }
  Channel& channel(int src, int dst);
  // Atomically reserve every shared resource on a route. Returns the actual
  // start time after waiting for earlier users and marks the resources busy
  // for `duration_us`.
  double reserve_shared_resources(const std::vector<int>& resources,
                                  double earliest_start_us, double duration_us);
  void abort();

 private:
  WorldConfig cfg_;
  std::vector<std::unique_ptr<Channel>> channels_;  // index src * n + dst
  std::mutex shared_resource_mu_;
  std::unordered_map<int, double> shared_resource_free_us_;
};

}  // namespace gclab::sim
