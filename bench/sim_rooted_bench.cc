// Simulated benchmark for rooted collectives (Module 1). CPU_ONLY.
//
// Prints CSV to stdout. Every row is labelled mode=SIMULATED: sim_time_us
// comes from the alpha-beta model, NOT from hardware. wall_time_us is how long
// your CPU took to run the simulation and is only useful for spotting
// simulator overhead.
//
//   ./sim_rooted_bench --ranks 2,4,8,16 --alpha-us 5 --bw-GBps 25 > m1.csv
//
// Columns:
//   mode,collective,algo,nranks,bytes,alpha_us,bw_GBps,sim_time_us,
//   algbw_GBps,wall_time_us
//
// algbw = bytes / sim_time. (busbw is introduced in Module 2, where it starts
// to mean something for AllReduce/AllGather/ReduceScatter.)

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "gclab/common/error.h"
#include "gclab/sim/collectives.h"
#include "gclab/sim/world.h"

using namespace gclab::sim;

namespace {

std::vector<long long> parse_list(const std::string& s) {
  std::vector<long long> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoll(item));
  return out;
}

struct Args {
  std::vector<long long> ranks{2, 4, 8, 16};
  double alpha_us = 5.0;
  double bw_GBps = 25.0;
  long long min_bytes = 1 << 10;   // 1 KiB
  long long max_bytes = 16 << 20;  // 16 MiB (simulator holds p copies in RAM)
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--ranks") a.ranks = parse_list(v);
    else if (k == "--alpha-us") a.alpha_us = std::stod(v);
    else if (k == "--bw-GBps") a.bw_GBps = std::stod(v);
    else if (k == "--min-bytes") a.min_bytes = std::stoll(v);
    else if (k == "--max-bytes") a.max_bytes = std::stoll(v);
    else {
      std::fprintf(stderr, "unknown flag %s\n", k.c_str());
      std::exit(2);
    }
  }
  return a;
}

using RankFn = std::function<void(Transport&, std::size_t bytes)>;

struct Case {
  const char* collective;
  const char* algo;
  RankFn fn;
};

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);

  const std::vector<Case> cases = {
      {"broadcast", "linear",
       [](Transport& tp, std::size_t n) {
         std::vector<char> buf(n);
         broadcast_linear(tp, buf.data(), n, 0);
       }},
      {"broadcast", "binomial",
       [](Transport& tp, std::size_t n) {
         std::vector<char> buf(n);
         broadcast_binomial(tp, buf.data(), n, 0);
       }},
      {"reduce", "linear",
       [](Transport& tp, std::size_t n) {
         const std::size_t count = n / sizeof(float);
         std::vector<float> s(count, 1.0f), r(count);
         reduce_linear(tp, s.data(), r.data(), count, ReduceOp::Sum, 0);
       }},
      {"reduce", "binomial",
       [](Transport& tp, std::size_t n) {
         const std::size_t count = n / sizeof(float);
         std::vector<float> s(count, 1.0f), r(count);
         reduce_binomial(tp, s.data(), r.data(), count, ReduceOp::Sum, 0);
       }},
  };

  std::printf("mode,collective,algo,nranks,bytes,alpha_us,bw_GBps,sim_time_us,algbw_GBps,wall_time_us\n");
  for (const auto& c : cases) {
    for (long long p : args.ranks) {
      for (long long n = args.min_bytes; n <= args.max_bytes; n *= 2) {
        WorldConfig cfg;
        cfg.nranks = static_cast<int>(p);
        cfg.link = {args.alpha_us, args.bw_GBps};
        World w(cfg);
        RunResult res;
        try {
          res = w.run([&](Transport& tp) { c.fn(tp, static_cast<std::size_t>(n)); });
        } catch (const gclab::NotImplemented& e) {
          std::fprintf(stderr, "skip %s/%s: %s\n", c.collective, c.algo, e.what());
          goto next_case;
        }
        {
          const double algbw = res.makespan_us > 0 ? n / (res.makespan_us * 1e3) : 0.0;
          std::printf("SIMULATED,%s,%s,%lld,%lld,%.3f,%.3f,%.3f,%.4f,%.1f\n", c.collective, c.algo, p, n,
                      args.alpha_us, args.bw_GBps, res.makespan_us, algbw, res.wall_time_us);
        }
      }
    }
  next_case:;
  }
  return 0;
}
