# gclab: GPU Collective Communication & Performance Lab

A hands-on systems project for learning how GPU collective communication works, from α-β cost models
and hand-written ring/tree collectives up to NCCL internals and PyTorch DDP, with reproducible
benchmarks and failure-injection experiments.

Development is **CPU-first**: algorithms, schedulers, harnesses and analysis are built and tested on a
laptop. Rented GPUs (2×L40S PCIe, optionally 2×H100 NVLink) are used only in short sessions that run
prepared scripts. Every result is labelled `SIMULATED` or `MEASURED`.

## Status

| Module | Topic | State |
|---|---|---|
| M1 | Rank runtime, Transport, α-β model, Broadcast/Reduce | in progress |
| M2–M9 | see [docs/ROADMAP.md](docs/ROADMAP.md) | planned |

No hardware results yet.

## Quick start (CPU only)

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench/sim_rooted_bench --ranks 2,4,8 > m1_sim.csv
```

Requirements: C++17 compiler, CMake ≥ 3.20. No third-party dependencies.

## Layout

```
include/gclab/   public headers (common/, sim/; cuda/ arrives in M3)
src/             implementations
tests/           step-by-step unit tests, dependency-free harness
bench/           benchmark drivers (CSV to stdout)
docs/            roadmap, per-module guides
```

## Pinned references

NCCL `v2.32.6-1`, PyTorch `v2.15.0`: see [docs/ROADMAP.md §8](docs/ROADMAP.md).
