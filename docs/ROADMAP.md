# GPU Collective Communication & Performance Lab: Roadmap

Pinned versions for source reading: **NCCL `v2.32.6-1`**, **PyTorch `v2.15.0`**.
Every file/function reference below was checked with grep against those tags. Line numbers are
approximate anchors and will drift if you change versions.

---

## 0. Ground rules

### Execution labels (every task carries one)

| Label | Meaning | Where it runs |
|---|---|---|
| `CPU_ONLY` | No CUDA at all | Your Mac (M2, 8 cores, 8 GB) |
| `GPU_COMPILE` | Needs `nvcc` / NCCL headers, no GPU | Linux Docker container on your Mac (`nvidia/cuda:*-devel`, arm64). The CUDA toolkit has no macOS support, so a native Mac build is impossible |
| `SINGLE_GPU` | One physical GPU | Rented instance |
| `MULTI_GPU` | ≥2 GPUs, one node | Rented 2×L40S (PCIe) or 2×H100 (NVLink) |
| `MULTI_NODE` | ≥2 nodes | Out of scope for the 5–10 h budget (documented only) |

### Simulated vs. measured

- `mode=SIMULATED` rows come from the α-β model in `gclab::sim`. They show how *algorithms*
  behave relative to each other. They are not hardware numbers.
- `mode=MEASURED` rows come from CUDA events / nccl-tests on real GPUs.
- The CSV schema has one shared `mode` column, and plots never mix the two without labelling them.

### What genuinely requires a GPU (and why)

| Needs GPU | Why it cannot be simulated |
|---|---|
| H2D/D2H/D2D bandwidth, pinned vs pageable | Depends on PCIe gen/lanes, IOMMU, the driver's staging path |
| P2P availability (`cudaDeviceCanAccessPeer`) | Cloud VMs often disable P2P (ACS, vGPU, passthrough). **Do not assume L40S P2P works.** Detect it |
| Stream/event overlap, copy engines | Hardware scheduling of DMA engines vs SMs |
| NCCL transport selection (P2P/SHM/NET), channel count, LL/LL128/Simple behaviour | Chosen from the real topology at init |
| nsys timelines, kernel durations | Need the GPU |
| PyTorch NCCL backend, DDP overlap on CUDA streams | Gloo on CPU gives the same *logic* but different timing |

Everything else (algorithms, schedulers, harnesses, parsers, plots, source reading, failure-injection
logic) is `CPU_ONLY` or `GPU_COMPILE`.

---

## 1. Project architecture

```
                    ┌────────────────────────────────────────────────────────┐
  Layer 5           │ pytorch/  DDP · TP · bucketing · comm hooks (Gloo→NCCL) │
                    └───────────────▲────────────────────────────────────────┘
                                    │ torch.distributed → c10d → ProcessGroupNCCL → ncclAllReduce
                    ┌───────────────┴───────────┐   ┌──────────────────────────┐
  Layer 4           │ bench/ harness, sweeps,    │   │ analysis/ CSV→plots,     │
                    │ nccl-tests wrapper, nsys   │──▶│ regression gate          │
                    └───────────────▲───────────┘   └──────────────────────────┘
                                    │ same collectives, different transports
                    ┌───────────────┴────────────────────────────────────────┐
  Layer 2           │ collectives:  bcast · reduce · ring/tree allreduce ·    │
                    │               allgather · reducescatter  (Transport API) │
                    └───────▲─────────────────────────▲──────────────────────┘
                            │                         │
                ┌───────────┴──────────┐   ┌──────────┴─────────────────────┐
  Layer 1/2     │ sim::InProcTransport │   │ cuda::P2PTransport             │
  backends      │ threads + α-β clock  │   │ cudaMemcpyPeerAsync + kernels  │
                │ + topology model     │   │ streams/events                 │
                │ CPU_ONLY             │   │ GPU_COMPILE → MULTI_GPU        │
                └──────────────────────┘   └────────────────────────────────┘
  Layer 3: docs/nccl/  (annotated source walkthroughs; compare to our designs)
  Layer 6: faults/     (failure injection on sim, Gloo and NCCL, all with timeouts)
```

The main design decision is that collectives are written once, against `Transport`, then tested on
the simulator and run on GPUs. NCCL splits the same way: algorithm code in `src/device/*.h`, transports in
`src/transport/*.cc`.

## 2. Target repository layout

```
gclab/
├── CMakeLists.txt                 # GCLAB_ENABLE_CUDA=OFF by default
├── include/gclab/
│   ├── common/   error.h log.h  (cuda_check.h, nccl_check.h  ← M3)
│   ├── sim/      cost_model.h channel.h transport.h world.h inproc_transport.h
│   │             collectives.h  (topology.h, async requests ← M2)
│   └── cuda/     (M3–M4) device_info.h p2p_transport.h reduce_kernels.cuh timer.h
├── src/{common,sim,cuda}/
├── tests/        step-by-step unit tests (CPU) + gpu/ correctness tests (label gpu)
├── bench/
│   ├── sim_*_bench.cc             # SIMULATED sweeps
│   ├── cuda/  memcpy_bw.cu p2p_bw.cu stream_overlap.cu ring_allreduce.cu   (M3–M4)
│   └── nccl/  nccl_harness.cc     # own NCCL timing w/ P50/P95/P99           (M6)
├── pytorch/      ddp_toy.py tp_mlp.py bucket_sweep.py comm_hooks.py         (M7)
├── faults/       inject_*.py / *.cc + expected-signature docs               (M8)
├── analysis/     schema.py plot_*.py regress.py                             (M2, M6)
├── scripts/
│   ├── setup_local.sh  docker_compile.sh                                    (CPU side)
│   ├── gpu_bootstrap.sh  gpu_capability_check.sh  gpu_session_A.sh  gpu_session_B.sh
│   └── fetch_results.sh                                                     (rsync back)
├── docker/Dockerfile.cuda-dev     # nvcc + libnccl-dev for GPU_COMPILE on the Mac
├── results/<date>_<host>_<gpu>/   # raw CSV/JSON, nsys reps, NCCL_DEBUG logs, env.json
└── docs/  ROADMAP.md modules/ nccl/ pytorch/ experiments/
```

---

## 3. Dependency-aware module roadmap

```
 M1 rank runtime + α-β ──▶ M2 allreduce family + topology sim ──┬──▶ M4 CUDA collectives ──┐
                                   │                              │                          │
 M3 CUDA fundamentals (code) ──────┼──────────────────────────────┘                          ▼
                                   ├──▶ M5 NCCL internals (reading) ──▶ M6 bench framework ──▶ GPU SESSION A
                                   └──▶ M7 PyTorch on Gloo ───────────────────────────────────▶ GPU SESSION B
                                         M8 debugging (sim+Gloo now, NCCL bits in A/B)
                                         M9 advanced project ──▶ optional SESSION C
```

| Mod | Topic | Pri | Label (dev → validate) | CPU hrs | GPU hrs |
|---|---|---|---|---|---|
| M1 | Rank runtime, Transport, α-β clock, Broadcast/Reduce | P0 | CPU_ONLY | 8–10 | 0 |
| M2 | Ring/Tree AllReduce, AllGather, ReduceScatter, chunk pipelining, topology model (PCIe switch vs NVLink), algbw/busbw | P0 | CPU_ONLY | 12–14 | 0 |
| M3 | CUDA fundamentals: memory hierarchy, pinned vs pageable, H2D/D2H/D2D, P2P, streams/events, topology probe | P0 | GPU_COMPILE → SINGLE/MULTI_GPU | 8 | 1.0 (A) |
| M4 | CUDA `P2PTransport` + reduction kernel, ring AllReduce on 2 GPUs, compare vs NCCL | P0 | GPU_COMPILE → MULTI_GPU | 10 | 1.0 (A) |
| M5 | NCCL internals: init → topology → channels → algo/proto → enqueue → kernel/proxy | P0 | CPU_ONLY (+ NCCL_DEBUG logs from A) | 12 | 0 (shares A) |
| M6 | Benchmark framework: NCCL harness, nccl-tests driver, matrix runner, schema, plots, regression gate | P0 | CPU_ONLY → MULTI_GPU | 10 | 1.5 (A) + 0.5 (B) |
| M7 | PyTorch: DDP, bucketing, comm hooks, TP, trace c10d → NCCL, one optimization | P0 | CPU_ONLY (Gloo) → MULTI_GPU | 12 | 1.5 (B) |
| M8 | Debugging & reliability, failure injection | P1 (diagnosis skills P0) | CPU_ONLY → MULTI_GPU | 8 | 0.5 (B) |
| M9 | Advanced optimization project (pick 1 of 3) | P2 | mixed | 10+ | 1–2 (C) |

**GPU budget: about 6 h baseline (A ≈ 3.5 h, B ≈ 2.5 h), about 8 h with Session C, 10 h if you add an H100
NVLink comparison (≈1.5 h).** Hours are wall-clock *instance* hours, so 2×L40S for 1 h counts as 1
billable instance-hour. Check your provider's current price; don't trust a number from me.

CPU share: about 100–110 h of local work vs 6–10 h of GPU time, roughly 92% CPU.

---

## 4. 14-day intensive plan (≈8–9 h/day)

| Day | Focus | Label | Gate / output |
|---|---|---|---|
| 1 | M1 steps 1–3: cost model, Channel, InProcTransport, read CUDA Programming Guide ch. on memory/streams | CPU_ONLY | step1–3 tests green, TSAN clean |
| 2 | M1 steps 4–5: broadcast/reduce, bench, write-up. Start M2 reading (Thakur et al. 2005, Patarasuk & Yuan 2009) | CPU_ONLY | step4 green, `m1.csv` + crossover explanation |
| 3 | M2: async isend/irecv, ring ReduceScatter + AllGather → ring AllReduce, chunking | CPU_ONLY | correctness for p=1..16, odd counts |
| 4 | M2: tree AllReduce (double binary tree), topology model, algbw/busbw, plots (`analysis/`) | CPU_ONLY | ring-vs-tree crossover plot (SIMULATED) |
| 5 | M3: Docker `GPU_COMPILE` env; write `memcpy_bw.cu`, `p2p_bw.cu`, `stream_overlap.cu`, `gpu_capability_check.sh` | GPU_COMPILE | compiles in container; expected outputs documented |
| 6 | M4: `cuda::P2PTransport`, reduce kernel, ring AllReduce reusing M2 schedule; GPU correctness tests | GPU_COMPILE | compiles; CPU-mock tests of the schedule pass |
| 7 | M5: NCCL source: init + topology + rings/trees + transport selection; write `docs/nccl/01-init.md` | CPU_ONLY | annotated call graph, verified against source |
| 8 | M5: NCCL AllReduce trace (enqueue → plan → kernel → primitives → proxy). M6: NCCL harness + nccl-tests runner + schema | CPU_ONLY / GPU_COMPILE | `docs/nccl/02-allreduce-trace.md`; harness compiles |
| 9 | M6 finish + **readiness gate A** (checklist §10). Dry-run `gpu_session_A.sh` in container with `--dry-run` | CPU_ONLY | every command and output path printed, nothing left to edit |
| 10 | **GPU SESSION A (~3.5 h)**, then local analysis | MULTI_GPU | results synced, instance **deleted** |
| 11 | Analysis of A; M7 on Gloo: DDP toy, bucket sweep, comm hook, TP MLP; trace c10d → NCCL | CPU_ONLY | Gloo results + `docs/pytorch/trace.md` |
| 12 | M8 failure injection on sim + Gloo; prepare **readiness gate B** | CPU_ONLY | fault matrix with expected signatures |
| 13 | **GPU SESSION B (~2.5 h)**: DDP/NCCL, buckets, overlap nsys, NCCL faults | MULTI_GPU | results synced, instance deleted |
| 14 | Analysis, write-up, README results section, interview drill. Optional Session C (M9) | CPU_ONLY | final report |

---

## 5. Tools & environment

**Local (Mac, CPU_ONLY)**
- Apple clang 14 (C++17), CMake ≥ 3.20 (you have 4.4), Python **3.12** venv via `uv` or Homebrew.
  Your system Python is 3.14, and PyTorch CPU wheels may lag new Python versions, so pin 3.12.
- `pip install torch numpy pandas matplotlib` (CPU wheel, Gloo backend included).
- ThreadSanitizer/ASan via `-DGCLAB_SANITIZER=thread|address`.
- Docker Desktop (installed, daemon not running right now) with `nvidia/cuda:<ver>-devel-ubuntu24.04` (arm64 tags
  exist) + `libnccl-dev` for `GPU_COMPILE`. Code compiled there targets Linux; on the GPU box you
  rebuild natively for x86_64, which takes seconds.
- Nsight Systems GUI for macOS to *open* `.nsys-rep` files locally (capture happens on the GPU box).
- Local NCCL/PyTorch source checkouts at the pinned tags (`third_party/` or outside repo, git-ignored).

**GPU instance (Ubuntu 22.04/24.04)**: driver + CUDA toolkit matching the image, NCCL from the CUDA
repo or built from `v2.32.6-1`, `nccl-tests`, `nsys` CLI, PyTorch CUDA wheel, `numactl`, `hwloc`
(`lstopo`). One `scripts/gpu_bootstrap.sh` installs or verifies all of it and writes `env.json`
(driver, CUDA, NCCL, torch versions, `nvidia-smi topo -m`, `lspci -tv`).

---

## 6. Core implementation tasks

| # | Task | Mod | Pri | Label |
|---|---|---|---|---|
| T1 | α-β cost model, unit-correct | M1 | P0 | CPU_ONLY |
| T2 | Thread-safe tag-matched Channel with deadline + abort | M1 | P0 | CPU_ONLY |
| T3 | InProcTransport with single-port full-duplex clock | M1 | P0 | CPU_ONLY |
| T4 | Linear + binomial Broadcast, linear Reduce | M1 | P0 | CPU_ONLY |
| T5 | Binomial Reduce | M1 | P1 | CPU_ONLY |
| T6 | Non-blocking isend/irecv + wait | M2 | P0 | CPU_ONLY |
| T7 | Ring ReduceScatter, ring AllGather, ring AllReduce (= RS + AG) | M2 | P0 | CPU_ONLY |
| T8 | Chunked/pipelined ring; chunk-size sweep | M2 | P0 | CPU_ONLY |
| T9 | Tree AllReduce (binomial reduce+bcast, then double binary tree) | M2 | P1 | CPU_ONLY |
| T10 | Topology model: per-link α/β, shared links (PCIe switch), NVLink mesh | M2 | P1 | CPU_ONLY |
| T11 | algbw/busbw calculator matching nccl-tests formulas | M2 | P0 | CPU_ONLY |
| T12 | `gpu_capability_check`: devices, CC, P2P matrix, `nvidia-smi topo -m`, NUMA | M3 | P0 | GPU_COMPILE → MULTI_GPU |
| T13 | memcpy bandwidth: pageable vs pinned, H2D/D2H/D2D, size sweep, CUDA events | M3 | P0 | GPU_COMPILE → SINGLE_GPU |
| T14 | P2P bandwidth/latency with/without peer access; bidirectional | M3 | P0 | GPU_COMPILE → MULTI_GPU |
| T15 | Stream/event overlap: copy ∥ compute, false dependencies via default stream | M3 | P0 | GPU_COMPILE → SINGLE_GPU |
| T16 | `cuda::P2PTransport` + vectorized reduce kernel; ring AllReduce on 2 GPUs | M4 | P0 | GPU_COMPILE → MULTI_GPU |
| T17 | NCCL harness: warmup, CUDA-event timing, P50/P95/P99, CPU-side enqueue time | M6 | P0 | GPU_COMPILE → MULTI_GPU |
| T18 | Matrix runner (nccl-tests + harness) → CSV/JSON with env metadata | M6 | P0 | CPU_ONLY → MULTI_GPU |
| T19 | Plots + regression gate (`analysis/regress.py`, tolerance bands) | M6 | P0 | CPU_ONLY |
| T20 | DDP toy + bucket sweep + overlap measurement (Gloo, then NCCL) | M7 | P0 | CPU_ONLY → MULTI_GPU |
| T21 | Comm hook (bf16 compression) or bucket autotune; measure step time | M7 | P0 | CPU_ONLY → MULTI_GPU |
| T22 | TP MLP (column/row parallel) showing AllReduce vs AllGather/RS use | M7 | P1 | CPU_ONLY → MULTI_GPU |
| T23 | Fault injection suite with timeouts and expected log signatures | M8 | P1 | CPU_ONLY → MULTI_GPU |

---

## 7. Benchmark experiment matrix

Sizes: 1 KiB to 1 GiB, ×2 steps (21 points). Smaller set for slow configs: 1K, 8K, 64K, 512K, 4M, 32M, 256M, 1G.

| ID | Experiment | Variables | Tool | Label | Session |
|---|---|---|---|---|---|
| E1 | Sim algorithm crossover | algo∈{linear,binomial,ring,tree}, p∈{2..16}, α/β presets (PCIe-ish, NVLink-ish) | sim bench | CPU_ONLY | none |
| E2 | Host↔device bandwidth | pageable/pinned, H2D/D2H, size | memcpy_bw | SINGLE_GPU | A |
| E3 | GPU↔GPU path | peer access on/off, uni/bi-dir, size | p2p_bw | MULTI_GPU | A |
| E4 | Stream overlap | 1 vs N streams, copy+kernel | stream_overlap + nsys | SINGLE_GPU | A |
| E5 | Own ring vs NCCL | size; our ring vs `ncclAllReduce` | ring_allreduce + harness | MULTI_GPU | A |
| E6 | NCCL baseline | coll∈{allreduce,allgather,reducescatter,broadcast,reduce} | nccl-tests `-b 1K -e 1G -f 2 -n 50 -w 10` | MULTI_GPU | A |
| E7 | Algorithm/protocol | `NCCL_ALGO`∈{Ring,Tree}, `NCCL_PROTO`∈{LL,LL128,Simple}. *Record which are accepted. Tree/LL128 support depends on version/topology/arch, so check `NCCL_DEBUG=INFO`* | nccl-tests | MULTI_GPU | A |
| E8 | Transport path | default vs `NCCL_P2P_DISABLE=1` (forces SHM) vs `NCCL_SHM_DISABLE=1`, `NCCL_P2P_LEVEL` | nccl-tests + NCCL_DEBUG | MULTI_GPU | A |
| E9 | Channels | `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` ∈ {1,2,4,8,…} | nccl-tests | MULTI_GPU | A |
| E10 | Latency distribution / CPU overhead | P50/P95/P99, enqueue time, blocking vs async (`cudaStreamSynchronize` per op vs batched) | harness | MULTI_GPU | A |
| E11 | DDP bucket size | `bucket_cap_mb` ∈ {1,5,25,100}, model size | torch | MULTI_GPU | B |
| E12 | Overlap | `no_sync`/manual vs DDP overlap; nsys timeline | torch + nsys | MULTI_GPU | B |
| E13 | Comm hook | default vs bf16 compress vs your optimization | torch | MULTI_GPU | B |
| E14 | Faults (NCCL) | rank mismatch, order mismatch, timeout, P2P disabled, straggler | faults/ | MULTI_GPU | B |
| E15 | Regression gate | re-run E6 subset against stored baseline | regress.py | MULTI_GPU | B |

Every row of output carries: `mode, git_sha, host, gpu_model, driver, cuda, nccl, torch, env_vars, timestamp`.

---

## 8. Source-reading paths (verified at NCCL v2.32.6-1 / PyTorch v2.15.0)

### 8.1 NCCL: communicator initialization (M5, P0)
| Step | Location |
|---|---|
| Public API | `src/init.cc:3063` `ncclCommInitRank` → `ncclCommInitRankDev` (:2967) → async job `ncclCommInitRankFunc` (:2221) |
| Bootstrap (out-of-band rank exchange) | `src/bootstrap.cc:754` `bootstrapInit` |
| Main init | `src/init.cc:1219` `initTransportsRank` |
| Topology discovery | `src/graph/topo.cc:1994` `ncclTopoGetSystem`; XML from sysfs in `src/graph/xml.cc` (`ncclTopoGetXmlFromSys` :604, `ncclTopoDumpXmlToFile` :299; env `NCCL_TOPO_DUMP_FILE`) |
| Paths & P2P decision | `src/graph/paths.cc` `ncclTopoComputePaths` (:734), `ncclTopoCheckP2p` (:320), `ncclTopoTrimSystem` (:880), `ncclTopoComputeP2pChannels` (:991) |
| Graph search (rings/trees/channels) | `src/graph/search.cc` `ncclTopoSearchInit` (:39), `ncclTopoCompute` (:1245) |
| Ring/tree wiring | `src/graph/connect.cc` `ncclTopoPreset` (:22), `ncclTopoPostset` (:395); `src/graph/rings.cc:29` `ncclBuildRings`; `src/graph/trees.cc` |
| Transport selection & connect | `src/transport.cc:125` `ncclTransportP2pSetup`; `src/transport/generic.cc` `ncclTransportRingConnect` (:14), `ncclTransportTreeConnect` (:49); per-transport `canConnect`: `transport/p2p.cc:130` `p2pCanConnect`, `transport/shm.cc:61` `shmCanConnect`, `transport/net.cc` |
| Proxy thread creation | `src/proxy.cc:2134` `ncclProxyCreate` |
| Tuning/cost model | `src/tuning/cost_model.cc` (`ncclTuningCostModelInit` :342). Note: `src/include/graph.h:219` still *declares* `ncclTopoTuneModel`, but no definition exists in this tag. Older blog posts that cite `tuning.cc`/`ncclTopoTuneModel` describe older NCCL. Tuner plugin ABI and the algo/proto IDs (`NCCL_ALGO_TREE=0`, `RING=1`; `NCCL_PROTO_LL=0`, `LL128=1`, `SIMPLE=2`) are in `src/include/plugin/nccl_tuner.h` |

### 8.2 NCCL: one AllReduce from API to GPU (M5, P0)
Functions below are verified to exist. **Confirming the exact call order and conditions between them is
your M5 task:** step through it with grep / a debugger on the GPU box, and do not take my arrows as fact.
```
ncclAllReduce                         src/collectives.cc:206
 └ ncclAllReduceConfigImpl → ncclEnqueueCheck        src/enqueue/enqueue.cc:3570
     └ taskAppend (:3417) → collTaskAppend (:2832)    (task queued on comm; group semantics)
ncclGroupEnd / implicit group end      src/group.cc:175 → ncclGroupEndInternal (:1052) → groupLaunch (:1044)
 └ ncclPrepareTasks                    src/enqueue/enqueue.cc:432   (also src/enqueue/task_prep/*.cc)
     └ ncclGetAlgoInfo (:2220)         algo × proto × nChannels selection
 └ ncclLaunchPrepare (:1755) → scheduleCollTasksToPlan (:658)    (src/enqueue/task_sched/*.cc)
 └ uploadWork (:1400) → ncclLaunchKernel (:1947)
 └ hostStreamPlanTask (:1593) → ncclProxyStart  src/proxy.cc:1018 (network/proxy ops)
GPU side:  src/device/all_reduce.h  runRing (:14), runTreeUpDown (:86), runTreeSplit (:146)
           src/device/primitives.h + prims_simple.h / prims_ll.h / prims_ll128.h
Proxy:     src/proxy.cc  ncclProxyProgress (:956) → progressOps (:801)
```
Newer code to be aware of (do not over-invest, P2): `src/ce_coll.cc` (copy-engine collectives),
`src/device/symmetric/`, `src/transport/nvls.cc` (NVLink SHARP; needs NVSwitch, not on L40S).

### 8.3 PyTorch: collective to NCCL (M7, P0)
```
torch/distributed/distributed_c10d.py:4061      all_reduce(...)
 └ dispatcher op c10d::allreduce_               torch/csrc/distributed/c10d/Ops.cpp:25 (schema), ~:212 impl macro
   └ ProcessGroupNCCL::allreduce                torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:4594
     └ allreduce_impl (:4566) → collective(...) (:3755 and overloads) → ncclAllReduce (:4580)
   Work / async handle:   Work.hpp, ProcessGroupNCCL.hpp (WorkNCCL), NCCLUtils.hpp
```
DDP:
```
torch/nn/parallel/distributed.py                 bucket_cap_mb, Reducer construction
torch/csrc/distributed/c10d/reducer.cpp
   compute_bucket_assignment_by_size  :2385      bucket formation (reverse param order)
   autograd_hook                      :697       fires per grad
   mark_variable_ready                :923  → mark_variable_ready_dense :358
   all_reduce_bucket                  :1002      launches the collective (or comm hook)
   finalize_backward                  :1799      waits on Work, copies back
torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py   (allreduce/fp16/bf16 hooks)
torch/distributed/tensor/parallel/{api.py, style.py}            ColwiseParallel / RowwiseParallel
```

---

## 9. Interview questions by module

**M1 (runtime, α-β)**
1. Derive the cost of linear vs binomial broadcast under α-β. When does linear win, and why does real hardware
   make that less likely than your simulator says? (Hint: per-message CPU overhead, the "o" in LogP.)
2. Your Channel uses a condition variable. Walk through a lost-wakeup bug and how the predicate loop prevents it.
3. Why compute an absolute deadline instead of passing a relative timeout to each wait?
4. NCCL has no message tags. What does it rely on instead, and what failure mode does that create?
5. What is n½ = α·β and why does it matter for choosing a protocol?

**M2 (algorithms)**
1. Prove ring AllReduce moves 2(p−1)/p · n bytes per rank, and why that is bandwidth-optimal.
2. Define algbw and busbw. Derive the busbw factor for AllReduce, AllGather, ReduceScatter, Broadcast.
3. Why does ring latency scale O(p) and tree O(log p)? Why did NCCL add double binary trees?
4. How does chunking/pipelining change the cost of a ring? What limits the chunk size from below?
5. Two flows share one PCIe switch uplink. How does that change your model and the best algorithm?

**M3 (CUDA fundamentals)**
1. Why is pinned memory faster for H2D, and what does the driver do for pageable copies?
2. Explain the legacy default stream's implicit synchronization and how per-thread default stream changes it.
3. What does `cudaEventRecord` timestamp, and why can't you time async work with `std::chrono` without syncing?
4. What determines whether two GPUs can do P2P? Why might it be off in a cloud VM?
5. Copy engines vs SMs for data movement: tradeoffs?

**M4 (CUDA collectives)**
1. Your ring on 2 GPUs: where are the syncs, and could any be replaced by events?
2. Why does NCCL run collectives as a persistent kernel that copies data, instead of issuing `cudaMemcpyPeerAsync` per step?
3. How do you vectorize the reduction kernel and what bounds its throughput?

**M5 (NCCL internals)**
1. Trace `ncclCommInitRank`: what is exchanged during bootstrap and why is it out-of-band?
2. How does NCCL decide P2P vs SHM vs NET between two GPUs? Where in the source?
3. What is a channel? How do channels map to SMs/CTAs and to links?
4. Explain Simple vs LL vs LL128: what each trades, and why LL uses 8-byte stores with a flag.
5. What does the proxy thread do, and why does NVLink/PCIe P2P mostly not need it?
6. How is the algorithm/protocol chosen for a given size? How can a tuner plugin override it?

**M6 (benchmarking)**
1. Why warm up? What do the first iterations of an NCCL collective pay for?
2. P99 is 3× P50 on small messages. List causes and how you'd tell which one it is.
3. How do you measure CPU overhead of enqueue separately from GPU time?
4. How do you set a regression threshold that does not flap on noisy cloud hardware?

**M7 (PyTorch)**
1. Why does DDP bucket gradients in reverse order of parameters?
2. How does DDP overlap communication with backward? What breaks the overlap?
3. Where does the NCCL call run relative to the compute stream, and how are they synchronized?
4. Column- vs row-parallel linear layers: which collective does each need, forward and backward?
5. When does gradient compression help on PCIe, and when does it hurt?

**M8 (debugging)**
1. A job hangs at step 1. Give your first five commands and what each would tell you.
2. How would you detect an unexpected transport fallback from logs alone?
3. One rank is consistently 20% slower. How do you prove whether it's compute, comm, or host?
4. What does PyTorch's NCCL watchdog do, and what is the flight recorder?

---

## 10. GPU readiness gates

**Gate A (before Session A).** All must be true:
- [ ] `ctest -L cpu_only` green, TSAN + ASan builds green
- [ ] All `.cu`/NCCL code compiles in `docker/Dockerfile.cuda-dev`
- [ ] Reviewed: M2 schedule code reused by M4 has CPU-mock tests
- [ ] `scripts/gpu_bootstrap.sh`, `gpu_capability_check.sh`, `gpu_session_A.sh` exist, are idempotent, and `--dry-run` prints every command
- [ ] Every experiment has a per-command `timeout`, and the whole session has a hard wall-clock budget
- [ ] Expected outputs + failure conditions written in `docs/experiments/session_A.md` (e.g. "if P2P matrix shows 0, skip E3 peer-on rows; record it")
- [ ] Results land in `results/<stamp>/` with `env.json`. `scripts/fetch_results.sh` tested against a local dir
- [ ] Teardown step documented (provider CLI command to delete the instance)

**Gate B**: same, plus M7 Gloo versions green and the fault matrix (M8) written with expected signatures.

**Session rules:** smoke test first (capability check + one 1 MiB allreduce). If something needs a real
code change, save the logs, `fetch_results.sh`, **delete the instance**, and debug locally.

---

## 11. Three advanced optimization projects (M9, P2)

1. **PCIe-aware NCCL tuner plugin.** Use E6–E9 data to build a measured cost table for your exact
   2×L40S topology, implement a tuner plugin against `src/include/plugin/nccl_tuner.h` that picks
   algo/proto/nChannels per size, and show the busbw/latency difference vs the default tuner, with a regression
   test. This uses a production extension point for real.
2. **DDP bucket autotuner + overlap-aware comm hook.** Profile per-bucket ready times from the
   Reducer (nsys + NVTX), choose bucket boundaries to maximize overlap, optionally combine with bf16
   compression, and report step time, comm time, and exposed (non-overlapped) comm.
3. **Copy-engine vs SM-driven collectives on PCIe.** Implement AllGather/ReduceScatter two ways in
   your `cuda::` backend (DMA via `cudaMemcpyPeerAsync` vs a copy kernel), measure SM occupancy cost
   and overlap with a concurrent GEMM, and compare with NCCL's own CE path (`src/ce_coll.cc`,
   read it first to see what it actually supports at this tag).
