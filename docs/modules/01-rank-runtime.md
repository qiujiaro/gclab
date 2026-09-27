# Module 1: Rank Runtime, Transport Abstraction, and the α-β Model

`P0 · CPU_ONLY · ~8–10 h · GPU cost: 0`

## 1. The problem

A collective is a contract among *p* processes ("ranks"). Each rank owns a buffer, and after the call
every rank holds some function of all buffers. Before thinking about GPUs you need three things:

1. **Ranks that run concurrently** and can block waiting for each other.
2. **A point-to-point primitive** (send/recv) that every collective is built from.
3. **A way to reason about time** that doesn't depend on how fast your laptop is.

Real systems split these the same way. In NCCL, `bootstrap.cc` gets ranks talking, transports
(`src/transport/p2p.cc`, `shm.cc`, `net.cc`) move bytes, and a cost model (`src/tuning/cost_model.cc`)
predicts time to pick an algorithm. You are building a small version of each.

## 2. Architecture

```
 World (one process)
 ┌───────────────────────────────────────────────────────────────┐
 │ thread r0          thread r1          thread r2               │
 │ InProcTransport    InProcTransport    InProcTransport         │
 │  clock, tx, rx      clock, tx, rx      clock, tx, rx          │
 │     │  ▲               │  ▲               │  ▲                │
 │     ▼  │               ▼  │               ▼  │                │
 │  Channel grid  ch[src][dst]  (N×N, mutex + condvar, tagged)   │
 │                                                               │
 │ failure: first exception → World::abort() → every Channel     │
 │          wakes waiters with AbortedError → threads join →     │
 │          root-cause exception rethrown to caller              │
 └───────────────────────────────────────────────────────────────┘
```

Two separate notions of time:
- **Wall time** (`RunResult::wall_time_us`): how long your Mac took. Mostly memcpy and thread wakeups.
- **Simulated time** (`now_us()`, `makespan_us`): what the α-β model says the network would take.
  Each message carries its sender's simulated start time, and the receiver works out when it lands. This
  is the same trick as Lamport clocks: causality flows with the message.

## 3. Data/control flow for one send/recv

```
 rank A: send(B, buf, n, tag)                 rank B: recv(A, buf, n, tag)
   start = max(clockA, txA)                     m = ch[A][B].pop(tag, timeout)  ── blocks (real time)
   txA = clockA = start + n/β                   check m.size == n
   copy buf → Message{tag, payload, start}      copy payload → buf
   ch[A][B].push(msg) ───────── real time ───▶  begin  = max(m.start + α, rxB)
   (returns immediately: eager)                 finish = begin + n/β
                                                rxB = finish; clockB = max(clockB, finish)
```

## 4. Algorithms and tradeoffs

**α-β (Hockney):** `T(n) = α + n/β`. Below n½ = α·β you are latency-bound, above it bandwidth-bound.
Every algorithm choice in NCCL (ring vs tree, LL vs Simple) is a bet about which side of n½ you are on.

**Port model:** each rank has one send port and one receive port (full duplex). Without the receive port,
a linear reduce would cost one message no matter how many senders there are. Counting the receive port
is what makes the model honest about congestion at the root.

**Linear broadcast:** the root sends p−1 times. Cost α + (p−1)·n/β.
**Binomial broadcast:** the number of ranks holding the data doubles each round. Cost ⌈log₂p⌉·(α + n/β) for p a power of two.

Before you run the benchmark, predict which one wins at 1 KiB and which at 1 MiB for p=8, α=5 µs,
β=25 GB/s. Then explain *why* the small-message result looks the way it does, and why real
hardware would push it the other way. (LogP's per-message overhead *o* is the missing term.)

**Eager vs rendezvous:** our `send` copies into the channel and returns (eager), so "both ranks send,
then both recv" works. With rendezvous (send waits for the matching recv) the same code deadlocks. MPI
implementations switch from eager to rendezvous above a size threshold, so code can work at small sizes
and hang at large ones. NCCL has neither: it uses fixed-size slots in ring buffers with credit-based flow control.
You will meet that in M5.

## 5. Implementation tasks

| Step | File | What | Tests |
|---|---|---|---|
| 1 | `src/sim/cost_model.cc` | `wire_time_us`, `transfer_time_us` | `test_step1_cost_model` |
| 2 | `src/sim/channel.cc` | `push`, `pop` (tag match, deadline, abort, spurious wakeups) | `test_step2_channel` (+TSAN) |
| 3 | `src/sim/inproc_transport.cc` | `send`, `recv` with port-model clock + validation | `test_step3_transport` |
| 4 | `src/sim/collectives.cc` | `broadcast_binomial`, `reduce_linear` | `test_step4_collectives` |
| 5 (P1) | `src/sim/collectives.cc` | `reduce_binomial` | `reduce_binomial_correct_p1_stretch` |

Provided (read them, since you will be asked about them): `World::run` failure propagation,
`Channel::abort`, `broadcast_linear`, logger, test harness, bench.

Rules for yourself:
- Don't change test expectations to make them pass. If you think a test is wrong, argue it in your notes and I'll review.
- Each TODO comment has a question in it. Answer those in `docs/modules/01-notes.md`. I'll review the answers along with the code.

## 6. Build & test

```bash
cmake -S . -B build && cmake --build build -j8
ctest --test-dir build --output-on-failure          # all steps
./build/tests/test_step2_channel                    # one step
./build/tests/test_step3_transport deadlock         # one test (substring filter)

# Concurrency bugs hide. Always re-run step 2/3 under TSAN:
cmake -S . -B build-tsan -DGCLAB_SANITIZER=thread -DCMAKE_BUILD_TYPE=Debug
cmake --build build-tsan -j8 && ctest --test-dir build-tsan
```

The output labels each test `[ TODO ]`, `[ FAIL ]` or `[ PASS ]`. Every test binary has a 60 s ctest timeout,
and every blocking test also checks its own elapsed time, so a hang shows up as a failure.

## 7. Correctness strategy

- **Exhaustive small cases:** p = 1..9 × every root × sizes {0, 1, 4096}. Most collective bugs are
  off-by-one in rank arithmetic, p not a power of two, root ≠ 0, or zero-byte messages.
- **Exact arithmetic:** reduce inputs are small integers, so float sums are exact regardless of order.
  (With real data they aren't. Note it, because it matters in M2 and M7.)
- **Analytic timing:** the simulated clock is deterministic, so cost tests compare against closed-form
  formulas to 1e-6. If a formula test fails while the data is correct, your send order or clock rule is off.
- **Failure paths are tested too:** size mismatch, bad peer, deadlock → bounded timeout, mismatched collective
  order, and one rank crashing all have tests.

## 8. Benchmarking

```bash
./build/bench/sim_rooted_bench --ranks 2,4,8,16 --alpha-us 5 --bw-GBps 25 > results/m1_sim.csv
```
Every row says `SIMULATED`. Things to do with it:
1. Find the crossover message size between linear and binomial broadcast for each p. Derive it on
   paper and check that it matches.
2. Re-run with NVLink-like (α≈2 µs, β≈150 GB/s) and PCIe-like (α≈8 µs, β≈20 GB/s) parameters. These are
   illustrative, not measurements; Session A will give you real ones. How does the crossover move?
3. Look at `wall_time_us` for 16 MiB. That's simulator overhead (memcpy of every payload). Why does it
   not matter for the conclusions?

Plotting comes in M2 (`analysis/`), once there are enough algorithms to compare.

## 9. Interview questions

1. Derive linear vs binomial broadcast cost under α-β with a single-port model. At what n do they cross for p=8?
2. Where can a lost wakeup happen in a condvar queue, and why does `wait_until` inside a predicate loop fix it?
3. `notify_one` vs `notify_all` in `Channel`: which is correct here and why? (Consider `abort()`.)
4. Your recv validates size. NCCL doesn't. What happens in NCCL if rank 0 calls AllReduce with
   count=1024 and rank 1 with count=512?
5. Why must `World::run` abort peers when one rank fails? Map it to `ncclCommAbort` and PyTorch's NCCL watchdog.
6. Is a binomial tree the best broadcast for very large messages? What would you do instead? (Preview of M2:
   pipelined chains / scatter+allgather.)

## 10. Relation to production

| Here | NCCL | PyTorch |
|---|---|---|
| `World` threads | one process per GPU (or per rank), launched by your launcher | `torchrun` |
| Channel grid | per-peer connectors with fixed-slot FIFOs in GPU/host memory (`src/transport/*.cc`, `src/include/device.h`) | n/a |
| Tags | none: ordering is implied by every rank issuing collectives in the same order on the same stream | c10d sequence numbers for debugging (flight recorder) |
| Size check | none (mismatch = UB / hang) | `TORCH_DISTRIBUTED_DEBUG=DETAIL` adds collective fingerprint checks |
| α-β clock | `src/tuning/cost_model.cc` latency/bandwidth tables per algo×proto | n/a |
| `abort()` | `ncclCommAbort` | watchdog thread in `ProcessGroupNCCL` |
