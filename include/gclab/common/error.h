#pragma once
// Error types shared across the project.
//
// Design note: the simulator uses exceptions because a failed rank must be able
// to unwind its thread cleanly. The CUDA/NCCL layers (Module 3+) will use
// status codes + CHECK macros instead, mirroring how NCCL itself returns
// ncclResult_t and never throws.

#include <stdexcept>
#include <string>

namespace gclab {

// Thrown by starter-code stubs. A test that hits this reports "TODO" instead
// of a confusing wrong answer.
struct NotImplemented : std::logic_error {
  explicit NotImplemented(const std::string& what)
      : std::logic_error("TODO not implemented: " + what) {}
};

// A blocking operation exceeded its deadline (e.g. a recv whose matching send
// never happened). This is the simulator's equivalent of NCCL's watchdog
// timeout in PyTorch (TORCH_NCCL_* timeouts).
struct TimeoutError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The world was aborted because some other rank failed. Analogous to
// ncclCommAbort() tearing down a communicator.
struct AbortedError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Protocol violation between ranks: size mismatch, bad peer, etc.
// In real NCCL, a count mismatch between ranks is *undefined behaviour* (often a
// hang or silent corruption) because NCCL does not exchange message headers.
// The simulator can detect it because every message carries its size.
struct CommError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace gclab
