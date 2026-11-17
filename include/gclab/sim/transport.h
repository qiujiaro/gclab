#pragma once
// Transport: the point-to-point interface every collective is written against.
//
// This is the seam of the whole project:
//
//     collectives (broadcast, ring allreduce, ...)
//            |  only ever call send/recv/now_us
//            v
//     Transport  <-- abstract
//       |-- InProcTransport   (Module 1-2, CPU threads + simulated clock)
//       |-- CudaP2PTransport  (Module 4, cudaMemcpyPeerAsync on real GPUs)
//       '-- MockTransport     (tests: record and replay traffic)
//
// Because the collective code never touches CUDA, it is tested exhaustively on
// a laptop and reused unchanged on a GPU. NCCL has the same split: the
// algorithm (ring/tree code in src/device/*.h) is separate from the transports
// (src/transport/p2p.cc, shm.cc, net.cc), joined by a "connector".
//
// Semantics (Module 1): blocking, eager.
//   send() returns once the payload has been handed to the transport; it does
//          NOT wait for the receiver. The caller may reuse `buf` immediately.
//   recv() blocks until a matching message from `src` arrives and copies it
//          into `buf`. The byte count must match exactly.
// Module 2 adds non-blocking isend/irecv for pipelining.

#include <cstddef>
#include <memory>
#include <vector>

namespace gclab::sim {

// Opaque completion handle returned by non-blocking operations. Concrete
// transports own the state needed to complete or fail the operation.
class Request {
 public:
  virtual ~Request() = default;
};

using RequestPtr = std::unique_ptr<Request>;

class Transport {
 public:
  virtual ~Transport() = default;

  virtual int rank() const = 0;
  virtual int size() const = 0;

  virtual void send(int dst, const void* buf, std::size_t bytes, int tag) = 0;
  virtual void recv(int src, void* buf, std::size_t bytes, int tag) = 0;

  // Module 2: non-blocking operations used by chunked collectives.
  //
  // Buffer lifetime contract:
  //   - isend: `buf` must remain readable until wait() completes.
  //   - irecv: `buf` must remain writable until wait() completes.
  // A Request belongs to the Transport that created it and may be waited once.
  virtual RequestPtr isend(int dst, const void* buf, std::size_t bytes, int tag) = 0;
  virtual RequestPtr irecv(int src, void* buf, std::size_t bytes, int tag) = 0;
  virtual void wait(Request& request) = 0;

  // Wait in input order and propagate the first operation error.
  virtual void wait_all(std::vector<RequestPtr>& requests) = 0;

  // Current SIMULATED time on this rank, in microseconds.
  virtual double now_us() const = 0;
};

}  // namespace gclab::sim
