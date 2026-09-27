#include "gclab/sim/collectives.h"

#include <vector>

#include "gclab/common/error.h"

namespace gclab::sim {

void broadcast_linear(Transport& tp, void* buf, std::size_t bytes, int root) {
  const int me = tp.rank();
  const int p = tp.size();
  if (me == root) {
    for (int r = 0; r < p; ++r) {
      if (r != root) tp.send(r, buf, bytes, tags::kBroadcast);
    }
  } else {
    tp.recv(root, buf, bytes, tags::kBroadcast);
  }
}

void broadcast_binomial(Transport& tp, void* buf, std::size_t bytes, int root) {
  // TODO(step 4):
  //   Work in "virtual ranks" so the root is always 0:
  //       vrank = (rank - root + p) % p,   real = (vrank + root) % p
  //   Phase 1 (receive): a non-root rank receives exactly once, from the
  //   vrank you get by clearing its lowest set bit.
  //   Phase 2 (send): forward to vrank + 2^k for every power of two smaller
  //   than your own lowest set bit (root: every power of two < p), skipping
  //   targets >= p.
  //   Draw the p=8 tree on paper first. Then p=6 (not a power of two).
  //   Does the order of sends in phase 2 change the simulated time? Predict,
  //   then measure with the bench.
  (void)tp;
  (void)buf;
  (void)bytes;
  (void)root;
  throw NotImplemented("broadcast_binomial");
}

void reduce_linear(Transport& tp, const float* sendbuf, float* recvbuf, std::size_t count,
                   ReduceOp op, int root) {
  // TODO(step 4):
  //   Non-root: send sendbuf to root.
  //   Root: copy sendbuf into recvbuf, then for each other rank receive into
  //   a scratch buffer and combine element-wise with `op`.
  //   Questions: does receiving in rank order matter for the RESULT? For Sum
  //   on floats, is the result bit-identical across different receive
  //   orders? (This is a real issue: NCCL results can differ between ring
  //   and tree for the same input.)
  (void)tp;
  (void)sendbuf;
  (void)recvbuf;
  (void)count;
  (void)op;
  (void)root;
  throw NotImplemented("reduce_linear");
}

void reduce_binomial(Transport& tp, const float* sendbuf, float* recvbuf, std::size_t count,
                     ReduceOp op, int root) {
  // TODO(step 5, P1 stretch): reverse the broadcast tree. Leaves send first;
  // each inner node receives from its children, combines, then sends to its
  // parent.
  (void)tp;
  (void)sendbuf;
  (void)recvbuf;
  (void)count;
  (void)op;
  (void)root;
  throw NotImplemented("reduce_binomial");
}

}  // namespace gclab::sim
