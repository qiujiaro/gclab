#include "gclab/sim/collectives.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "gclab/common/error.h"

namespace gclab::sim {

namespace {

void validate_root(const Transport& tp, int root) {
  if (root < 0 || root >= tp.size()) {
    throw CommError("invalid collective root " + std::to_string(root));
  }
}

void validate_reduce_buffers(const Transport& tp, const float* sendbuf, float* recvbuf,
                             std::size_t count, int root) {
  validate_root(tp, root);
  if (count != 0 && sendbuf == nullptr) throw CommError("null reduce send buffer");
  if (count != 0 && tp.rank() == root && recvbuf == nullptr) {
    throw CommError("null reduce receive buffer at root");
  }
}

void combine(float* dst, const float* src, std::size_t count, ReduceOp op) {
  if (op == ReduceOp::Sum) {
    for (std::size_t i = 0; i < count; ++i) dst[i] += src[i];
  } else {
    for (std::size_t i = 0; i < count; ++i) dst[i] = std::max(dst[i], src[i]);
  }
}

int highest_power_of_two_below(int value) {
  int power = 1;
  while (power <= (value - 1) / 2) power *= 2;
  return value > 1 ? power : 0;
}

}  // namespace

BlockRange block_for_rank(std::size_t total_count, int nranks, int rank) {
  // TODO(M2/T7): validate nranks/rank and return an overflow-safe balanced
  // partition. Cover total_count < nranks and total_count == 0.
  (void)total_count;
  (void)nranks;
  (void)rank;
  throw NotImplemented("block_for_rank");
}

void broadcast_linear(Transport& tp, void* buf, std::size_t bytes, int root) {
  validate_root(tp, root);
  if (bytes != 0 && buf == nullptr) throw CommError("null broadcast buffer");
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
  validate_root(tp, root);
  if (bytes != 0 && buf == nullptr) throw CommError("null broadcast buffer");

  const int p = tp.size();
  const int vrank = (tp.rank() - root + p) % p;
  int subtree_span = p;

  if (vrank != 0) {
    // The least significant set bit is the span of this rank's subtree.
    subtree_span = vrank & -vrank;
    const int parent_vrank = vrank - subtree_span;
    tp.recv((parent_vrank + root) % p, buf, bytes, tags::kBroadcast);
  }

  // Large subtrees go first. With one send port this ordering is what lets
  // every tree level progress concurrently and gives log2(p) rounds.
  int offset = vrank == 0 ? highest_power_of_two_below(p) : subtree_span / 2;
  for (; offset > 0; offset /= 2) {
    const int child_vrank = vrank + offset;
    if (child_vrank < p) tp.send((child_vrank + root) % p, buf, bytes, tags::kBroadcast);
  }
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
  validate_reduce_buffers(tp, sendbuf, recvbuf, count, root);
  const std::size_t bytes = count * sizeof(float);
  if (tp.rank() != root) {
    tp.send(root, sendbuf, bytes, tags::kReduce);
    return;
  }

  if (bytes != 0) std::memcpy(recvbuf, sendbuf, bytes);
  std::vector<float> scratch(count);
  for (int rank = 0; rank < tp.size(); ++rank) {
    if (rank == root) continue;
    tp.recv(rank, scratch.data(), bytes, tags::kReduce);
    combine(recvbuf, scratch.data(), count, op);
  }
}

void reduce_binomial(Transport& tp, const float* sendbuf, float* recvbuf, std::size_t count,
                     ReduceOp op, int root) {
  // TODO(step 5, P1 stretch): reverse the broadcast tree. Leaves send first;
  // each inner node receives from its children, combines, then sends to its
  // parent.
  validate_reduce_buffers(tp, sendbuf, recvbuf, count, root);
  const int p = tp.size();
  const int vrank = (tp.rank() - root + p) % p;
  const int subtree_span = vrank == 0 ? p : (vrank & -vrank);
  const std::size_t bytes = count * sizeof(float);

  std::vector<float> accumulator(count);
  if (bytes != 0) std::memcpy(accumulator.data(), sendbuf, bytes);
  std::vector<float> scratch(count);

  // Reverse of broadcast's high-to-low sends: consume smaller, earlier-ready
  // child subtrees first so the receive port remains continuously occupied.
  for (int offset = 1; offset < subtree_span; offset *= 2) {
    const int child_vrank = vrank + offset;
    if (child_vrank >= p) break;
    tp.recv((child_vrank + root) % p, scratch.data(), bytes, tags::kReduce);
    combine(accumulator.data(), scratch.data(), count, op);
  }

  if (vrank == 0) {
    if (bytes != 0) std::memcpy(recvbuf, accumulator.data(), bytes);
  } else {
    const int parent_vrank = vrank - subtree_span;
    tp.send((parent_vrank + root) % p, accumulator.data(), bytes, tags::kReduce);
  }
}

void reduce_scatter_ring(Transport& tp, const float* sendbuf, float* recvbuf,
                         std::size_t count, ReduceOp op, std::size_t chunk_bytes) {
  // TODO(M2/T7,T8): implement p-1 ring steps, uneven blocks, and optional
  // chunk pipelining using irecv/isend/wait. Do not modify sendbuf.
  (void)tp;
  (void)sendbuf;
  (void)recvbuf;
  (void)count;
  (void)op;
  (void)chunk_bytes;
  throw NotImplemented("reduce_scatter_ring");
}

void allgather_ring(Transport& tp, const float* sendbuf, float* recvbuf,
                    std::size_t count, std::size_t chunk_bytes) {
  // TODO(M2/T7,T8): circulate uneven rank blocks for p-1 steps and support the
  // same chunking semantics as reduce_scatter_ring.
  (void)tp;
  (void)sendbuf;
  (void)recvbuf;
  (void)count;
  (void)chunk_bytes;
  throw NotImplemented("allgather_ring");
}

void allreduce_ring(Transport& tp, const float* sendbuf, float* recvbuf,
                    std::size_t count, ReduceOp op, std::size_t chunk_bytes) {
  // TODO(M2/T7,T8): compose ReduceScatter + AllGather without introducing an
  // extra full-size copy per rank.
  (void)tp;
  (void)sendbuf;
  (void)recvbuf;
  (void)count;
  (void)op;
  (void)chunk_bytes;
  throw NotImplemented("allreduce_ring");
}

void allreduce_tree(Transport& tp, const float* sendbuf, float* recvbuf,
                    std::size_t count, ReduceOp op) {
  // TODO(M2/T9): compose binomial Reduce and Broadcast, including p=1 and
  // zero-count inputs. Keep sendbuf unmodified.
  (void)tp;
  (void)sendbuf;
  (void)recvbuf;
  (void)count;
  (void)op;
  throw NotImplemented("allreduce_tree");
}

void allreduce_double_binary_tree(Transport& tp, const float* sendbuf, float* recvbuf,
                                  std::size_t count, ReduceOp op,
                                  std::size_t chunk_bytes) {
  // TODO(M2/T9): construct complementary trees, split data between them, and
  // pipeline reduce/broadcast chunks while preserving deterministic tags.
  (void)tp;
  (void)sendbuf;
  (void)recvbuf;
  (void)count;
  (void)op;
  (void)chunk_bytes;
  throw NotImplemented("allreduce_double_binary_tree");
}

}  // namespace gclab::sim
