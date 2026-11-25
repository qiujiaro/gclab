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

struct ElementChunk {
  std::size_t offset = 0;
  std::size_t count = 0;
};

std::vector<ElementChunk> element_chunks(std::size_t count, std::size_t chunk_bytes) {
  if (chunk_bytes != 0 &&
      (chunk_bytes < sizeof(float) || chunk_bytes % sizeof(float) != 0)) {
    throw CommError("chunk_bytes must be zero or a positive multiple of sizeof(float)");
  }
  if (count == 0) return {{0, 0}};
  if (chunk_bytes == 0) return {{0, count}};

  const std::size_t chunk_count = chunk_bytes / sizeof(float);
  std::vector<ElementChunk> chunks;
  chunks.reserve(count / chunk_count + (count % chunk_count != 0 ? 1 : 0));
  for (std::size_t offset = 0; offset < count; offset += chunk_count) {
    chunks.push_back({offset, std::min(chunk_count, count - offset)});
  }
  return chunks;
}

int highest_power_of_two_below(int value) {
  int power = 1;
  while (power <= (value - 1) / 2) power *= 2;
  return value > 1 ? power : 0;
}

struct BinaryTreePeers {
  int parent = -1;
  int child0 = -1;
  int child1 = -1;
};

int tree_rank_from_position(int position, int nranks, int tree_index) {
  return tree_index == 0 ? position : nranks - 1 - position;
}

BinaryTreePeers complementary_binary_tree_peers(int rank, int nranks, int tree_index) {
  const int position = tree_index == 0 ? rank : nranks - 1 - rank;
  BinaryTreePeers peers;
  if (position != 0) {
    peers.parent = tree_rank_from_position((position - 1) / 2, nranks, tree_index);
  }

  const int child0_position = 2 * position + 1;
  const int child1_position = 2 * position + 2;
  if (child0_position < nranks) {
    peers.child0 = tree_rank_from_position(child0_position, nranks, tree_index);
  }
  if (child1_position < nranks) {
    peers.child1 = tree_rank_from_position(child1_position, nranks, tree_index);
  }
  return peers;
}

void allreduce_binary_tree_slice(Transport& tp, const float* sendbuf, float* recvbuf,
                                 std::size_t count, ReduceOp op, int tree_index,
                                 int reduce_tag, int broadcast_tag) {
  const BinaryTreePeers peers =
      complementary_binary_tree_peers(tp.rank(), tp.size(), tree_index);
  const std::size_t bytes = count * sizeof(float);
  std::vector<float> accumulator(count);
  std::memcpy(accumulator.data(), sendbuf, bytes);
  std::vector<float> scratch(count);

  for (int child : {peers.child0, peers.child1}) {
    if (child < 0) continue;
    tp.recv(child, scratch.data(), bytes, reduce_tag);
    combine(accumulator.data(), scratch.data(), count, op);
  }
  if (peers.parent >= 0) {
    tp.send(peers.parent, accumulator.data(), bytes, reduce_tag);
    tp.recv(peers.parent, accumulator.data(), bytes, broadcast_tag);
  }
  for (int child : {peers.child0, peers.child1}) {
    if (child >= 0) tp.send(child, accumulator.data(), bytes, broadcast_tag);
  }

  std::memcpy(recvbuf, accumulator.data(), bytes);
}

}  // namespace

BlockRange block_for_rank(std::size_t total_count, int nranks, int rank) {
  if (nranks <= 0) {
    throw CommError("nranks must be positive");
  }

  if (rank < 0 || rank >= nranks) {
    throw CommError("rank out of range");
  }
  const auto n = static_cast<std::size_t>(nranks);
  const auto r = static_cast<std::size_t>(rank);

  const std::size_t base = total_count / n;
  const std::size_t remainder = total_count % n;

  BlockRange block;
  block.count = base + (r < remainder ? 1 : 0);
  block.offset = r * base + std::min(r, remainder);
  return block;
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
  const int rank = tp.rank();
  const int p = tp.size();
  if (p <= 0) throw CommError("reduce_scatter requires a positive world size");
  if (rank < 0 || rank >= p) throw CommError("reduce_scatter rank out of range");

  const BlockRange local_block = block_for_rank(count, p, rank);
  if (count != 0 && sendbuf == nullptr) {
    throw CommError("null reduce_scatter send buffer");
  }
  if (local_block.count != 0 && recvbuf == nullptr) {
    throw CommError("null reduce_scatter receive buffer");
  }
  // Validate the chunk size even for p=1, where the ring loop is empty.
  (void)element_chunks(local_block.count, chunk_bytes);
  if (count == 0) return;

  std::vector<float> working(count);
  std::memcpy(working.data(), sendbuf, count * sizeof(float));

  const int left = (rank - 1 + p) % p;
  const int right = (rank + 1) % p;
  for (int step = 0; step < p - 1; ++step) {
    // This rotation leaves reduced block r on rank r after p-1 steps.
    const int send_block_rank = (rank - step - 1 + p) % p;
    const int recv_block_rank = (rank - step - 2 + p) % p;
    const BlockRange send_block = block_for_rank(count, p, send_block_rank);
    const BlockRange recv_block = block_for_rank(count, p, recv_block_rank);
    std::vector<float> scratch(recv_block.count);
    const int tag = tags::kReduceScatterRing + step;

    const std::vector<ElementChunk> recv_chunks =
        element_chunks(recv_block.count, chunk_bytes);
    const std::vector<ElementChunk> send_chunks =
        element_chunks(send_block.count, chunk_bytes);
    std::vector<RequestPtr> requests;
    requests.reserve(recv_chunks.size() + send_chunks.size());

    for (const ElementChunk& chunk : recv_chunks) {
      void* ptr = chunk.count == 0 ? nullptr : scratch.data() + chunk.offset;
      requests.push_back(tp.irecv(left, ptr, chunk.count * sizeof(float), tag));
    }
    for (const ElementChunk& chunk : send_chunks) {
      const void* ptr = chunk.count == 0
                            ? nullptr
                            : working.data() + send_block.offset + chunk.offset;
      requests.push_back(tp.isend(right, ptr, chunk.count * sizeof(float), tag));
    }
    tp.wait_all(requests);

    if (recv_block.count != 0) {
      combine(working.data() + recv_block.offset, scratch.data(), recv_block.count, op);
    }
  }

  if (local_block.count != 0) {
    std::memcpy(recvbuf, working.data() + local_block.offset,
                local_block.count * sizeof(float));
  }
}

void allgather_ring(Transport& tp, const float* sendbuf, float* recvbuf,
                    std::size_t count, std::size_t chunk_bytes) {
  const int rank = tp.rank();
  const int p = tp.size();
  if (p <= 0) throw CommError("allgather requires a positive world size");
  if (rank < 0 || rank >= p) throw CommError("allgather rank out of range");

  const BlockRange local_block = block_for_rank(count, p, rank);
  if (local_block.count != 0 && sendbuf == nullptr) {
    throw CommError("null allgather send buffer");
  }
  if (count != 0 && recvbuf == nullptr) {
    throw CommError("null allgather receive buffer");
  }
  (void)element_chunks(local_block.count, chunk_bytes);
  if (count == 0) return;

  // Seed the final output with this rank's block. memmove also permits an
  // in-place call whose sendbuf already points at this range in recvbuf.
  if (local_block.count != 0) {
    std::memmove(recvbuf + local_block.offset, sendbuf,
                 local_block.count * sizeof(float));
  }

  const int left = (rank - 1 + p) % p;
  const int right = (rank + 1) % p;
  for (int step = 0; step < p - 1; ++step) {
    const int send_block_rank = (rank - step + p) % p;
    const int recv_block_rank = (rank - step - 1 + p) % p;
    const BlockRange send_block = block_for_rank(count, p, send_block_rank);
    const BlockRange recv_block = block_for_rank(count, p, recv_block_rank);
    const int tag = tags::kAllGatherRing + step;

    const std::vector<ElementChunk> recv_chunks =
        element_chunks(recv_block.count, chunk_bytes);
    const std::vector<ElementChunk> send_chunks =
        element_chunks(send_block.count, chunk_bytes);
    std::vector<RequestPtr> requests;
    requests.reserve(recv_chunks.size() + send_chunks.size());

    for (const ElementChunk& chunk : recv_chunks) {
      void* ptr = chunk.count == 0
                      ? nullptr
                      : recvbuf + recv_block.offset + chunk.offset;
      requests.push_back(tp.irecv(left, ptr, chunk.count * sizeof(float), tag));
    }
    for (const ElementChunk& chunk : send_chunks) {
      const void* ptr = chunk.count == 0
                            ? nullptr
                            : recvbuf + send_block.offset + chunk.offset;
      requests.push_back(tp.isend(right, ptr, chunk.count * sizeof(float), tag));
    }
    tp.wait_all(requests);
  }
}

void allreduce_ring(Transport& tp, const float* sendbuf, float* recvbuf,
                    std::size_t count, ReduceOp op, std::size_t chunk_bytes) {
  if (count != 0 && sendbuf == nullptr) {
    throw CommError("null ring allreduce send buffer");
  }
  if (count != 0 && recvbuf == nullptr) {
    throw CommError("null ring allreduce receive buffer");
  }
  const BlockRange local_block = block_for_rank(count, tp.size(), tp.rank());
  std::vector<float> reduced_block(local_block.count);
  reduce_scatter_ring(tp, sendbuf, reduced_block.data(), count, op, chunk_bytes);
  allgather_ring(tp, reduced_block.data(), recvbuf, count, chunk_bytes);
}

void allreduce_tree(Transport& tp, const float* sendbuf, float* recvbuf,
                    std::size_t count, ReduceOp op) {
  if (count != 0 && sendbuf == nullptr) {
    throw CommError("null tree allreduce send buffer");
  }
  if (count != 0 && recvbuf == nullptr) {
    throw CommError("null tree allreduce receive buffer");
  }
  if (count == 0) return;

  constexpr int root = 0;
  reduce_binomial(tp, sendbuf, recvbuf, count, op, root);
  broadcast_binomial(tp, recvbuf, count * sizeof(float), root);
}

void allreduce_double_binary_tree(Transport& tp, const float* sendbuf, float* recvbuf,
                                  std::size_t count, ReduceOp op,
                                  std::size_t chunk_bytes) {
  if (tp.size() <= 0) {
    throw CommError("double-tree allreduce requires a positive world size");
  }
  if (tp.rank() < 0 || tp.rank() >= tp.size()) {
    throw CommError("double-tree allreduce rank out of range");
  }
  if (count != 0 && sendbuf == nullptr) {
    throw CommError("null double-tree allreduce send buffer");
  }
  if (count != 0 && recvbuf == nullptr) {
    throw CommError("null double-tree allreduce receive buffer");
  }
  // Validate once up front; each half is split independently below.
  (void)element_chunks(count, chunk_bytes);
  if (count == 0) return;

  const std::size_t first_count = count / 2 + count % 2;
  const std::size_t second_count = count - first_count;
  const std::vector<ElementChunk> first_chunks = element_chunks(first_count, chunk_bytes);
  const std::vector<ElementChunk> second_chunks =
      second_count == 0 ? std::vector<ElementChunk>{}
                        : element_chunks(second_count, chunk_bytes);
  const std::size_t waves = std::max(first_chunks.size(), second_chunks.size());
  for (std::size_t wave = 0; wave < waves; ++wave) {
    if (wave < first_chunks.size()) {
      const ElementChunk chunk = first_chunks[wave];
      allreduce_binary_tree_slice(tp, sendbuf + chunk.offset, recvbuf + chunk.offset,
                                  chunk.count, op, 0, tags::kAllReduceDoubleTree,
                                  tags::kAllReduceDoubleTree + 1);
    }
    if (wave < second_chunks.size()) {
      const ElementChunk chunk = second_chunks[wave];
      allreduce_binary_tree_slice(
          tp, sendbuf + first_count + chunk.offset,
          recvbuf + first_count + chunk.offset, chunk.count, op, 1,
          tags::kAllReduceDoubleTree + 2, tags::kAllReduceDoubleTree + 3);
    }
  }
}

}  // namespace gclab::sim
