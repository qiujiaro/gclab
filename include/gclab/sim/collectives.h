#pragma once
// Collectives built purely on Transport::send/recv.
//
// Module 1: Broadcast and Reduce (rooted collectives).
// Module 2: AllReduce (ring, tree), AllGather, ReduceScatter.
//
// Contract shared by every function here (mirrors NCCL):
//   - Every rank in the world calls the same collective, in the same order,
//     with the same count/bytes and root.
//   - On return, this rank's part of the result is complete.

#include <cstddef>

#include "gclab/sim/transport.h"

namespace gclab::sim {

enum class ReduceOp { Sum, Max };

// Each collective uses its own tag so that a mismatched call sequence produces
// a timeout/CommError instead of silently pairing the wrong messages.
namespace tags {
inline constexpr int kBroadcast = 100;
inline constexpr int kReduce = 200;
}  // namespace tags

// Root sends the full buffer to every other rank, one after another.
// Simulated cost: alpha + (p-1) * n/beta. (Provided as a worked example.)
void broadcast_linear(Transport& tp, void* buf, std::size_t bytes, int root);

// Binomial tree: in round k, every rank that already has the data sends it to
// one rank that does not. ceil(log2 p) rounds.
// Simulated cost for p a power of two: log2(p) * (alpha + n/beta).
void broadcast_binomial(Transport& tp, void* buf, std::size_t bytes, int root);

// Every rank contributes `count` floats; root ends with the element-wise
// reduction in `recvbuf`. recvbuf is only written on root (may be null
// elsewhere). sendbuf is never modified.
void reduce_linear(Transport& tp, const float* sendbuf, float* recvbuf, std::size_t count,
                   ReduceOp op, int root);

// Binomial-tree reduce (the mirror image of broadcast_binomial). P1 stretch.
void reduce_binomial(Transport& tp, const float* sendbuf, float* recvbuf, std::size_t count,
                     ReduceOp op, int root);

}  // namespace gclab::sim
