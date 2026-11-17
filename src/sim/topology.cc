#include "gclab/sim/topology.h"

#include <utility>

#include "gclab/common/error.h"

namespace gclab::sim {

Topology::Topology(int nranks)
    : nranks_(nranks), paths_(nranks > 0 ? static_cast<std::size_t>(nranks) * nranks : 0) {
  // TODO(M2/T10): reject non-positive rank counts without risking signed
  // overflow while sizing the route matrix.
}

std::size_t Topology::index(int src, int dst) const {
  // TODO(M2/T10): validate both endpoints and return the directed matrix
  // index. Decide whether a self-route is legal and test that decision.
  (void)src;
  (void)dst;
  throw NotImplemented("Topology::index");
}

const LinkPath& Topology::path(int src, int dst) const {
  // TODO(M2/T10): return the configured directed path.
  return paths_.at(index(src, dst));
}

void Topology::set_path(int src, int dst, LinkPath path_value) {
  // TODO(M2/T10): validate link parameters and shared-resource identifiers.
  paths_.at(index(src, dst)) = std::move(path_value);
}

Topology Topology::fully_connected(int nranks, LinkParams link) {
  // TODO(M2/T10): populate every directed rank pair with an independent link.
  (void)nranks;
  (void)link;
  throw NotImplemented("Topology::fully_connected");
}

Topology Topology::pcie_switch(int nranks, LinkParams device_link,
                               LinkParams shared_uplink) {
  // TODO(M2/T10): model per-device PCIe links plus a contended switch uplink.
  // Document which routes share each resource identifier.
  (void)nranks;
  (void)device_link;
  (void)shared_uplink;
  throw NotImplemented("Topology::pcie_switch");
}

Topology Topology::nvlink_mesh(int nranks, LinkParams nvlink) {
  // TODO(M2/T10): populate direct NVLink-like routes without a shared PCIe
  // uplink. Extend this factory if multi-hop NVLink routes are later needed.
  (void)nranks;
  (void)nvlink;
  throw NotImplemented("Topology::nvlink_mesh");
}

}  // namespace gclab::sim
