#pragma once
// Module-2 topology description for the simulated Transport.
//
// A directed rank pair maps to one route. Routes carry their own alpha-beta
// parameters and name the shared resources they consume (for example, a PCIe
// switch uplink). Transfers whose routes mention the same resource contend.

#include <vector>

#include "gclab/sim/cost_model.h"

namespace gclab::sim {

using SharedResourceId = int;

struct LinkPath {
  LinkParams link{};
  std::vector<SharedResourceId> shared_resources;
};

class Topology {
 public:
  explicit Topology(int nranks);

  int nranks() const { return nranks_; }

  const LinkPath& path(int src, int dst) const;
  void set_path(int src, int dst, LinkPath path);

  static Topology fully_connected(int nranks, LinkParams link);
  static Topology pcie_switch(int nranks, LinkParams device_link,
                              LinkParams shared_uplink);
  static Topology nvlink_mesh(int nranks, LinkParams nvlink);

 private:
  std::size_t index(int src, int dst) const;

  int nranks_;
  std::vector<LinkPath> paths_;
};

}  // namespace gclab::sim
