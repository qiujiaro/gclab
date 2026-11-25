#include "gclab/sim/topology.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "gclab/common/error.h"

namespace gclab::sim {

namespace {

void validate_link(const LinkParams& link, const char* name) {
  if (!std::isfinite(link.alpha_us) || link.alpha_us < 0.0) {
    throw CommError(std::string(name) + " alpha_us must be finite and non-negative");
  }
  if (!std::isfinite(link.beta_GBps) || link.beta_GBps <= 0.0) {
    throw CommError(std::string(name) + " beta_GBps must be finite and positive");
  }
}

LinkPath self_path() {
  return {{0.0, std::numeric_limits<double>::max()}, {}};
}

}  // namespace

Topology::Topology(int nranks) : nranks_(nranks) {
  if (nranks <= 0) throw CommError("topology nranks must be positive");
  const auto n = static_cast<std::size_t>(nranks);
  if (n > std::numeric_limits<std::size_t>::max() / n) {
    throw CommError("topology route matrix size overflow");
  }
  paths_.resize(n * n);
}

std::size_t Topology::index(int src, int dst) const {
  if (src < 0 || src >= nranks_) {
    throw CommError("topology source rank out of range: " + std::to_string(src));
  }
  if (dst < 0 || dst >= nranks_) {
    throw CommError("topology destination rank out of range: " + std::to_string(dst));
  }
  return static_cast<std::size_t>(src) * static_cast<std::size_t>(nranks_) +
         static_cast<std::size_t>(dst);
}

const LinkPath& Topology::path(int src, int dst) const {
  return paths_.at(index(src, dst));
}

void Topology::set_path(int src, int dst, LinkPath path_value) {
  validate_link(path_value.link, "topology path");
  std::sort(path_value.shared_resources.begin(), path_value.shared_resources.end());
  for (SharedResourceId resource : path_value.shared_resources) {
    if (resource < 0) throw CommError("shared resource id must be non-negative");
  }
  if (std::adjacent_find(path_value.shared_resources.begin(),
                         path_value.shared_resources.end()) !=
      path_value.shared_resources.end()) {
    throw CommError("duplicate shared resource id in path");
  }
  paths_.at(index(src, dst)) = std::move(path_value);
}

Topology Topology::fully_connected(int nranks, LinkParams link) {
  validate_link(link, "fully-connected link");
  Topology topology(nranks);
  for (int src = 0; src < nranks; ++src) {
    for (int dst = 0; dst < nranks; ++dst) {
      topology.set_path(src, dst, src == dst ? self_path() : LinkPath{link, {}});
    }
  }
  return topology;
}

Topology Topology::pcie_switch(int nranks, LinkParams device_link,
                               LinkParams shared_uplink) {
  validate_link(device_link, "PCIe device link");
  validate_link(shared_uplink, "PCIe shared uplink");
  const LinkParams effective_link{
      2.0 * device_link.alpha_us + shared_uplink.alpha_us,
      std::min(device_link.beta_GBps, shared_uplink.beta_GBps)};
  validate_link(effective_link, "PCIe effective path");

  Topology topology(nranks);
  constexpr SharedResourceId kSharedPcieUplink = 0;
  for (int src = 0; src < nranks; ++src) {
    for (int dst = 0; dst < nranks; ++dst) {
      topology.set_path(src, dst,
                        src == dst
                            ? self_path()
                            : LinkPath{effective_link, {kSharedPcieUplink}});
    }
  }
  return topology;
}

Topology Topology::nvlink_mesh(int nranks, LinkParams nvlink) {
  validate_link(nvlink, "NVLink");
  return fully_connected(nranks, nvlink);
}

}  // namespace gclab::sim
