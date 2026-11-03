#include "adaptive/allocation/adaptive_sampler.h"

#include <algorithm>
#include <cmath>

#include "util/check.h"

namespace aegis {

AdaptiveSampler::AdaptiveSampler(const GameTree& tree, const MCCFR& solver, AllocationMode mode, PriorityParams pp)
    : tree_(tree), solver_(solver), mode_(mode), pp_(pp), sumW_(tree.num_infosets(), 0.0),
      regret_stats_(tree.num_infosets()), cost_(tree.num_infosets()), visits_(tree.num_infosets(), 0),
      edge_prio_(tree.children.size()) {
  AEGIS_CHECK(pp_.lambda >= 0.0 && pp_.lambda < 1.0);
  for (int I = 0; I < tree.num_infosets(); ++I) regret_stats_[I].resize(tree.infosets[I].num_actions);
  if (mode_ == AllocationMode::kReach) {
    pp_.alpha = 1.0;
    pp_.beta = pp_.gamma = pp_.delta = 0.0;
  }
}

double AdaptiveSampler::reach_estimate(int I) const {
  const int p = tree_.infosets[I].player;
  return traversals_[p] > 0 ? sumW_[I] / static_cast<double>(traversals_[p]) : 0.0;
}

double AdaptiveSampler::uncertainty(int I) const {
  const auto& st = regret_stats_[I];
  double u = 0.0;
  for (const auto& s : st) u += s.std_error();
  return st.empty() ? 0.0 : u / static_cast<double>(st.size());
}

double AdaptiveSampler::sensitivity(int I) const {
  const auto& info = tree_.infosets[I];
  const auto& R = solver_.cumulative_regret();
  double g = 0.0;
  for (int a = 0; a < info.num_actions; ++a) g += std::max(0.0, R[info.offset + a]);
  const int64_t T = std::max<int64_t>(1, traversals_[info.player]);
  return g / static_cast<double>(T);
}

double AdaptiveSampler::cost(int I) const { return cost_[I].set ? cost_[I].value : 1.0; }

double AdaptiveSampler::score(int I) const {
  const double e = pp_.eps;
  double s = 1.0;
  if (pp_.alpha != 0.0) s *= std::pow(e + reach_estimate(I), pp_.alpha);
  if (pp_.beta != 0.0) s *= std::pow(e + uncertainty(I), pp_.beta);
  if (pp_.gamma != 0.0) s *= std::pow(e + sensitivity(I), pp_.gamma);
  if (pp_.delta != 0.0) s /= std::pow(e + cost(I), pp_.delta);
  return s;
}

double AdaptiveSampler::infoset_priority(int I) { return mode_ == AllocationMode::kExternal ? 0.0 : score(I); }

void AdaptiveSampler::sampling_distribution(int node, int, std::span<const double> p, std::span<double> q) {
  const int c = static_cast<int>(p.size());
  if (mode_ == AllocationMode::kExternal || pp_.lambda == 0.0) {
    for (int k = 0; k < c; ++k) q[k] = p[k];
    return;
  }
  const int b = tree_.child_begin[node];
  double psi[64];
  double total = 0.0;
  // Unvisited edges are optimistic: they get the largest priority seen at
  // this node (or 1 if none), so new regions are explored.
  double best = 0.0;
  bool any = false;
  for (int k = 0; k < c; ++k)
    if (edge_prio_[b + k].set) {
      best = std::max(best, edge_prio_[b + k].value);
      any = true;
    }
  if (!any) best = 1.0;
  for (int k = 0; k < c; ++k) {
    if (p[k] <= 0.0) {
      psi[k] = 0.0;
      continue;
    }
    if (mode_ == AllocationMode::kRandom) {
      psi[k] = 1.0;
    } else {
      const double P = edge_prio_[b + k].set ? edge_prio_[b + k].value : best;
      psi[k] = std::pow(P + pp_.prio_floor, pp_.tau);
    }
    total += psi[k];
  }
  for (int k = 0; k < c; ++k) q[k] = (1.0 - pp_.lambda) * p[k] + pp_.lambda * (total > 0.0 ? psi[k] / total : p[k]);
}

void AdaptiveSampler::on_traverser_visit(int I, int, double W, double, std::span<const double> vals, double v,
                                         int64_t cost) {
  visits_[I]++;
  sumW_[I] += W;
  auto& st = regret_stats_[I];
  for (size_t a = 0; a < vals.size(); ++a) st[a].add(W * (vals[a] - v));
  cost_[I].add(static_cast<double>(cost), 0.1);
}

void AdaptiveSampler::on_edge_backup(int node, int k, double priority, int64_t) {
  edge_prio_[tree_.child_begin[node] + k].add(priority, pp_.edge_rate);
}

void AdaptiveSampler::begin_traversal(int traverser) { traversals_[traverser]++; }
void AdaptiveSampler::end_traversal(int) {}

}  // namespace aegis
