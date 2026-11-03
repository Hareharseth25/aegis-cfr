#include "solvers/subgame_resolver/resolver.h"

#include <algorithm>
#include <cmath>

#include "util/check.h"
#include "util/timer.h"

namespace aegis::poker {

void reach_at(const SubgameContext& ctx, const VectorPolicy& policy, int node, std::vector<double> out[2]) {
  ctx.tree().node(node);  // bounds check
  for (int p = 0; p < 2; ++p) {
    // Walk the path from the root, multiplying the acting player's action
    // probabilities and masking hands blocked by dealt cards.
    std::vector<int> path;
    for (int n = node; n > 0; n = ctx.tree().nodes[n].parent) path.push_back(n);
    std::reverse(path.begin(), path.end());
    std::vector<double> r = ctx.hands(p).root_weight;
    int cur = 0;
    for (int n : path) {
      const auto& pn = ctx.tree().nodes[cur];
      if (pn.type == PNodeType::kDecision && pn.player == p) {
        const int k = n - pn.first_child;
        const int np = ctx.n(p);
        const double* s = policy.strategy[pn.decision_index].data() + static_cast<size_t>(k) * np;
        for (int i = 0; i < np; ++i) r[i] *= s[i];
      } else if (pn.type == PNodeType::kChance) {
        for (int i = 0; i < ctx.n(p); ++i)
          if (!ctx.valid(n, p, i)) r[i] = 0.0;
      }
      cur = n;
    }
    out[p] = std::move(r);
  }
}

std::vector<double> opponent_constraint(const SubgameContext& ctx, const VectorPolicy& policy, int root, int opp,
                                        bool cbv) {
  std::vector<double> reach[2], out[2];
  reach_at(ctx, policy, root, reach);
  ctx.subtree_values(policy, root, reach, cbv ? opp : -1, out);
  return out[opp];
}

std::vector<double> best_alternative_constraint(const SubgameContext& ctx, const VectorPolicy& policy, int d,
                                                const std::vector<int>& allowed, int opp) {
  const auto& nd = ctx.tree().node(d);
  AEGIS_CHECK_MSG(nd.type == PNodeType::kDecision && nd.player == opp, "constraint node must be an opponent decision");
  AEGIS_CHECK(!allowed.empty());
  std::vector<double> best(ctx.n(opp), -1e300);
  for (int c : allowed) {
    AEGIS_CHECK(ctx.tree().nodes[c].parent == d);
    const auto v = opponent_constraint(ctx, policy, c, opp, true);
    for (int i = 0; i < ctx.n(opp); ++i) best[i] = std::max(best[i], v[i]);
  }
  return best;
}

void splice_policy(const PublicTree& t, VectorPolicy& dst, const VectorPolicy& src, int root, int player) {
  std::vector<int> stack{root};
  while (!stack.empty()) {
    const int n = stack.back();
    stack.pop_back();
    const auto& nd = t.nodes[n];
    if (nd.type == PNodeType::kDecision && (player < 0 || nd.player == player)) {
      AEGIS_CHECK_MSG(!src.strategy[nd.decision_index].empty(), "source policy does not cover the subtree");
      dst.strategy[nd.decision_index] = src.strategy[nd.decision_index];
    }
    for (int k = 0; k < nd.num_children; ++k) stack.push_back(nd.first_child + k);
  }
}

std::vector<int> street_roots(const PublicTree& t, int street) {
  std::vector<int> out;
  for (int n = 0; n < t.num_nodes(); ++n) {
    const auto& nd = t.nodes[n];
    if (nd.type != PNodeType::kDecision || nd.street != street) continue;
    if (n == 0 || t.nodes[nd.parent].type == PNodeType::kChance) out.push_back(n);
  }
  return out;
}

bool action_in_abstraction(const ActionLabel& a, const StreetSizes& sizes, bool facing_bet) {
  switch (a.kind) {
    case ActionKind::kFold:
    case ActionKind::kCheck:
    case ActionKind::kCall:
    case ActionKind::kDeal:
      return true;
    case ActionKind::kAllIn:
      if (sizes.allow_all_in) return true;
      break;
    default:
      break;
  }
  const auto& list = facing_bet ? sizes.raise_fractions : sizes.bet_fractions;
  for (double f : list)
    if (a.nominal >= 0 && std::abs(a.nominal - f) < 1e-9) return true;
  return false;
}

VectorResolveResult resolve_subtree(const SubgameContext& ctx, const VectorPolicy& base, int root,
                                    const VectorResolveOptions& o) {
  Stopwatch sw;
  AEGIS_CHECK(o.us == 0 || o.us == 1);
  const int opp = 1 - o.us;
  VectorResolveResult res;
  reach_at(ctx, base, root, res.root_reach);
  VectorCFRConfig c;
  c.cfr = o.cfr;
  c.root = root;
  c.leaf = o.leaf;
  c.root_reach[0] = res.root_reach[0];
  c.root_reach[1] = res.root_reach[1];
  if (o.method != ResolveMethod::kNaive) {
    res.constraint = o.constraint.empty() ? opponent_constraint(ctx, base, root, opp, o.use_cbv) : o.constraint;
    AEGIS_CHECK(static_cast<int>(res.constraint.size()) == ctx.n(opp));
    c.gadget.enabled = true;
    c.gadget.opponent = opp;
    c.gadget.mode = o.method == ResolveMethod::kMaxMargin ? GadgetConfig::Mode::kMaxMargin
                                                           : GadgetConfig::Mode::kResolve;
    c.gadget.constraint = res.constraint;
  }
  VectorCFR solver(ctx, c);
  solver.iterate(o.iterations);
  const auto solved = solver.average_policy();
  res.policy = base;
  splice_policy(ctx.tree(), res.policy, solved, root, o.replace_opponent ? -1 : o.us);
  if (!res.constraint.empty()) {
    const auto now = opponent_constraint(ctx, res.policy, root, opp, true);
    res.margin.resize(now.size());
    for (size_t i = 0; i < now.size(); ++i) {
      res.margin[i] = res.constraint[i] - now[i];
      const double viol = std::max(0.0, -res.margin[i]);
      res.worst_violation = std::max(res.worst_violation, viol);
      res.total_violation += viol;
    }
  }
  res.seconds = sw.seconds();
  return res;
}

}  // namespace aegis::poker
