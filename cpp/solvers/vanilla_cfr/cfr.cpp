#include "solvers/vanilla_cfr/cfr.h"

#include "util/check.h"

namespace aegis {

TabularCFR::TabularCFR(const GameTree& tree, CFRConfig config)
    : tree_(tree), cfg_(config), regret_(tree.total_actions, 0.0), inst_regret_(tree.total_actions, 0.0),
      strategy_sum_(tree.total_actions, 0.0), sigma_(tree.total_actions, 0.0) {
  AEGIS_CHECK_MSG(tree.zero_sum, "CFR requires a two-player zero-sum game");
  for (const auto& I : tree_.infosets) regret_matching(&regret_[I.offset], &sigma_[I.offset], I.num_actions);
}

void TabularCFR::set_regrets(const std::vector<double>& r) {
  AEGIS_CHECK(static_cast<int>(r.size()) == tree_.total_actions);
  regret_ = r;
  for (const auto& I : tree_.infosets) regret_matching(&regret_[I.offset], &sigma_[I.offset], I.num_actions);
}

double TabularCFR::traverse(int h, int i, double reach0, double reach1, double reach_c) {
  ++nodes_touched_;
  const auto& t = tree_;
  switch (t.type[h]) {
    case NodeType::kTerminal:
      return t.utility[i][h];
    case NodeType::kChance: {
      const int b = t.child_begin[h], c = t.child_count[h];
      double v = 0.0;
      for (int k = 0; k < c; ++k) {
        const double p = t.chance_prob[b + k];
        v += p * traverse(t.children[b + k], i, reach0, reach1, reach_c * p);
      }
      return v;
    }
    case NodeType::kDecision:
      break;
  }
  // No player can reach this history: it cannot affect any regret or average.
  if (reach0 == 0.0 && reach1 == 0.0) return 0.0;

  const int p = t.player[h];
  const auto& I = t.infosets[t.infoset[h]];
  const int b = t.child_begin[h], n = I.num_actions;
  const double* sigma = &sigma_[I.offset];
  double child_v[64];
  AEGIS_CHECK(n <= 64);
  double v = 0.0;
  for (int a = 0; a < n; ++a) {
    const double r0 = p == 0 ? reach0 * sigma[a] : reach0;
    const double r1 = p == 1 ? reach1 * sigma[a] : reach1;
    child_v[a] = traverse(t.children[b + a], i, r0, r1, reach_c);
    v += sigma[a] * child_v[a];
  }
  if (p == i) {
    const double cf_reach = (p == 0 ? reach1 : reach0) * reach_c;  // opponent x chance
    const double own_reach = p == 0 ? reach0 : reach1;
    const double w = avg_w_;
    double* inst = &inst_regret_[I.offset];
    double* ssum = &strategy_sum_[I.offset];
    const double tol = cfg_.tie_tolerance;
    for (int a = 0; a < n; ++a) {
      inst[a] += cf_reach * tie_snapped(child_v[a], v, tol);
      ssum[a] += w * own_reach * sigma[a];
    }
  }
  return v;
}

void TabularCFR::traverse_both(int h, double reach0, double reach1, double reach_c, double out[2]) {
  ++nodes_touched_;
  const auto& t = tree_;
  if (t.type[h] == NodeType::kTerminal) {
    out[0] = t.utility[0][h];
    out[1] = t.utility[1][h];
    return;
  }
  const int b = t.child_begin[h], c = t.child_count[h];
  if (t.type[h] == NodeType::kChance) {
    out[0] = out[1] = 0.0;
    for (int k = 0; k < c; ++k) {
      const double p = t.chance_prob[b + k];
      double cv[2];
      traverse_both(t.children[b + k], reach0, reach1, reach_c * p, cv);
      out[0] += p * cv[0];
      out[1] += p * cv[1];
    }
    return;
  }
  if (reach0 == 0.0 && reach1 == 0.0) {
    out[0] = out[1] = 0.0;
    return;
  }
  const int p = t.player[h];
  const auto& I = t.infosets[t.infoset[h]];
  const int n = I.num_actions;
  const double* sigma = &sigma_[I.offset];
  double child_v[64][2];
  AEGIS_CHECK(n <= 64);
  out[0] = out[1] = 0.0;
  for (int a = 0; a < n; ++a) {
    const double r0 = p == 0 ? reach0 * sigma[a] : reach0;
    const double r1 = p == 1 ? reach1 * sigma[a] : reach1;
    traverse_both(t.children[b + a], r0, r1, reach_c, child_v[a]);
    out[0] += sigma[a] * child_v[a][0];
    out[1] += sigma[a] * child_v[a][1];
  }
  const double cf_reach = (p == 0 ? reach1 : reach0) * reach_c;
  const double own_reach = p == 0 ? reach0 : reach1;
  const double w = avg_w_;
  double* inst = &inst_regret_[I.offset];
  double* ssum = &strategy_sum_[I.offset];
  const double tol = cfg_.tie_tolerance;
  for (int a = 0; a < n; ++a) {
    inst[a] += cf_reach * tie_snapped(child_v[a][p], out[p], tol);
    ssum[a] += w * own_reach * sigma[a];
  }
}

void TabularCFR::finish_player(int i) {
  // Discount factors depend only on t: compute them once per iteration.
  const bool clip = cfg_.clips_regrets(), disc = cfg_.discounts();
  const double dp = cfg_.regret_discount(t_, true), dn = cfg_.regret_discount(t_, false);
  for (const auto& I : tree_.infosets) {
    if (i >= 0 && I.player != i) continue;
    for (int a = 0; a < I.num_actions; ++a) {
      double& R = regret_[I.offset + a];
      R += inst_regret_[I.offset + a];
      if (clip) {
        if (R < 0.0) R = 0.0;
      } else if (disc) {
        R *= R >= 0.0 ? dp : dn;
      }
      inst_regret_[I.offset + a] = 0.0;
    }
    regret_matching(&regret_[I.offset], &sigma_[I.offset], I.num_actions);
  }
}

void TabularCFR::iterate(int n) {
  for (int k = 0; k < n; ++k) {
    ++t_;
    avg_w_ = cfg_.average_weight(t_);
    if (cfg_.alternating) {
      for (int i = 0; i < 2; ++i) {
        traverse(0, i, 1.0, 1.0, 1.0);
        finish_player(i);
      }
    } else {
      double v[2];
      traverse_both(0, 1.0, 1.0, 1.0, v);
      finish_player(-1);
    }
  }
}

TabularPolicy TabularCFR::average_policy() const { return TabularPolicy::from_weights(tree_, strategy_sum_); }

TabularPolicy TabularCFR::current_policy() const {
  TabularPolicy p(tree_);
  p.probs = sigma_;
  return p;
}

}  // namespace aegis
