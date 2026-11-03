#include "solvers/subgame_resolver/vector_cfr.h"

#include <algorithm>
#include <atomic>
#include <cmath>

#include "util/check.h"

namespace aegis::poker {

VectorCFR::VectorCFR(const SubgameContext& ctx, VectorCFRConfig cfg) : ctx_(ctx), tree_(ctx.tree()), cfg_(cfg) {
  AEGIS_CHECK_MSG(cfg_.root >= 0 && cfg_.root < tree_.num_nodes(), "invalid subtree root");
  for (int p = 0; p < 2; ++p) {
    if (cfg_.root_reach[p].empty()) {
      AEGIS_CHECK_MSG(cfg_.root == 0, "re-solving a subtree requires explicit root reach vectors");
      cfg_.root_reach[p] = ctx_.hands(p).root_weight;
    }
    AEGIS_CHECK_MSG(static_cast<int>(cfg_.root_reach[p].size()) == ctx_.n(p), "root reach has wrong size");
  }
  in_subtree_.assign(tree_.num_nodes(), 0);
  collect_leaves(cfg_.root, leaves_);
  for (uint8_t x : in_subtree_) subtree_nodes_ += x;
  const int D = tree_.num_decision_nodes();
  regret_.resize(D);
  sum_.resize(D);
  sigma_.resize(D);
  for (int n = 0; n < tree_.num_nodes(); ++n) {
    const auto& nd = tree_.nodes[n];
    if (nd.type != PNodeType::kDecision || !in_subtree_[n]) continue;
    const size_t sz = static_cast<size_t>(nd.num_children) * ctx_.n(nd.player);
    regret_[nd.decision_index].assign(sz, 0.0);
    sum_[nd.decision_index].assign(sz, 0.0);
    sigma_[nd.decision_index].assign(sz, 1.0 / nd.num_children);
  }
  for (int p = 0; p < 2; ++p) {
    reach_[p].assign(tree_.num_nodes(), {});
    leaf_cfv_[p].assign(tree_.num_nodes(), {});
  }
  cfv_.assign(tree_.num_nodes(), {});
  if (!leaves_.empty())
    AEGIS_CHECK_MSG(cfg_.leaf != nullptr, "depth-limited tree requires a LeafValueProvider");
  if (cfg_.gadget.enabled) {
    const int o = cfg_.gadget.opponent;
    AEGIS_CHECK(o == 0 || o == 1);
    AEGIS_CHECK_MSG(static_cast<int>(cfg_.gadget.constraint.size()) == ctx_.n(o),
                    "gadget constraint must have one value per opponent hand");
    g_regret_follow_.assign(ctx_.n(o), 0.0);
    g_regret_term_.assign(ctx_.n(o), 0.0);
    if (cfg_.gadget.mode == GadgetConfig::Mode::kResolve) {
      g_root_reach_.assign(ctx_.n(o), 0.5);
    } else {
      g_root_reach_.assign(ctx_.n(o), 1.0 / ctx_.n(o));
    }
    // Hands that are impossible at the root never enter.
    for (int i = 0; i < ctx_.n(o); ++i)
      if (!ctx_.valid(cfg_.root, o, i)) g_root_reach_[i] = 0.0;
  }
}

void VectorCFR::collect_leaves(int node, std::vector<int>& out) {
  in_subtree_[node] = 1;
  const auto& nd = tree_.nodes[node];
  if (nd.type == PNodeType::kLeaf) out.push_back(node);
  for (int k = 0; k < nd.num_children; ++k) collect_leaves(nd.first_child + k, out);
}

void VectorCFR::compute_sigma(int node) {
  const auto& nd = tree_.nodes[node];
  const int np = ctx_.n(nd.player);
  const int A = nd.num_children;
  const double* R = regret_[nd.decision_index].data();
  double* s = sigma_[nd.decision_index].data();
  for (int i = 0; i < np; ++i) {
    double total = 0.0;
    for (int a = 0; a < A; ++a) total += std::max(0.0, R[static_cast<size_t>(a) * np + i]);
    if (total > 0.0) {
      const double inv = 1.0 / total;
      for (int a = 0; a < A; ++a) s[static_cast<size_t>(a) * np + i] = std::max(0.0, R[static_cast<size_t>(a) * np + i]) * inv;
    } else {
      for (int a = 0; a < A; ++a) s[static_cast<size_t>(a) * np + i] = 1.0 / A;
    }
  }
}

void VectorCFR::forward(int node) {
  const auto& nd = tree_.nodes[node];
  if (nd.num_children == 0) return;
  if (nd.type == PNodeType::kChance) {
    auto task = [&](int k) {
      const int c = nd.first_child + k;
      for (int p = 0; p < 2; ++p) {
        const int np = ctx_.n(p);
        const uint8_t* v = ctx_.valid_mask(tree_.nodes[c].board_id, p);
        const auto& r = reach_[p][node];
        auto& rc = reach_[p][c];
        rc.resize(np);
        for (int i = 0; i < np; ++i) rc[i] = v[i] ? r[i] : 0.0;
      }
      forward(c);
    };
    if (ctx_.pool()) ctx_.pool()->parallel_for(nd.num_children, task);
    else
      for (int k = 0; k < nd.num_children; ++k) task(k);
    return;
  }
  const int actor = nd.player;
  const int na = ctx_.n(actor);
  const double* s = sigma_[nd.decision_index].data();
  for (int k = 0; k < nd.num_children; ++k) {
    const int c = nd.first_child + k;
    reach_[1 - actor][c] = reach_[1 - actor][node];
    auto& rc = reach_[actor][c];
    const auto& r = reach_[actor][node];
    rc.resize(na);
    const double* sk = s + static_cast<size_t>(k) * na;
    for (int i = 0; i < na; ++i) rc[i] = r[i] * sk[i];
    forward(c);
  }
}

void VectorCFR::backward(int node, int p) {
  const auto& nd = tree_.nodes[node];
  const int np = ctx_.n(p);
  const int o = 1 - p;
  auto& out = cfv_[node];
  out.resize(np);
  switch (nd.type) {
    case PNodeType::kFold:
      ctx_.fold_cfv(node, p, reach_[o][node].data(), out.data());
      return;
    case PNodeType::kShowdown:
      ctx_.showdown_cfv(node, p, reach_[o][node].data(), out.data());
      return;
    case PNodeType::kLeaf:
      out = leaf_cfv_[p][node];
      return;
    case PNodeType::kChance: {
      auto task = [&](int k) { backward(nd.first_child + k, p); };
      if (ctx_.pool()) ctx_.pool()->parallel_for(nd.num_children, task);
      else
        for (int k = 0; k < nd.num_children; ++k) task(k);
      const double w = ctx_.chance_weight(node);
      std::fill(out.begin(), out.end(), 0.0);
      for (int k = 0; k < nd.num_children; ++k) {
        const double* cc = cfv_[nd.first_child + k].data();
        for (int i = 0; i < np; ++i) out[i] += w * cc[i];
      }
      return;
    }
    case PNodeType::kDecision:
      break;
  }
  for (int k = 0; k < nd.num_children; ++k) backward(nd.first_child + k, p);
  std::fill(out.begin(), out.end(), 0.0);
  if (nd.player != p) {
    for (int k = 0; k < nd.num_children; ++k) {
      const double* cc = cfv_[nd.first_child + k].data();
      for (int i = 0; i < np; ++i) out[i] += cc[i];
    }
    return;
  }
  const int A = nd.num_children;
  double* R = regret_[nd.decision_index].data();
  double* S = sum_[nd.decision_index].data();
  const double* s = sigma_[nd.decision_index].data();
  const double* own = reach_[p][node].data();
  for (int k = 0; k < A; ++k) {
    const double* cc = cfv_[nd.first_child + k].data();
    const double* sk = s + static_cast<size_t>(k) * np;
    for (int i = 0; i < np; ++i) out[i] += sk[i] * cc[i];
  }
  const bool clip = cfg_.cfr.clips_regrets();
  const bool disc = cfg_.cfr.discounts();
  const double dp = dpos_, dn = dneg_, w = avg_w_;
  const double tol = cfg_.cfr.tie_tolerance;
  for (int k = 0; k < A; ++k) {
    const double* cc = cfv_[nd.first_child + k].data();
    double* Rk = R + static_cast<size_t>(k) * np;
    double* Sk = S + static_cast<size_t>(k) * np;
    const double* sk = s + static_cast<size_t>(k) * np;
    if (clip) {
      for (int i = 0; i < np; ++i) Rk[i] = std::max(0.0, Rk[i] + tie_snapped(cc[i], out[i], tol));
    } else if (disc) {
      for (int i = 0; i < np; ++i) {
        const double x = Rk[i] + tie_snapped(cc[i], out[i], tol);
        Rk[i] = x * (x >= 0.0 ? dp : dn);
      }
    } else {
      for (int i = 0; i < np; ++i) Rk[i] += tie_snapped(cc[i], out[i], tol);
    }
    for (int i = 0; i < np; ++i) Sk[i] += w * own[i] * sk[i];
  }
  compute_sigma(node);
}

void VectorCFR::update_gadget() {
  const int o = cfg_.gadget.opponent;
  const auto& v = root_cfv_[o];
  const auto& wv = cfg_.gadget.constraint;
  const int no = ctx_.n(o);
  if (cfg_.gadget.mode == GadgetConfig::Mode::kResolve) {
    for (int i = 0; i < no; ++i) {
      if (!ctx_.valid(cfg_.root, o, i)) continue;
      const double f = g_root_reach_[i];
      const double g = f * v[i] + (1.0 - f) * wv[i];
      g_regret_follow_[i] = std::max(0.0, g_regret_follow_[i] + v[i] - g);
      g_regret_term_[i] = std::max(0.0, g_regret_term_[i] + wv[i] - g);
      const double tot = g_regret_follow_[i] + g_regret_term_[i];
      g_root_reach_[i] = tot > 0.0 ? g_regret_follow_[i] / tot : 0.5;
    }
  } else {
    // MaxMargin: one opponent decision over hands with utility v(h) - w(h).
    double mean = 0.0;
    for (int i = 0; i < no; ++i) mean += g_root_reach_[i] * (v[i] - wv[i]);
    double total = 0.0;
    int live = 0;
    for (int i = 0; i < no; ++i) {
      if (!ctx_.valid(cfg_.root, o, i)) continue;
      ++live;
      g_regret_follow_[i] = std::max(0.0, g_regret_follow_[i] + (v[i] - wv[i]) - mean);
      total += g_regret_follow_[i];
    }
    for (int i = 0; i < no; ++i) {
      if (!ctx_.valid(cfg_.root, o, i)) {
        g_root_reach_[i] = 0.0;
        continue;
      }
      g_root_reach_[i] = total > 0.0 ? g_regret_follow_[i] / total : 1.0 / live;
    }
  }
}

void VectorCFR::iterate(int n) {
  for (int rep = 0; rep < n; ++rep) {
    ++t_;
    dpos_ = cfg_.cfr.regret_discount(t_, true);
    dneg_ = cfg_.cfr.regret_discount(t_, false);
    avg_w_ = cfg_.cfr.average_weight(t_);
    const bool alternating = cfg_.cfr.alternating;
    AEGIS_CHECK_MSG(alternating, "vector CFR uses alternating updates");
    const int root = cfg_.root;
    for (int p = 0; p < 2; ++p) {
      for (int q = 0; q < 2; ++q) reach_[q][root] = cfg_.root_reach[q];
      if (cfg_.gadget.enabled) reach_[cfg_.gadget.opponent][root] = g_root_reach_;
      forward(root);
      if (!leaves_.empty()) {
        std::vector<LeafValueProvider::Query> qs(leaves_.size());
        for (size_t k = 0; k < leaves_.size(); ++k) {
          const int leaf = leaves_[k];
          qs[k].node = leaf;
          for (int q = 0; q < 2; ++q) {
            qs[k].reach[q] = &reach_[q][leaf];
            leaf_cfv_[q][leaf].assign(ctx_.n(q), 0.0);
            qs[k].cfv[q] = &leaf_cfv_[q][leaf];
          }
        }
        cfg_.leaf->evaluate(ctx_, qs);
        leaf_queries_ += static_cast<long long>(qs.size());
      }
      backward(root, p);
      root_cfv_[p] = cfv_[root];
      hand_updates_ += subtree_nodes_ * (ctx_.n(0) + ctx_.n(1));
      if (cfg_.gadget.enabled && p == cfg_.gadget.opponent) update_gadget();
    }
  }
}

VectorPolicy VectorCFR::average_policy() const {
  VectorPolicy pol;
  pol.strategy.resize(tree_.num_decision_nodes());
  for (int n = 0; n < tree_.num_nodes(); ++n) {
    const auto& nd = tree_.nodes[n];
    if (nd.type != PNodeType::kDecision || !in_subtree_[n]) continue;
    const int np = ctx_.n(nd.player);
    const int A = nd.num_children;
    const auto& S = sum_[nd.decision_index];
    auto& out = pol.strategy[nd.decision_index];
    out.assign(S.size(), 0.0);
    for (int i = 0; i < np; ++i) {
      double total = 0.0;
      for (int a = 0; a < A; ++a) total += S[static_cast<size_t>(a) * np + i];
      for (int a = 0; a < A; ++a)
        out[static_cast<size_t>(a) * np + i] = total > 0.0 ? S[static_cast<size_t>(a) * np + i] / total : 1.0 / A;
    }
  }
  return pol;
}

VectorPolicy VectorCFR::current_policy() const {
  VectorPolicy pol;
  pol.strategy = sigma_;
  return pol;
}

std::vector<double> VectorCFR::gadget_root_reach() const { return g_root_reach_; }

size_t VectorCFR::memory_bytes() const {
  size_t b = 0;
  for (const auto& v : regret_) b += v.capacity() * sizeof(double);
  for (const auto& v : sum_) b += v.capacity() * sizeof(double);
  for (const auto& v : sigma_) b += v.capacity() * sizeof(double);
  for (int p = 0; p < 2; ++p)
    for (const auto& v : reach_[p]) b += v.capacity() * sizeof(double);
  for (const auto& v : cfv_) b += v.capacity() * sizeof(double);
  return b;
}

}  // namespace aegis::poker
