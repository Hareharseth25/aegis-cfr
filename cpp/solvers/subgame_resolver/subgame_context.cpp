#include "solvers/subgame_resolver/subgame_context.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "util/check.h"

namespace aegis::poker {

namespace {
// Inclusion-exclusion (total - Σ blocked-card sums) of non-negative masses can
// leave rounding residue of order 1e-17 where the exact compatible mass is
// zero. Regret matching would amplify such residue into a pure strategy, so
// masses below a relative tolerance are snapped to zero.
inline double clean_mass(double m, double total) { return m > 1e-13 * total ? m : 0.0; }
}  // namespace

double VectorPolicy::prob(const PublicTree& t, const HandSpace hs[2], int node, int a, int i) const {
  const auto& nd = t.node(node);
  AEGIS_CHECK(nd.type == PNodeType::kDecision && a >= 0 && a < nd.num_children && i >= 0 && i < hs[nd.player].size());
  const int n = hs[nd.player].size();
  return strategy[nd.decision_index][static_cast<size_t>(a) * n + i];
}

SubgameContext::SubgameContext(const PublicTree& tree, const Range& r0, const Range& r1, int threads,
                               std::array<bool, 2> full_support, std::array<const Range*, 2> support)
    : tree_(tree), threads_(std::max(1, threads)) {
  if (threads_ > 1) pool_ = std::make_unique<ThreadPool>(threads_);
  build_hands(r0, r1, full_support, support);
  build_showdowns();
}

void SubgameContext::build_hands(const Range& r0, const Range& r1, std::array<bool, 2> full,
                                 std::array<const Range*, 2> support) {
  const auto& C = combos();
  const CardMask bm = tree_.board_masks[0];
  const Range* rs[2] = {&r0, &r1};
  for (int p = 0; p < 2; ++p) {
    auto& H = hands_[p];
    H.local_of_combo.assign(kNumCombos, -1);
    for (int c = 0; c < kNumCombos; ++c) {
      if (C.mask[c] & bm) continue;
      const double w = (*rs[p])[c];
      AEGIS_CHECK_MSG(w >= 0.0 && std::isfinite(w), "invalid range weight");
      if (w <= 0.0 && !full[p] && !(support[p] && (*support[p])[c] > 0.0)) continue;
      H.local_of_combo[c] = H.size();
      H.combos.push_back(c);
      H.cards.push_back({C.cards[c][0], C.cards[c][1]});
      H.root_weight.push_back(w);
    }
    AEGIS_CHECK_MSG(H.size() > 0, "player " << p << " has an empty range on this board");
  }
  for (int p = 0; p < 2; ++p) {
    auto& H = hands_[p];
    H.same_in_opponent.assign(H.size(), -1);
    for (int i = 0; i < H.size(); ++i) H.same_in_opponent[i] = hands_[1 - p].local_of_combo[H.combos[i]];
  }
  // Z = Σ_{i,j disjoint} r0(i) r1(j).
  const auto& H1 = hands_[1];
  double total = 0.0, card[kNumCards] = {};
  for (int j = 0; j < H1.size(); ++j) {
    total += H1.root_weight[j];
    card[H1.cards[j][0]] += H1.root_weight[j];
    card[H1.cards[j][1]] += H1.root_weight[j];
  }
  const auto& H0 = hands_[0];
  Z_ = 0.0;
  for (int i = 0; i < H0.size(); ++i) {
    const int same = H0.same_in_opponent[i];
    const double compat = clean_mass(
        total - card[H0.cards[i][0]] - card[H0.cards[i][1]] + (same >= 0 ? H1.root_weight[same] : 0.0), total);
    Z_ += H0.root_weight[i] * compat;
  }
  AEGIS_CHECK_MSG(Z_ > 0.0, "ranges have no compatible hand pairs on this board");
}

void SubgameContext::build_showdowns() {
  const int B = static_cast<int>(tree_.boards.size());
  for (int p = 0; p < 2; ++p) {
    valid_[p].assign(B, {});
    for (int b = 0; b < B; ++b) {
      valid_[p][b].assign(hands_[p].size(), 0);
      const CardMask bm = tree_.board_masks[b];
      for (int i = 0; i < hands_[p].size(); ++i)
        valid_[p][b][i] = (combos().mask[hands_[p].combos[i]] & bm) ? 0 : 1;
    }
  }
  showdown_.assign(B, {});
  // Only boards that are reached by a showdown node need strengths.
  std::vector<uint8_t> need(B, 0);
  for (const auto& n : tree_.nodes)
    if (n.type == PNodeType::kShowdown) need[n.board_id] = 1;
  for (int b = 0; b < B; ++b) {
    if (!need[b]) continue;
    AEGIS_CHECK(tree_.boards[b].size() == 5);
    auto& S = showdown_[b];
    const CardMask bm = tree_.board_masks[b];
    for (int p = 0; p < 2; ++p) {
      const auto& H = hands_[p];
      S.value[p].assign(H.size(), 0);
      S.order[p].clear();
      for (int i = 0; i < H.size(); ++i) {
        if (!valid_[p][b][i]) continue;
        S.value[p][i] = poker::evaluate(bm | combos().mask[H.combos[i]]);
        S.order[p].push_back(i);
      }
      std::stable_sort(S.order[p].begin(), S.order[p].end(),
                       [&](int a, int c) { return S.value[p][a] < S.value[p][c]; });
    }
    S.ready = true;
  }
}

double SubgameContext::chance_weight(int node) const {
  const auto& nd = tree_.nodes[node];
  const int board_cards = static_cast<int>(tree_.boards[nd.board_id].size());
  return 1.0 / (kNumCards - board_cards - 4);
}

void SubgameContext::fold_cfv(int node, int p, const double* opp, double* out) const {
  const auto& nd = tree_.nodes[node];
  const int o = 1 - p;
  const double u = nd.player == p ? -nd.commit[p] : nd.commit[nd.player];
  const auto& Ho = hands_[o];
  const auto& Hp = hands_[p];
  double total = 0.0, card[kNumCards] = {};
  for (int j = 0; j < Ho.size(); ++j) {
    const double r = opp[j];
    total += r;
    card[Ho.cards[j][0]] += r;
    card[Ho.cards[j][1]] += r;
  }
  const uint8_t* v = valid_[p][nd.board_id].data();
  for (int i = 0; i < Hp.size(); ++i) {
    if (!v[i]) {
      out[i] = 0.0;
      continue;
    }
    const int same = Hp.same_in_opponent[i];
    const double m = total - card[Hp.cards[i][0]] - card[Hp.cards[i][1]] + (same >= 0 ? opp[same] : 0.0);
    out[i] = u * clean_mass(m, total);
  }
}

void SubgameContext::showdown_cfv(int node, int p, const double* opp, double* out) const {
  const auto& nd = tree_.nodes[node];
  const auto& S = showdown_[nd.board_id];
  AEGIS_CHECK(S.ready);
  const int o = 1 - p;
  const double C = nd.commit[0];
  AEGIS_CHECK_MSG(std::abs(nd.commit[0] - nd.commit[1]) < 1e-7, "showdown with unequal commitments");
  const auto& Hp = hands_[p];
  const auto& Ho = hands_[o];
  const auto& op = S.order[p];
  const auto& oo = S.order[o];
  const auto& vp = S.value[p];
  const auto& vo = S.value[o];
  std::fill(out, out + Hp.size(), 0.0);
  // Ascending sweep: opponent mass strictly weaker.
  {
    double total = 0.0, card[kNumCards] = {};
    size_t j = 0;
    for (size_t k = 0; k < op.size(); ++k) {
      const int i = op[k];
      const HandValue v = vp[i];
      while (j < oo.size() && vo[oo[j]] < v) {
        const int h = oo[j++];
        const double r = opp[h];
        total += r;
        card[Ho.cards[h][0]] += r;
        card[Ho.cards[h][1]] += r;
      }
      out[i] = clean_mass(total - card[Hp.cards[i][0]] - card[Hp.cards[i][1]], total);
    }
  }
  // Descending sweep: opponent mass strictly stronger.
  {
    double total = 0.0, card[kNumCards] = {};
    ptrdiff_t j = static_cast<ptrdiff_t>(oo.size()) - 1;
    for (ptrdiff_t k = static_cast<ptrdiff_t>(op.size()) - 1; k >= 0; --k) {
      const int i = op[k];
      const HandValue v = vp[i];
      while (j >= 0 && vo[oo[j]] > v) {
        const int h = oo[j--];
        const double r = opp[h];
        total += r;
        card[Ho.cards[h][0]] += r;
        card[Ho.cards[h][1]] += r;
      }
      out[i] = C * (out[i] - clean_mass(total - card[Hp.cards[i][0]] - card[Hp.cards[i][1]], total));
    }
  }
}

VectorPolicy SubgameContext::uniform_policy() const {
  VectorPolicy pol;
  pol.strategy.resize(tree_.num_decision_nodes());
  for (int n = 0; n < tree_.num_nodes(); ++n) {
    const auto& nd = tree_.nodes[n];
    if (nd.type != PNodeType::kDecision) continue;
    const int np = hands_[nd.player].size();
    pol.strategy[nd.decision_index].assign(static_cast<size_t>(nd.num_children) * np, 1.0 / nd.num_children);
  }
  return pol;
}

void SubgameContext::forward(const VectorPolicy& pol, int node, std::vector<std::vector<double>>& reach,
                             int p) const {
  const auto& nd = tree_.nodes[node];
  if (nd.num_children == 0) return;
  const int np = hands_[p].size();
  const auto& r = reach[node];
  if (nd.type == PNodeType::kChance) {
    auto task = [&](int k) {
      const int c = nd.first_child + k;
      const uint8_t* v = valid_[p][tree_.nodes[c].board_id].data();
      auto& rc = reach[c];
      rc.resize(np);
      for (int i = 0; i < np; ++i) rc[i] = v[i] ? r[i] : 0.0;
      forward(pol, c, reach, p);
    };
    if (pool_) pool_->parallel_for(nd.num_children, task);
    else
      for (int k = 0; k < nd.num_children; ++k) task(k);
    return;
  }
  for (int k = 0; k < nd.num_children; ++k) {
    const int c = nd.first_child + k;
    auto& rc = reach[c];
    if (nd.player == p) {
      const double* s = pol.strategy[nd.decision_index].data() + static_cast<size_t>(k) * np;
      rc.resize(np);
      for (int i = 0; i < np; ++i) rc[i] = r[i] * s[i];
    } else {
      rc = r;
    }
    forward(pol, c, reach, p);
  }
}

void SubgameContext::backward(const VectorPolicy& pol, int node, int p, int br_player,
                              const std::vector<std::vector<double>>& opp, std::vector<std::vector<double>>& cfv) const {
  const auto& nd = tree_.nodes[node];
  const int np = hands_[p].size();
  auto& out = cfv[node];
  out.assign(np, 0.0);
  switch (nd.type) {
    case PNodeType::kFold:
      fold_cfv(node, p, opp[node].data(), out.data());
      return;
    case PNodeType::kShowdown:
      showdown_cfv(node, p, opp[node].data(), out.data());
      return;
    case PNodeType::kLeaf:
      AEGIS_CHECK_MSG(false, "policy evaluation of depth-limited leaves needs a leaf value provider");
      return;
    case PNodeType::kChance: {
      auto task = [&](int k) { backward(pol, nd.first_child + k, p, br_player, opp, cfv); };
      if (pool_) pool_->parallel_for(nd.num_children, task);
      else
        for (int k = 0; k < nd.num_children; ++k) task(k);
      const double w = chance_weight(node);
      for (int k = 0; k < nd.num_children; ++k) {
        const auto& cc = cfv[nd.first_child + k];
        for (int i = 0; i < np; ++i) out[i] += w * cc[i];
      }
      return;
    }
    case PNodeType::kDecision:
      break;
  }
  for (int k = 0; k < nd.num_children; ++k) backward(pol, nd.first_child + k, p, br_player, opp, cfv);
  if (nd.player != p) {
    for (int k = 0; k < nd.num_children; ++k) {
      const auto& cc = cfv[nd.first_child + k];
      for (int i = 0; i < np; ++i) out[i] += cc[i];
    }
  } else if (br_player == p) {
    for (int i = 0; i < np; ++i) out[i] = -1e300;
    for (int k = 0; k < nd.num_children; ++k) {
      const auto& cc = cfv[nd.first_child + k];
      for (int i = 0; i < np; ++i) out[i] = std::max(out[i], cc[i]);
    }
  } else {
    const double* s = pol.strategy[nd.decision_index].data();
    for (int k = 0; k < nd.num_children; ++k) {
      const auto& cc = cfv[nd.first_child + k];
      const double* sk = s + static_cast<size_t>(k) * np;
      for (int i = 0; i < np; ++i) out[i] += sk[i] * cc[i];
    }
  }
}

std::vector<std::vector<double>> SubgameContext::node_reach(const VectorPolicy& pol, int p,
                                                            const std::vector<double>* root) const {
  std::vector<std::vector<double>> reach(tree_.num_nodes());
  reach[0] = root ? *root : hands_[p].root_weight;
  AEGIS_CHECK(static_cast<int>(reach[0].size()) == hands_[p].size());
  forward(pol, 0, reach, p);
  return reach;
}

void SubgameContext::check_policy(const VectorPolicy& pol) const {
  AEGIS_CHECK_MSG(static_cast<int>(pol.strategy.size()) == tree_.num_decision_nodes(),
                  "policy has " << pol.strategy.size() << " decision nodes, tree has " << tree_.num_decision_nodes());
  for (const auto& nd : tree_.nodes) {
    if (nd.type != PNodeType::kDecision) continue;
    const auto& s = pol.strategy[nd.decision_index];
    if (s.empty()) continue;  // sub-tree policies leave unsolved nodes empty
    AEGIS_CHECK_MSG(s.size() == static_cast<size_t>(nd.num_children) * hands_[nd.player].size(),
                    "policy entry for a decision node has the wrong size");
  }
}

SubgameContext::Values SubgameContext::evaluate(const VectorPolicy& pol) const {
  check_policy(pol);
  Values v;
  for (int p = 0; p < 2; ++p) {
    const auto opp = node_reach(pol, 1 - p);
    std::vector<std::vector<double>> cfv(tree_.num_nodes());
    backward(pol, 0, p, -1, opp, cfv);
    v.root_cfv[p] = cfv[0];
    double s = 0.0;
    for (int i = 0; i < hands_[p].size(); ++i) s += hands_[p].root_weight[i] * cfv[0][i];
    v.ev[p] = s / Z_;
  }
  return v;
}

SubgameContext::Values SubgameContext::best_response(const VectorPolicy& pol, int p) const {
  AEGIS_CHECK(p == 0 || p == 1);
  check_policy(pol);
  Values v;
  const auto opp = node_reach(pol, 1 - p);
  std::vector<std::vector<double>> cfv(tree_.num_nodes());
  backward(pol, 0, p, p, opp, cfv);
  v.root_cfv[p] = cfv[0];
  double s = 0.0;
  for (int i = 0; i < hands_[p].size(); ++i) s += hands_[p].root_weight[i] * cfv[0][i];
  v.ev[p] = s / Z_;
  return v;
}

SubgameContext::Exploitability SubgameContext::exploitability(const VectorPolicy& pol) const {
  Exploitability e;
  const auto ev = evaluate(pol);
  for (int p = 0; p < 2; ++p) {
    e.br_value[p] = best_response(pol, p).ev[p];
    e.policy_value[p] = ev.ev[p];
    e.nash_conv += e.br_value[p] - e.policy_value[p];
  }
  e.exploitability = e.nash_conv / 2.0;
  e.exploitability_pct_pot = 100.0 * e.exploitability / tree_.config().pot;
  return e;
}

void SubgameContext::subtree_values(const VectorPolicy& pol, int node, const std::vector<double> reach[2],
                                    int br_player, std::vector<double> out[2]) const {
  tree_.node(node);  // bounds check
  AEGIS_CHECK(br_player >= -1 && br_player <= 1);
  check_policy(pol);
  for (int p = 0; p < 2; ++p) AEGIS_CHECK(static_cast<int>(reach[p].size()) == hands_[p].size());
  std::vector<std::vector<double>> opp_reach[2];
  for (int p = 0; p < 2; ++p) {
    opp_reach[p].assign(tree_.num_nodes(), {});
    opp_reach[p][node] = reach[p];
    forward(pol, node, opp_reach[p], p);
  }
  for (int p = 0; p < 2; ++p) {
    std::vector<std::vector<double>> cfv(tree_.num_nodes());
    backward(pol, node, p, br_player, opp_reach[1 - p], cfv);
    out[p] = cfv[node];
  }
}

}  // namespace aegis::poker
