#include "solvers/subgame_resolver/tabular_resolving.h"

#include <algorithm>
#include <unordered_map>

#include "core/best_response/best_response.h"
#include "util/check.h"

namespace aegis {

std::vector<int> nodes_with_public_state(const GameTree& t, int ps) {
  std::vector<int> out;
  for (int h = 0; h < t.num_nodes(); ++h)
    if (t.public_state[h] == ps) out.push_back(h);
  return out;
}

int public_state_id(const GameTree& t, const std::string& key) {
  for (int i = 0; i < static_cast<int>(t.public_keys.size()); ++i)
    if (t.public_keys[i] == key) return i;
  return -1;
}

namespace {

class SubgameResolver {
 public:
  SubgameResolver(const GameTree& t, const TabularPolicy& bp, const std::vector<int>& roots,
                  const TabularResolveOptions& o)
      : t_(t), bp_(bp), roots_(roots), o_(o), us_(o.us), opp_(1 - o.us) {
    AEGIS_CHECK(us_ == 0 || us_ == 1);
    AEGIS_CHECK_MSG(!roots_.empty(), "subgame has no roots");
    mark_subgame();
    reach_ = compute_reach(t_, bp_);
    // Opponent root infosets J.
    for (int h : roots_) {
      AEGIS_CHECK_MSG(t_.type[h] != NodeType::kTerminal, "subgame root is terminal");
      const int aug = t_.aug_infoset[opp_][h];
      auto it = j_index_.find(aug);
      if (it == j_index_.end()) {
        it = j_index_.emplace(aug, static_cast<int>(j_index_.size())).first;
      }
      root_j_.push_back(it->second);
    }
    const int J = static_cast<int>(j_index_.size());
    // Alternative values w(J) in counterfactual units: Σ_{h∈J} π_c(h) π_us(h) v_opp(h).
    std::vector<double> v_opp(t_.num_nodes(), 0.0);
    if (o_.use_cbv) {
      v_opp = best_response(t_, bp_, opp_).node_value;
    } else {
      v_opp = node_values(bp_, opp_);
    }
    w_.assign(J, 0.0);
    for (size_t k = 0; k < roots_.size(); ++k) {
      const int h = roots_[k];
      w_[root_j_[k]] += reach_.chance[h] * reach_.player[us_][h] * v_opp[h];
    }
    regret_.assign(t_.total_actions, 0.0);
    inst_.assign(t_.total_actions, 0.0);
    sum_.assign(t_.total_actions, 0.0);
    sigma_ = bp_.probs;
    for (int I : infosets_) {
      const auto& info = t_.infosets[I];
      regret_matching(&regret_[info.offset], &sigma_[info.offset], info.num_actions);
    }
    gR_follow_.assign(J, 0.0);
    gR_term_.assign(J, 0.0);
    follow_.assign(J, o_.method == ResolveMethod::kMaxMargin ? 1.0 / J : 0.5);
  }

  TabularResolveResult run() {
    for (int it = 1; it <= o_.iterations; ++it) {
      t_iter_ = it;
      for (int p = 0; p < 2; ++p) {
        std::vector<double> vF(follow_.size(), 0.0);
        for (size_t k = 0; k < roots_.size(); ++k) {
          const int h = roots_[k];
          double r[2];
          r[us_] = reach_.player[us_][h];
          r[opp_] = o_.method == ResolveMethod::kNaive ? reach_.player[opp_][h] : follow_[root_j_[k]];
          const double v = traverse(h, p, r[0], r[1], reach_.chance[h]);
          if (p == opp_) vF[root_j_[k]] += reach_.chance[h] * reach_.player[us_][h] * v;
        }
        finish(p);
        if (p == opp_ && o_.method != ResolveMethod::kNaive) update_gadget(vF);
      }
    }
    TabularResolveResult res;
    res.policy = bp_;
    const TabularPolicy avg = TabularPolicy::from_weights(t_, sum_);
    for (int I : infosets_) {
      const auto& info = t_.infosets[I];
      if (info.player != us_ && !o_.replace_opponent) continue;
      for (int a = 0; a < info.num_actions; ++a) res.policy.probs[info.offset + a] = avg.probs[info.offset + a];
    }
    res.subgame_infosets = infosets_;
    res.constraint = w_;
    res.gadget_follow = follow_;
    // Safety margins of the re-solved strategy: w(J) - CBV_new(J).
    const auto br = best_response(t_, res.policy, opp_);
    const auto reach_new = compute_reach(t_, res.policy);
    res.final_margin.assign(w_.size(), 0.0);
    std::vector<double> v_new(w_.size(), 0.0);
    for (size_t k = 0; k < roots_.size(); ++k) {
      const int h = roots_[k];
      v_new[root_j_[k]] += reach_new.chance[h] * reach_new.player[us_][h] * br.node_value[h];
    }
    for (size_t j = 0; j < w_.size(); ++j) res.final_margin[j] = w_[j] - v_new[j];
    return res;
  }

 private:
  void mark_subgame() {
    in_.assign(t_.num_nodes(), 0);
    std::vector<int> stack = roots_;
    std::vector<uint8_t> seen_iset(t_.num_infosets(), 0);
    while (!stack.empty()) {
      const int h = stack.back();
      stack.pop_back();
      if (in_[h]) continue;
      in_[h] = 1;
      if (t_.type[h] == NodeType::kDecision && !seen_iset[t_.infoset[h]]) {
        seen_iset[t_.infoset[h]] = 1;
        infosets_.push_back(t_.infoset[h]);
      }
      for (int k = 0; k < t_.child_count[h]; ++k) stack.push_back(t_.children[t_.child_begin[h] + k]);
    }
    std::sort(infosets_.begin(), infosets_.end());
    for (int I : infosets_)
      for (int h : t_.infosets[I].nodes)
        AEGIS_CHECK_MSG(in_[h], "subgame is not closed: information set '" << t_.infosets[I].key
                                                                            << "' extends outside the subgame");
  }

  std::vector<double> node_values(const TabularPolicy& pol, int p) const {
    std::vector<double> v(t_.num_nodes(), 0.0);
    for (int h = t_.num_nodes() - 1; h >= 0; --h) {
      if (t_.type[h] == NodeType::kTerminal) {
        v[h] = t_.utility[p][h];
        continue;
      }
      const int b = t_.child_begin[h], c = t_.child_count[h];
      double s = 0.0;
      if (t_.type[h] == NodeType::kChance)
        for (int k = 0; k < c; ++k) s += t_.chance_prob[b + k] * v[t_.children[b + k]];
      else
        for (int k = 0; k < c; ++k) s += pol.probs[t_.infosets[t_.infoset[h]].offset + k] * v[t_.children[b + k]];
      v[h] = s;
    }
    return v;
  }

  double traverse(int h, int i, double r0, double r1, double rc) {
    if (t_.type[h] == NodeType::kTerminal) return t_.utility[i][h];
    const int b = t_.child_begin[h], c = t_.child_count[h];
    if (t_.type[h] == NodeType::kChance) {
      double v = 0.0;
      for (int k = 0; k < c; ++k) {
        const double p = t_.chance_prob[b + k];
        v += p * traverse(t_.children[b + k], i, r0, r1, rc * p);
      }
      return v;
    }
    if (r0 == 0.0 && r1 == 0.0) return 0.0;
    const int p = t_.player[h];
    const auto& info = t_.infosets[t_.infoset[h]];
    const double* s = &sigma_[info.offset];
    double cv[64];
    AEGIS_CHECK(c <= 64);
    double v = 0.0;
    for (int a = 0; a < c; ++a) {
      cv[a] = traverse(t_.children[b + a], i, p == 0 ? r0 * s[a] : r0, p == 1 ? r1 * s[a] : r1, rc);
      v += s[a] * cv[a];
    }
    if (p == i) {
      const double cf = (p == 0 ? r1 : r0) * rc;
      const double own = p == 0 ? r0 : r1;
      const double w = o_.cfr.average_weight(t_iter_);
      for (int a = 0; a < c; ++a) {
        inst_[info.offset + a] += cf * tie_snapped(cv[a], v, o_.cfr.tie_tolerance);
        sum_[info.offset + a] += w * own * s[a];
      }
    }
    return v;
  }

  void finish(int p) {
    for (int I : infosets_) {
      const auto& info = t_.infosets[I];
      if (info.player != p) continue;
      for (int a = 0; a < info.num_actions; ++a) {
        o_.cfr.update_regret(regret_[info.offset + a], inst_[info.offset + a], t_iter_);
        inst_[info.offset + a] = 0.0;
      }
      regret_matching(&regret_[info.offset], &sigma_[info.offset], info.num_actions);
    }
  }

  void update_gadget(const std::vector<double>& vF) {
    const int J = static_cast<int>(follow_.size());
    if (o_.method == ResolveMethod::kResolve) {
      for (int j = 0; j < J; ++j) {
        const double f = follow_[j];
        const double g = f * vF[j] + (1.0 - f) * w_[j];
        gR_follow_[j] = std::max(0.0, gR_follow_[j] + vF[j] - g);
        gR_term_[j] = std::max(0.0, gR_term_[j] + w_[j] - g);
        const double tot = gR_follow_[j] + gR_term_[j];
        follow_[j] = tot > 0.0 ? gR_follow_[j] / tot : 0.5;
      }
    } else {
      double mean = 0.0;
      for (int j = 0; j < J; ++j) mean += follow_[j] * (vF[j] - w_[j]);
      double total = 0.0;
      for (int j = 0; j < J; ++j) {
        gR_follow_[j] = std::max(0.0, gR_follow_[j] + (vF[j] - w_[j]) - mean);
        total += gR_follow_[j];
      }
      for (int j = 0; j < J; ++j) follow_[j] = total > 0.0 ? gR_follow_[j] / total : 1.0 / J;
    }
  }

  const GameTree& t_;
  const TabularPolicy& bp_;
  std::vector<int> roots_;
  TabularResolveOptions o_;
  int us_, opp_;
  std::vector<uint8_t> in_;
  std::vector<int> infosets_;
  ReachProbabilities reach_;
  std::unordered_map<int, int> j_index_;
  std::vector<int> root_j_;
  std::vector<double> w_;
  std::vector<double> regret_, inst_, sum_, sigma_;
  std::vector<double> gR_follow_, gR_term_, follow_;
  int t_iter_ = 0;
};

}  // namespace

TabularResolveResult resolve_subgame(const GameTree& tree, const TabularPolicy& blueprint,
                                     const std::vector<int>& roots, const TabularResolveOptions& opts) {
  SubgameResolver r(tree, blueprint, roots, opts);
  return r.run();
}

}  // namespace aegis
