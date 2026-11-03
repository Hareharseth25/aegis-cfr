#include "solvers/vanilla_cfr/naive_cfr.h"

#include "solvers/common/regret_rules.h"
#include "util/check.h"

namespace aegis {

NaiveCFR::Node& NaiveCFR::node(const State& s, int p) {
  const std::string key = std::to_string(p) + "|" + s.information_state_key(p);
  auto it = nodes_.find(key);
  if (it != nodes_.end()) return *it->second;
  auto n = std::make_unique<Node>();
  const size_t k = s.legal_actions().size();
  n->regret.assign(k, 0.0);
  n->inst.assign(k, 0.0);
  n->sum.assign(k, 0.0);
  n->sigma.assign(k, 1.0 / k);
  n->player = p;
  auto& ref = *n;
  nodes_.emplace(key, std::move(n));
  return ref;
}

double NaiveCFR::walk(const State& s, int i, double r0, double r1, double rc) {
  ++touched_;
  if (s.is_terminal()) return s.returns()[i];
  if (s.is_chance()) {
    double v = 0.0;
    for (const auto& [a, p] : s.chance_outcomes()) v += p * walk(*s.child(a), i, r0, r1, rc * p);
    return v;
  }
  if (r0 == 0.0 && r1 == 0.0) return 0.0;
  const int p = s.current_player();
  Node& n = node(s, p);
  const auto actions = s.legal_actions();
  std::vector<double> cv(actions.size());
  double v = 0.0;
  for (size_t a = 0; a < actions.size(); ++a) {
    const double s0 = p == 0 ? r0 * n.sigma[a] : r0;
    const double s1 = p == 1 ? r1 * n.sigma[a] : r1;
    cv[a] = walk(*s.child(actions[a]), i, s0, s1, rc);
    v += n.sigma[a] * cv[a];
  }
  if (p == i) {
    const double cf = (p == 0 ? r1 : r0) * rc;
    const double own = p == 0 ? r0 : r1;
    for (size_t a = 0; a < actions.size(); ++a) {
      n.inst[a] += cf * (cv[a] - v);
      n.sum[a] += own * n.sigma[a];
    }
  }
  return v;
}

void NaiveCFR::iterate(int n) {
  for (int k = 0; k < n; ++k) {
    ++t_;
    for (int i = 0; i < 2; ++i) {
      auto root = game_.initial_state();
      walk(*root, i, 1.0, 1.0, 1.0);
      for (auto& [key, node] : nodes_) {
        if (node->player != i) continue;
        for (size_t a = 0; a < node->regret.size(); ++a) {
          node->regret[a] += node->inst[a];
          node->inst[a] = 0.0;
        }
        regret_matching(node->regret.data(), node->sigma.data(), static_cast<int>(node->regret.size()));
      }
    }
  }
}

std::vector<double> NaiveCFR::average_strategy(int p, const std::string& key) const {
  auto it = nodes_.find(std::to_string(p) + "|" + key);
  AEGIS_CHECK_MSG(it != nodes_.end(), "unknown infoset " << key);
  const auto& s = it->second->sum;
  double total = 0.0;
  for (double x : s) total += x;
  std::vector<double> out(s.size(), 1.0 / s.size());
  if (total > 0.0)
    for (size_t a = 0; a < s.size(); ++a) out[a] = s[a] / total;
  return out;
}

}  // namespace aegis
