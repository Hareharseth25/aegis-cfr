#include "core/best_response/best_response.h"

#include <cmath>
#include <limits>

#include "util/check.h"

namespace aegis {

ReachProbabilities compute_reach(const GameTree& t, const TabularPolicy& pol) {
  const int n = t.num_nodes();
  ReachProbabilities r;
  r.player[0].assign(n, 0.0);
  r.player[1].assign(n, 0.0);
  r.chance.assign(n, 0.0);
  r.player[0][0] = r.player[1][0] = r.chance[0] = 1.0;
  // Pre-order ids: every parent is processed before its children.
  for (int h = 0; h < n; ++h) {
    if (t.type[h] == NodeType::kTerminal) continue;
    const int b = t.child_begin[h], c = t.child_count[h];
    for (int k = 0; k < c; ++k) {
      const int ch = t.children[b + k];
      r.player[0][ch] = r.player[0][h];
      r.player[1][ch] = r.player[1][h];
      r.chance[ch] = r.chance[h];
      if (t.type[h] == NodeType::kChance) {
        r.chance[ch] *= t.chance_prob[b + k];
      } else {
        const int p = t.player[h];
        r.player[p][ch] *= pol.probs[t.infosets[t.infoset[h]].offset + k];
      }
    }
  }
  return r;
}

std::array<double, 2> expected_values(const GameTree& t, const TabularPolicy& pol) {
  // Bottom-up over reverse pre-order: children have larger ids than parents.
  const int n = t.num_nodes();
  std::vector<double> v(n, 0.0);
  for (int h = n - 1; h >= 0; --h) {
    if (t.type[h] == NodeType::kTerminal) {
      v[h] = t.utility[0][h];
      continue;
    }
    const int b = t.child_begin[h], c = t.child_count[h];
    double s = 0.0;
    if (t.type[h] == NodeType::kChance) {
      for (int k = 0; k < c; ++k) s += t.chance_prob[b + k] * v[t.children[b + k]];
    } else {
      const int off = t.infosets[t.infoset[h]].offset;
      for (int k = 0; k < c; ++k) s += pol.probs[off + k] * v[t.children[b + k]];
    }
    v[h] = s;
  }
  if (t.zero_sum) return {v[0], -v[0]};
  // General-sum: recompute player 1's value separately.
  std::vector<double> w(n, 0.0);
  for (int h = n - 1; h >= 0; --h) {
    if (t.type[h] == NodeType::kTerminal) {
      w[h] = t.utility[1][h];
      continue;
    }
    const int b = t.child_begin[h], c = t.child_count[h];
    double s = 0.0;
    if (t.type[h] == NodeType::kChance) {
      for (int k = 0; k < c; ++k) s += t.chance_prob[b + k] * w[t.children[b + k]];
    } else {
      const int off = t.infosets[t.infoset[h]].offset;
      for (int k = 0; k < c; ++k) s += pol.probs[off + k] * w[t.children[b + k]];
    }
    w[h] = s;
  }
  return {v[0], w[0]};
}

namespace {

class BestResponder {
 public:
  BestResponder(const GameTree& t, const TabularPolicy& pol, int i)
      : t_(t), pol_(pol), i_(i), value_(t.num_nodes(), 0.0), done_(t.num_nodes(), 0),
        best_(t.num_infosets(), -1), q_(t.total_actions, 0.0) {
    auto reach = compute_reach(t, pol);
    opp_reach_.resize(t.num_nodes());
    for (int h = 0; h < t.num_nodes(); ++h) opp_reach_[h] = reach.chance[h] * reach.player[1 - i][h];
  }

  double value(int h) {
    if (done_[h]) return value_[h];
    double v = 0.0;
    const int b = t_.child_begin[h], c = t_.child_count[h];
    switch (t_.type[h]) {
      case NodeType::kTerminal:
        v = t_.utility[i_][h];
        break;
      case NodeType::kChance:
        for (int k = 0; k < c; ++k) v += t_.chance_prob[b + k] * value(t_.children[b + k]);
        break;
      case NodeType::kDecision: {
        const int I = t_.infoset[h];
        if (t_.player[h] == i_) {
          v = value(t_.children[b + best_action(I)]);
        } else {
          const int off = t_.infosets[I].offset;
          for (int k = 0; k < c; ++k) {
            const double p = pol_.probs[off + k];
            if (p > 0.0) v += p * value(t_.children[b + k]);
          }
        }
        break;
      }
    }
    value_[h] = v;
    done_[h] = 1;
    return v;
  }

  int best_action(int I) {
    if (best_[I] >= 0) return best_[I];
    const auto& info = t_.infosets[I];
    for (int a = 0; a < info.num_actions; ++a) q_[info.offset + a] = 0.0;
    for (int h : info.nodes) {
      const double w = opp_reach_[h];
      const int b = t_.child_begin[h];
      for (int a = 0; a < info.num_actions; ++a) {
        // Values are computed even for zero-reach histories so that every node
        // has a defined BR value (needed by ancestors' expectations).
        const double cv = value(t_.children[b + a]);
        q_[info.offset + a] += w * cv;
      }
    }
    int best = 0;
    for (int a = 1; a < info.num_actions; ++a)
      if (q_[info.offset + a] > q_[info.offset + best] + 1e-15) best = a;
    best_[I] = best;
    return best;
  }

  BestResponseResult finish() {
    BestResponseResult r;
    r.player = i_;
    r.value = value(0);
    // Make sure every infoset of the responder has a decision (unreachable
    // ones included) so the returned policy is complete.
    for (int I = 0; I < t_.num_infosets(); ++I)
      if (t_.infosets[I].player == i_) best_action(I);
    r.policy = pol_;
    r.best_action.assign(t_.num_infosets(), -1);
    r.cf_policy_value.assign(t_.num_infosets(), 0.0);
    for (int I = 0; I < t_.num_infosets(); ++I) {
      const auto& info = t_.infosets[I];
      if (info.player != i_) continue;
      r.best_action[I] = best_[I];
      double pv = 0.0;
      for (int a = 0; a < info.num_actions; ++a) {
        r.policy.probs[info.offset + a] = (a == best_[I]) ? 1.0 : 0.0;
        pv += pol_.probs[info.offset + a] * q_[info.offset + a];
      }
      r.cf_policy_value[I] = pv;
    }
    r.cf_action_value = q_;
    r.opponent_reach = opp_reach_;
    for (int h = 0; h < t_.num_nodes(); ++h) value(h);
    r.node_value = value_;
    return r;
  }

 private:
  const GameTree& t_;
  const TabularPolicy& pol_;
  int i_;
  std::vector<double> value_;
  std::vector<uint8_t> done_;
  std::vector<int> best_;
  std::vector<double> q_;
  std::vector<double> opp_reach_;
};

}  // namespace

BestResponseResult best_response(const GameTree& t, const TabularPolicy& pol, int i) {
  AEGIS_CHECK(i == 0 || i == 1);
  AEGIS_CHECK(static_cast<int>(pol.probs.size()) == t.total_actions);
  BestResponder br(t, pol, i);
  return br.finish();
}

ExploitabilityReport exploitability_report(const GameTree& t, const TabularPolicy& pol) {
  ExploitabilityReport rep;
  const auto ev = expected_values(t, pol);
  for (int i = 0; i < 2; ++i) {
    rep.br_value[i] = best_response(t, pol, i).value;
    rep.policy_value[i] = ev[i];
    rep.nash_conv += rep.br_value[i] - rep.policy_value[i];
  }
  rep.exploitability = rep.nash_conv / 2.0;
  return rep;
}

std::vector<double> counterfactual_regrets(const GameTree& t, const TabularPolicy& pol, int i) {
  AEGIS_CHECK(i == 0 || i == 1);
  const int n = t.num_nodes();
  const auto reach = compute_reach(t, pol);
  std::vector<double> v(n, 0.0);
  for (int h = n - 1; h >= 0; --h) {
    if (t.type[h] == NodeType::kTerminal) {
      v[h] = t.utility[i][h];
      continue;
    }
    const int b = t.child_begin[h], c = t.child_count[h];
    double s = 0.0;
    if (t.type[h] == NodeType::kChance)
      for (int k = 0; k < c; ++k) s += t.chance_prob[b + k] * v[t.children[b + k]];
    else
      for (int k = 0; k < c; ++k) s += pol.probs[t.infosets[t.infoset[h]].offset + k] * v[t.children[b + k]];
    v[h] = s;
  }
  std::vector<double> r(t.total_actions, 0.0);
  for (int h = 0; h < n; ++h) {
    if (t.type[h] != NodeType::kDecision || t.player[h] != i) continue;
    const auto& info = t.infosets[t.infoset[h]];
    const double w = reach.chance[h] * reach.player[1 - i][h];
    const int b = t.child_begin[h];
    for (int a = 0; a < info.num_actions; ++a) r[info.offset + a] += w * (v[t.children[b + a]] - v[h]);
  }
  return r;
}

double nash_conv(const GameTree& t, const TabularPolicy& pol) { return exploitability_report(t, pol).nash_conv; }
double exploitability(const GameTree& t, const TabularPolicy& pol) { return nash_conv(t, pol) / 2.0; }

}  // namespace aegis
