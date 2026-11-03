#include "solvers/mccfr/mccfr.h"

#include <algorithm>

#include "solvers/common/regret_rules.h"
#include "util/check.h"
#include "util/thread_pool.h"

namespace aegis {

namespace {
constexpr int kMaxActions = 64;
}

MCCFR::MCCFR(const GameTree& tree, MCCFRConfig cfg)
    : tree_(tree), cfg_(cfg), rng_(cfg.seed), regret_(tree.total_actions, 0.0),
      strategy_sum_(tree.total_actions, 0.0), updates_(tree.num_infosets(), 0) {
  AEGIS_CHECK_MSG(tree.zero_sum, "MCCFR requires a two-player zero-sum game");
  AEGIS_CHECK(cfg.os_epsilon >= 0.0 && cfg.os_epsilon <= 1.0);
}

void MCCFR::current_sigma(int I, double* out) const {
  const auto& info = tree_.infosets[I];
  if (snapshot_sigma_) {
    std::copy_n(snapshot_sigma_->data() + info.offset, info.num_actions, out);
  } else {
    regret_matching(&regret_[info.offset], out, info.num_actions);
  }
}

void MCCFR::add_regret(Buffers& buf, int slot, double r) {
  if (buf.dregret) {
    (*buf.dregret)[slot] += r;
    return;
  }
  regret_[slot] += r;
  if (cfg_.regret_matching_plus && regret_[slot] < 0.0) regret_[slot] = 0.0;
}

void MCCFR::add_avg(Buffers& buf, int slot, double w) {
  if (buf.dsum) (*buf.dsum)[slot] += w;
  else strategy_sum_[slot] += w;
}

MCCFR::Out MCCFR::traverse_es(int h, int i, double W, Buffers& buf) {
  ++buf.touched;
  const auto& t = tree_;
  const bool importance = cfg_.scheme == MCCFRScheme::kImportance && sampler_ != nullptr;
  if (t.type[h] == NodeType::kTerminal) return {t.utility[i][h], 0.0, 1};

  const int b = t.child_begin[h], c = t.child_count[h];
  AEGIS_CHECK(c <= kMaxActions);

  if (t.type[h] == NodeType::kChance) {
    const std::span<const double> p(&t.chance_prob[b], c);
    int k;
    double ratio = 1.0;
    if (importance) {
      double q[kMaxActions];
      sampler_->sampling_distribution(h, i, p, std::span<double>(q, c));
      k = buf.rng->sample(std::span<const double>(q, c));
      ratio = p[k] / q[k];
    } else {
      k = buf.rng->sample(p);
    }
    Out o = traverse_es(t.children[b + k], i, W * ratio, buf);
    if (sampler_) sampler_->on_edge_backup(h, k, o.prio, o.cost);
    return {ratio * o.v, o.prio, o.cost + 1};
  }

  const int p = t.player[h];
  const int I = t.infoset[h];
  const auto& info = t.infosets[I];
  double sigma[kMaxActions];
  current_sigma(I, sigma);

  if (p != i) {
    // Opponent node: average-strategy update ("simple" averaging), then sample.
    const double w = (cfg_.linear_averaging ? static_cast<double>(t_) : 1.0) * W;
    for (int a = 0; a < c; ++a) add_avg(buf, info.offset + a, w * sigma[a]);
    int k;
    double ratio = 1.0;
    if (importance) {
      double q[kMaxActions];
      sampler_->sampling_distribution(h, i, std::span<const double>(sigma, c), std::span<double>(q, c));
      k = buf.rng->sample(std::span<const double>(q, c));
      ratio = sigma[k] / q[k];
    } else {
      k = buf.rng->sample(std::span<const double>(sigma, c));
    }
    Out o = traverse_es(t.children[b + k], i, W * ratio, buf);
    if (sampler_) sampler_->on_edge_backup(h, k, o.prio, o.cost);
    return {ratio * o.v, o.prio, o.cost + 1};
  }

  // Traverser node: explore every action.
  double vals[kMaxActions];
  double v = 0.0, prio = 0.0;
  int64_t cost = 1;
  for (int a = 0; a < c; ++a) {
    Out o = traverse_es(t.children[b + a], i, W, buf);
    vals[a] = o.v;
    v += sigma[a] * o.v;
    prio += o.prio;
    cost += o.cost;
  }
  for (int a = 0; a < c; ++a) add_regret(buf, info.offset + a, W * (vals[a] - v));
  if (buf.dupdates) (*buf.dupdates)[I]++;
  else updates_[I]++;
  if (sampler_) {
    sampler_->on_traverser_visit(I, h, W, 0.0, std::span<const double>(vals, c), v, cost);
    prio += sampler_->infoset_priority(I);
  }
  return {v, prio, cost};
}

double MCCFR::traverse_os(int h, int i, double my_reach, double opp_reach, double sample_reach, double& tail,
                          Buffers& buf) {
  // Returns W = u_i(z) π_{-i}(z) / q(z) of the sampled terminal z and sets
  // tail = π_i(h, z). `opp_reach` here is the opponent's OWN reach (chance is
  // sampled on-policy, so its probability cancels against q).
  ++buf.touched;
  const auto& t = tree_;
  if (t.type[h] == NodeType::kTerminal) {
    tail = 1.0;
    return t.utility[i][h] / sample_reach * opp_reach;
  }
  const int b = t.child_begin[h], c = t.child_count[h];
  AEGIS_CHECK(c <= kMaxActions);
  if (t.type[h] == NodeType::kChance) {
    const int k = buf.rng->sample(std::span<const double>(&t.chance_prob[b], c));
    // Chance: π_c/q_c = 1, so neither reach changes.
    return traverse_os(t.children[b + k], i, my_reach, opp_reach, sample_reach, tail, buf);
  }
  const int p = t.player[h];
  const int I = t.infoset[h];
  const auto& info = t.infosets[I];
  double sigma[kMaxActions];
  current_sigma(I, sigma);

  if (p == i) {
    double q[kMaxActions];
    const double eps = cfg_.os_epsilon;
    for (int a = 0; a < c; ++a) q[a] = eps / c + (1.0 - eps) * sigma[a];
    const int k = buf.rng->sample(std::span<const double>(q, c));
    double child_tail = 1.0;
    const double W = traverse_os(t.children[b + k], i, my_reach * sigma[k], opp_reach, sample_reach * q[k],
                                 child_tail, buf);
    tail = sigma[k] * child_tail;
    for (int a = 0; a < c; ++a) {
      const double r = (a == k) ? W * child_tail * (1.0 - sigma[k]) : -W * tail;
      add_regret(buf, info.offset + a, r);
    }
    if (buf.dupdates) (*buf.dupdates)[I]++;
    else updates_[I]++;
    return W;
  }
  // Opponent: stochastically-weighted average update, then sample on-policy.
  const double w = (cfg_.linear_averaging ? static_cast<double>(t_) : 1.0) * opp_reach / sample_reach;
  for (int a = 0; a < c; ++a) add_avg(buf, info.offset + a, w * sigma[a]);
  const int k = buf.rng->sample(std::span<const double>(sigma, c));
  double child_tail = 1.0;
  const double W = traverse_os(t.children[b + k], i, my_reach, opp_reach * sigma[k], sample_reach * sigma[k],
                               child_tail, buf);
  tail = child_tail;
  return W;
}

void MCCFR::iterate(int n) {
  for (int k = 0; k < n; ++k) {
    ++t_;
    for (int i = 0; i < 2; ++i) {
      Buffers buf{nullptr, nullptr, nullptr, &rng_};
      if (sampler_) sampler_->begin_traversal(i);
      if (cfg_.scheme == MCCFRScheme::kOutcome) {
        double tail = 1.0;
        traverse_os(0, i, 1.0, 1.0, 1.0, tail, buf);
      } else {
        traverse_es(0, i, 1.0, buf);
      }
      if (sampler_) sampler_->end_traversal(i);
      nodes_touched_ += buf.touched;
    }
  }
}

std::vector<double> MCCFR::sample_regret_estimate(int traverser) {
  std::vector<double> dreg(tree_.total_actions, 0.0), dsum(tree_.total_actions, 0.0);
  std::vector<int64_t> dupd(tree_.num_infosets(), 0);
  Buffers buf{&dreg, &dsum, &dupd, &rng_};
  if (cfg_.scheme == MCCFRScheme::kOutcome) {
    double tail = 1.0;
    traverse_os(0, traverser, 1.0, 1.0, 1.0, tail, buf);
  } else {
    traverse_es(0, traverser, 1.0, buf);
  }
  nodes_touched_ += buf.touched;
  return dreg;
}

void MCCFR::iterate_parallel(int batches, int traversals_per_thread, int threads) {
  AEGIS_CHECK_MSG(cfg_.scheme == MCCFRScheme::kExternal, "parallel MCCFR supports external sampling");
  AEGIS_CHECK(threads >= 1 && traversals_per_thread >= 1);
  AEGIS_CHECK_MSG(sampler_ == nullptr, "sampling-policy hooks are not thread-safe; detach before parallel runs");
  ThreadPool pool(threads);
  const int A = tree_.total_actions;
  std::vector<std::vector<double>> dreg(threads, std::vector<double>(A)), dsum(threads, std::vector<double>(A));
  std::vector<std::vector<int64_t>> dupd(threads, std::vector<int64_t>(tree_.num_infosets()));
  std::vector<int64_t> touched(threads, 0);
  std::vector<double> sigma(A);
  auto refresh_sigma = [&] {
    for (const auto& I : tree_.infosets) regret_matching(&regret_[I.offset], &sigma[I.offset], I.num_actions);
  };
  refresh_sigma();
  for (int bt = 0; bt < batches; ++bt) {
    ++t_;
    for (int i = 0; i < 2; ++i) {
      snapshot_sigma_ = &sigma;
      pool.parallel_for(threads, [&](int th) {
        std::fill(dreg[th].begin(), dreg[th].end(), 0.0);
        std::fill(dsum[th].begin(), dsum[th].end(), 0.0);
        std::fill(dupd[th].begin(), dupd[th].end(), 0);
        uint64_t s = cfg_.seed ^ (0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(t_) * 2 + i + 1)) ^
                     (0xD1B54A32D192ED03ULL * (th + 1));
        Rng rng(splitmix64(s));
        Buffers buf{&dreg[th], &dsum[th], &dupd[th], &rng};
        for (int k = 0; k < traversals_per_thread; ++k) traverse_es(0, i, 1.0, buf);
        touched[th] = buf.touched;
      });
      snapshot_sigma_ = nullptr;
      // Deterministic reduction in thread order.
      for (int th = 0; th < threads; ++th) {
        nodes_touched_ += touched[th];
        for (int s = 0; s < A; ++s) {
          regret_[s] += dreg[th][s];
          strategy_sum_[s] += dsum[th][s];
        }
        for (int I = 0; I < tree_.num_infosets(); ++I) updates_[I] += dupd[th][I];
      }
      if (cfg_.regret_matching_plus)
        for (auto& r : regret_) r = std::max(r, 0.0);
      refresh_sigma();
    }
  }
}

TabularPolicy MCCFR::average_policy() const { return TabularPolicy::from_weights(tree_, strategy_sum_); }

TabularPolicy MCCFR::current_policy() const {
  TabularPolicy p(tree_);
  for (const auto& I : tree_.infosets) regret_matching(&regret_[I.offset], &p.probs[I.offset], I.num_actions);
  return p;
}

}  // namespace aegis
