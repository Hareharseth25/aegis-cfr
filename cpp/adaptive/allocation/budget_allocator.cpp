#include "adaptive/allocation/budget_allocator.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "util/check.h"
#include "util/timer.h"

namespace aegis::poker {

namespace {

struct Game {
  const SubgameContext* ctx;
  double w;
  std::unique_ptr<VectorCFR> solver;
  double work_per_iter;
  std::vector<std::pair<double, double>> points;  // (log t, log expl)
  double a = 1.0, b = 0.5, sigma = 0.5;
  int chunks_since_fit = 0;
  double last_expl = 0.0;
};

double measure(Game& g) {
  const double e = g.ctx->exploitability(g.solver->average_policy()).exploitability;
  g.last_expl = std::max(e, 1e-12);
  g.points.emplace_back(std::log(std::max(1, g.solver->iteration())), std::log(g.last_expl));
  return g.last_expl;
}

void fit(Game& g) {
  const auto& P = g.points;
  const int n = static_cast<int>(P.size());
  if (n < 2) {
    g.a = g.last_expl * std::sqrt(std::max(1, g.solver->iteration()));
    g.b = 0.5;
    g.sigma = 0.5;
    return;
  }
  // Least squares on the most recent points (convergence rates drift).
  const int start = std::max(0, n - 5);
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const int m = n - start;
  for (int k = start; k < n; ++k) {
    sx += P[k].first;
    sy += P[k].second;
    sxx += P[k].first * P[k].first;
    sxy += P[k].first * P[k].second;
  }
  const double den = m * sxx - sx * sx;
  double slope = den > 1e-12 ? (m * sxy - sx * sy) / den : -0.5;
  slope = std::clamp(slope, -2.0, -0.05);
  const double icpt = (sy - slope * sx) / m;
  g.b = -slope;
  g.a = std::exp(icpt);
  double ss = 0.0;
  for (int k = start; k < n; ++k) {
    const double r = P[k].second - (icpt + slope * P[k].first);
    ss += r * r;
  }
  g.sigma = m > 2 ? std::sqrt(ss / (m - 2)) : 0.5;
}

double predicted(const Game& g, double t) { return g.a * std::pow(std::max(1.0, t), -g.b); }

}  // namespace

BudgetResult allocate_budget(const std::vector<BudgetSubgame>& in, BudgetPolicy policy, const BudgetOptions& o) {
  Stopwatch sw;
  AEGIS_CHECK_MSG(!in.empty(), "no subgames to allocate to");
  AEGIS_CHECK_MSG(std::isfinite(o.total_work) && o.total_work >= 0.0, "total_work must be a non-negative number");
  AEGIS_CHECK_MSG(o.warmup_iterations >= 0, "warmup_iterations must be >= 0");
  AEGIS_CHECK_MSG(o.chunk_iterations >= 1, "chunk_iterations must be >= 1");
  AEGIS_CHECK_MSG(o.refit_every >= 1, "refit_every must be >= 1");
  std::vector<Game> games;
  double wsum = 0.0;
  for (const auto& s : in) {
    AEGIS_CHECK_MSG(s.ctx != nullptr, "null subgame context");
    AEGIS_CHECK_MSG(std::isfinite(s.weight) && s.weight >= 0.0, "subgame weights must be non-negative");
    Game g;
    g.ctx = s.ctx;
    g.w = s.weight;
    g.solver = std::make_unique<VectorCFR>(*s.ctx, VectorCFRConfig{o.cfr});
    g.work_per_iter = 2.0 * s.ctx->tree().num_nodes() * (s.ctx->n(0) + s.ctx->n(1));
    wsum += s.weight;
    games.push_back(std::move(g));
  }
  AEGIS_CHECK_MSG(wsum > 0.0, "at least one subgame needs positive weight");
  BudgetResult res;
  double used = 0.0;
  // Every unit of work is admitted only if it fits in what is left of the
  // budget, so work_used <= total_work on every path. A subgame may end up
  // with zero iterations if the budget is too small for it.
  auto fits = [&](double cost) { return used + cost <= o.total_work * (1.0 + 1e-12); };
  auto run = [&](Game& g, int iters) {
    g.solver->iterate(iters);
    used += iters * g.work_per_iter;
  };
  // An exploitability measurement costs about two iterations of work.
  auto measure_cost = [&](const Game& g) { return o.charge_measurements ? 2.0 * g.work_per_iter : 0.0; };
  auto charge_measure = [&](Game& g) {
    measure(g);
    res.measurements++;
    used += measure_cost(g);
  };
  auto weighted_estimate = [&] {
    double s = 0.0;
    for (const auto& g : games) s += g.w * (g.points.empty() ? 0.0 : std::exp(g.points.back().second));
    return s;
  };

  if (policy != BudgetPolicy::kAdaptive) {
    double total_w = 0.0;
    for (const auto& g : games) total_w += policy == BudgetPolicy::kUniform ? 1.0 : g.w;
    for (auto& g : games) {
      const double share = (policy == BudgetPolicy::kUniform ? 1.0 : g.w) / total_w;
      // floor() keeps each subgame inside its share, so the total fits too.
      const int iters = static_cast<int>(std::floor(share * o.total_work / g.work_per_iter));
      if (iters > 0) run(g, iters);
    }
  } else {
    // Warm-up, highest-weight subgames first; skip any whose warm-up (with its
    // measurements) no longer fits. Skipped subgames keep zero iterations.
    std::vector<int> order(games.size());
    for (size_t k = 0; k < games.size(); ++k) order[k] = static_cast<int>(k);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return games[a].w > games[b].w; });
    std::vector<uint8_t> warm(games.size(), 0);
    const int w1 = o.warmup_iterations / 2, w2 = o.warmup_iterations - w1;
    for (int k : order) {
      auto& g = games[k];
      if (o.warmup_iterations == 0) break;
      const int measurements = (w1 > 0 ? 1 : 0) + 1;
      const double cost = o.warmup_iterations * g.work_per_iter + measurements * measure_cost(g);
      if (!fits(cost)) continue;
      if (w1 > 0) {
        run(g, w1);
        charge_measure(g);
      }
      run(g, w2);
      charge_measure(g);
      fit(g);
      warm[k] = 1;
    }
    res.trace.emplace_back(used, weighted_estimate());
    while (true) {
      int best = -1;
      double best_gain = 0.0;
      for (size_t k = 0; k < games.size(); ++k) {
        if (!warm[k]) continue;
        auto& g = games[k];
        const double t = g.solver->iteration();
        const double chunk_work = o.chunk_iterations * g.work_per_iter;
        if (!fits(chunk_work)) continue;
        // Scale the fitted curve to pass through the latest measurement.
        const double scale = g.last_expl / predicted(g, t);
        const double gain = g.w * scale * (predicted(g, t) - predicted(g, t + o.chunk_iterations)) *
                            std::exp(o.ucb_kappa * g.sigma) / chunk_work;
        if (gain > best_gain) {
          best_gain = gain;
          best = static_cast<int>(k);
        }
      }
      if (best < 0) break;
      auto& g = games[best];
      run(g, o.chunk_iterations);
      const bool due = ++g.chunks_since_fit >= o.refit_every;
      if (due && fits(measure_cost(g))) {
        g.chunks_since_fit = 0;
        charge_measure(g);
        fit(g);
        res.trace.emplace_back(used, weighted_estimate());
      } else {
        // No measurement (not due yet, or it would not fit): advance the
        // estimate along the fitted curve instead.
        const double t = g.solver->iteration();
        g.last_expl = g.last_expl * predicted(g, t) / predicted(g, t - o.chunk_iterations);
      }
    }
  }
  // Final exact evaluation (not charged: it is the experiment's measurement).
  res.weighted_exploitability = 0.0;
  for (auto& g : games) {
    const double e = g.ctx->exploitability(g.solver->average_policy()).exploitability;
    res.iterations.push_back(g.solver->iteration());
    res.exploitability.push_back(e);
    res.weighted_exploitability += g.w * e;
  }
  res.work_used = used;
  res.trace.emplace_back(used, res.weighted_exploitability);
  res.seconds = sw.seconds();
  return res;
}

}  // namespace aegis::poker
