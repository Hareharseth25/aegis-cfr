// Uncertainty-aware computation budgets across many subgames.
//
// Setting: K independent subgames (e.g. the river subgames that continual
// re-solving must answer), subgame k reached with weight w_k, and a fixed
// total work budget B measured in hand-updates (Σ over node visits of the
// hands processed - a hardware-independent cost). Objective:
//     minimise Σ_k w_k * expl_k(work_k)  subject to Σ_k work_k <= B.
//
// Policies:
//  * kUniform  - equal work per subgame;
//  * kWeighted - work prop. to w_k;
//  * kAdaptive - warm up every subgame, fit expl_k(t) ~ a_k t^{-b_k} by
//    log-log regression on exact exploitability checkpoints, then greedily
//    give the next chunk to the subgame with the largest predicted
//    w_k * (expl_k(t) - expl_k(t+Δ)) per unit of work, inflated by
//    exp(κ σ_k) where σ_k is the fit's residual std (optimism under
//    uncertainty). The cost of every exploitability measurement is charged
//    to the budget.
#pragma once

#include <vector>

#include "solvers/subgame_resolver/vector_cfr.h"

namespace aegis::poker {

enum class BudgetPolicy { kUniform, kWeighted, kAdaptive };

struct BudgetSubgame {
  const SubgameContext* ctx = nullptr;
  double weight = 1.0;
};

struct BudgetOptions {
  double total_work = 1e9;     // hand-updates
  int warmup_iterations = 8;
  int chunk_iterations = 8;
  int refit_every = 2;         // chunks between re-measurements of a subgame
  double ucb_kappa = 1.0;
  CFRConfig cfr = CFRConfig::dcfr();
  bool charge_measurements = true;
};

struct BudgetResult {
  std::vector<int> iterations;
  std::vector<double> exploitability;  // chips, per subgame
  double weighted_exploitability = 0.0;
  double work_used = 0.0;
  int measurements = 0;
  double seconds = 0.0;
  // (work used, weighted exploitability estimate) trace for plotting.
  std::vector<std::pair<double, double>> trace;
};

BudgetResult allocate_budget(const std::vector<BudgetSubgame>& games, BudgetPolicy policy, const BudgetOptions& opts);

}  // namespace aegis::poker
