#include "adaptive/refinement/action_refinement.h"

#include <algorithm>
#include <cmath>

#include "solvers/subgame_resolver/resolver.h"
#include "util/check.h"
#include "util/timer.h"

namespace aegis::poker {

ActionRefiner::ActionRefiner(RefinementSpot spot) : spot_(std::move(spot)) {
  AEGIS_CHECK_MSG(spot_.base.board.size() == 5, "action refinement operates on river spots");
  AEGIS_CHECK(!spot_.reference_bets.empty());
  std::sort(spot_.reference_bets.begin(), spot_.reference_bets.end());
  ref_tree_ = PublicTree::build(with_bets({spot_.reference_bets, spot_.reference_bets}));
  ref_ctx_ = std::make_unique<SubgameContext>(ref_tree_, spot_.r0, spot_.r1, spot_.threads);
}

TreeConfig ActionRefiner::with_bets(const std::array<std::vector<double>, 2>& bets) const {
  TreeConfig c = spot_.base;
  for (int p = 0; p < 2; ++p) c.sizes[1][p].bet_fractions = bets[p];
  return c;
}

AbstractionEval ActionRefiner::evaluate(const std::array<std::vector<double>, 2>& bets, bool compute_candidates) {
  AbstractionEval ev;
  ev.bets = bets;
  for (int p = 0; p < 2; ++p) {
    for (double b : bets[p]) {
      bool in_grid = false;
      for (double r : spot_.reference_bets) in_grid |= std::abs(r - b) < 1e-9;
      AEGIS_CHECK_MSG(in_grid, "abstract size " << b << " is not in the reference grid");
    }
  }
  const PublicTree abs_tree = PublicTree::build(with_bets(bets));
  const SubgameContext abs_ctx(abs_tree, spot_.r0, spot_.r1, spot_.threads);
  Stopwatch sw;
  VectorCFR solver(abs_ctx, {spot_.cfr});
  solver.iterate(spot_.iterations);
  const auto abs_pol = solver.average_policy();
  ev.solve_seconds = sw.seconds();
  sw.reset();
  ev.abs_nodes = abs_tree.num_nodes();
  for (const auto& n : abs_tree.nodes)
    if (n.type == PNodeType::kDecision) ev.abs_infosets += abs_ctx.n(n.player);
  ev.abs_exploitability_pct = abs_ctx.exploitability(abs_pol).exploitability_pct_pot;
  const auto ref_pol = translate_policy(*ref_ctx_, abs_ctx, abs_pol, spot_.translation);
  const auto rex = ref_ctx_->exploitability(ref_pol);
  ev.ref_exploitability_pct = rex.exploitability_pct_pot;
  ev.ref_nash_conv = rex.nash_conv;
  ev.ref_ev0 = rex.policy_value[0];
  if (compute_candidates) {
    const auto& t = ref_tree_;
    const double Z = ref_ctx_->normalizer();
    for (int d = 0; d < t.num_nodes(); ++d) {
      const auto& nd = t.nodes[d];
      if (nd.type != PNodeType::kDecision || nd.street != 1) continue;
      if (std::abs(nd.street_bet[0] - nd.street_bet[1]) > 1e-9) continue;  // facing a bet
      const int p = nd.player;
      std::vector<int> on, off;
      for (int k = 0; k < nd.num_children; ++k) {
        const int c = nd.first_child + k;
        const auto& a = t.nodes[c].action;
        bool in_abs = a.kind == ActionKind::kCheck || (a.kind == ActionKind::kAllIn && spot_.base.sizes[1][p].allow_all_in);
        if (a.kind == ActionKind::kBet)
          for (double b : bets[p]) in_abs |= std::abs(a.nominal - b) < 1e-9;
        (in_abs ? on : off).push_back(c);
      }
      if (off.empty() || on.empty()) continue;
      std::vector<double> reach[2];
      reach_at(*ref_ctx_, ref_pol, d, reach);
      double prior = 0.0, mass = 0.0;
      for (int i = 0; i < ref_ctx_->n(p); ++i) {
        prior += ref_ctx_->hands(p).root_weight[i];
        mass += reach[p][i];
      }
      std::vector<double> best_on(ref_ctx_->n(p), -1e300);
      for (int c : on) {
        const auto v = opponent_constraint(*ref_ctx_, ref_pol, c, p, true);
        for (int i = 0; i < ref_ctx_->n(p); ++i) best_on[i] = std::max(best_on[i], v[i]);
      }
      std::vector<double> abs_sizes = bets[p];
      std::sort(abs_sizes.begin(), abs_sizes.end());
      for (int c : off) {
        const auto& a = t.nodes[c].action;
        const auto v = opponent_constraint(*ref_ctx_, ref_pol, c, p, true);
        double hole = 0.0;
        for (int i = 0; i < ref_ctx_->n(p); ++i) hole += reach[p][i] * std::max(0.0, v[i] - best_on[i]);
        CandidateInfo ci;
        ci.player = p;
        ci.node = d;
        ci.fraction = a.nominal;
        ci.hole = hole / Z;
        ci.hole_pct_pot = 100.0 * ci.hole / spot_.base.pot;
        ci.reach_mass = prior > 0 ? mass / prior : 0.0;
        double lo = -1, hi = -1;
        for (double s : abs_sizes) {
          if (s < a.nominal) lo = s;
          if (s > a.nominal && hi < 0) hi = s;
        }
        ci.gap_low = lo > 0 ? std::log(a.nominal / lo) : std::log(a.nominal / spot_.reference_bets.front()) + 1.0;
        ci.gap_high = hi > 0 ? std::log(hi / a.nominal) : 1.0;
        ev.candidates.push_back(ci);
      }
    }
  }
  ev.eval_seconds = sw.seconds();
  return ev;
}

}  // namespace aegis::poker
