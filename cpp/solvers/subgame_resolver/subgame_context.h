// Shared machinery for vector-form ("range") solving of Hold'em subgames.
//
// In a heads-up poker subgame a player's information set at public node n is
// the pair (n, private hand). Instead of materialising every hand pair we keep,
// for every public node, one value per private hand of each player (a
// "range vector") and evaluate terminals in O(n) with card removal:
//
//   fold:     cfv_p(h) = u_p * Σ_{h' ∩ h = empty} r_o(h')
//             = u_p * (R - R[c1] - R[c2] + r_o(h))      (inclusion-exclusion)
//   showdown: cfv_p(h) = C * (Σ_{h' < h, disjoint} r_o(h') - Σ_{h' > h, disjoint} r_o(h'))
//             via one sweep over hands sorted by strength with 52 per-card
//             running sums (cached sort order per board).
//   chance (turn -> river): cfv_p(h) = (1/44) Σ_{c ∉ h} cfv_p^c(h); in each
//             river child the hands containing c are impossible.
//
// Counterfactual values are weighted by the opponent's reach (which includes
// the opponent's range prior) - never by the acting player's own reach. For a
// player p the expected value is EV_p = Σ_h r_p(h) cfv_p(h) / Z with
// Z = Σ_{h, h' disjoint} r_0(h) r_1(h').
#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>

#include "poker/betting_rules/betting.h"
#include "poker/hand_evaluator/evaluator.h"
#include "poker/ranges/range.h"
#include "util/check.h"
#include "util/thread_pool.h"

namespace aegis::poker {

struct HandSpace {
  std::vector<int> combos;                // combo index of each local hand
  std::vector<std::array<Card, 2>> cards;
  std::vector<double> root_weight;        // prior weight (range) at the root
  std::vector<int> same_in_opponent;      // local index of the same combo in the opponent's list, or -1
  std::vector<int> local_of_combo;        // combo -> local index or -1 (size 1326)
  int size() const { return static_cast<int>(combos.size()); }
};

// A behaviour strategy for one player-agnostic public tree: for each
// decision node (by decision_index) an action-major array σ[a * n_p + i].
struct VectorPolicy {
  std::vector<std::vector<double>> strategy;
  // Probability of action a at decision node `node` for local hand i.
  double prob(const PublicTree& t, const HandSpace hs[2], int node, int a, int i) const;
};

class SubgameContext {
 public:
  // Hands with positive weight that do not intersect the root board make up
  // each player's local hand list. If `full_support[p]` is set, player p's
  // list includes every live hand (needed for re-solving constraints).
  // `support[p]`, when given, adds every combo with positive support weight to
  // player p's list even if its range weight is zero (values for currently
  // unreachable hands are still needed by counterfactual regrets).
  SubgameContext(const PublicTree& tree, const Range& r0, const Range& r1, int threads = 1,
                 std::array<bool, 2> full_support = {false, false},
                 std::array<const Range*, 2> support = {nullptr, nullptr});

  const PublicTree& tree() const { return tree_; }
  const HandSpace& hands(int p) const {
    AEGIS_CHECK_MSG(p == 0 || p == 1, "player must be 0 or 1, got " << p);
    return hands_[p];
  }
  const HandSpace* hand_spaces() const { return hands_; }
  int n(int p) const { return hands(p).size(); }
  double normalizer() const { return Z_; }  // Σ compatible root pair mass
  ThreadPool* pool() const { return pool_.get(); }
  int threads() const { return threads_; }

  // Is local hand i of player p possible on the board of public node `node`?
  bool valid(int node, int p, int i) const {
    AEGIS_CHECK(node >= 0 && node < tree_.num_nodes() && (p == 0 || p == 1) && i >= 0 && i < hands_[p].size());
    return valid_[p][tree_.nodes[node].board_id][i] != 0;
  }
  const uint8_t* valid_mask(int board_id, int p) const { return valid_[p][board_id].data(); }

  // Terminal kernels (out has n(p) entries; invalid hands get 0).
  void fold_cfv(int node, int p, const double* opp_reach, double* out) const;
  void showdown_cfv(int node, int p, const double* opp_reach, double* out) const;
  // Weight of each river child at a turn chance node (1 / #cards possible
  // given both hands and the board).
  double chance_weight(int node) const;

  // Uniform policy (every action equally likely) - useful as a baseline.
  VectorPolicy uniform_policy() const;
  int num_actions(int node) const { return tree_.nodes[node].num_children; }

  // ---- Policy evaluation ---------------------------------------------------
  struct Values {
    double ev[2] = {0, 0};              // expected value per player (chips)
    std::vector<double> root_cfv[2];    // per local hand
  };
  // Expected values when both players follow `policy`.
  Values evaluate(const VectorPolicy& policy) const;
  // Best response of player p against the opponent's part of `policy`:
  // returns BR value and BR counterfactual values per local hand at the root.
  Values best_response(const VectorPolicy& policy, int p) const;

  struct Exploitability {
    double br_value[2] = {0, 0};
    double policy_value[2] = {0, 0};
    double nash_conv = 0.0;
    double exploitability = 0.0;          // nash_conv / 2
    double exploitability_pct_pot = 0.0;  // 100 * exploitability / pot
  };
  Exploitability exploitability(const VectorPolicy& policy) const;

  // Counterfactual values at an arbitrary node for given reach vectors at that
  // node, under `policy` (mode: 0 = follow policy for both; 1 = player p best
  // responds). Used for blueprint values and re-solving constraints.
  void subtree_values(const VectorPolicy& policy, int node, const std::vector<double> reach[2], int br_player,
                      std::vector<double> out[2]) const;

  // Reach vectors of both players at every node under `policy` from the given
  // root reaches (defaults to the range priors).
  std::vector<std::vector<double>> node_reach(const VectorPolicy& policy, int p,
                                              const std::vector<double>* root = nullptr) const;

 private:
  void check_policy(const VectorPolicy& policy) const;
  void build_hands(const Range& r0, const Range& r1, std::array<bool, 2> full, std::array<const Range*, 2> support);
  void build_showdowns();

  // Generic recursive passes used by evaluation (not by the solver, which
  // has its own fused update pass).
  void forward(const VectorPolicy& pol, int node, std::vector<std::vector<double>>& reach, int p) const;
  void backward(const VectorPolicy& pol, int node, int p, int br_player,
                const std::vector<std::vector<double>>& opp_reach, std::vector<std::vector<double>>& cfv) const;

  const PublicTree& tree_;
  HandSpace hands_[2];
  double Z_ = 0.0;
  int threads_ = 1;
  std::unique_ptr<ThreadPool> pool_;

  // Per board id and player: validity mask over local hands.
  std::vector<std::vector<uint8_t>> valid_[2];
  // Per 5-card board id: hand strengths and ascending order of valid hands.
  struct Showdown {
    bool ready = false;
    std::vector<HandValue> value[2];
    std::vector<int> order[2];
  };
  std::vector<Showdown> showdown_;
  friend class VectorCFR;
};

}  // namespace aegis::poker
