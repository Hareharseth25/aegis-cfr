// A Hold'em turn/river subgame expressed through the generic extensive-form
// Game interface, so the SAME tabular CFR and exact best-response code used
// for Kuhn and Leduc can solve it.
//
// Chance first deals the private hand pair (h0, h1) with probability
// prop. to r_0(h0) r_1(h1) [h0 ∩ h1 = empty], then betting follows the PublicTree; turn
// trees deal the river uniformly among cards not on the board or in either
// hand. Player p's information set at public node n is (n, h_p).
//
// This is exponentially less efficient than the vector-form solver (it
// enumerates hand pairs), which is the point: on small ranges both must give
// identical strategies, values and exploitability.
#pragma once

#include <memory>

#include "core/extensive_game/game.h"
#include "solvers/subgame_resolver/subgame_context.h"

namespace aegis {

class HoldemSubgameGame final : public Game {
 public:
  // The context must outlive the game.
  explicit HoldemSubgameGame(const poker::SubgameContext& ctx);
  std::unique_ptr<State> initial_state() const override;
  std::string name() const override { return "holdem_subgame_efg"; }

  const poker::SubgameContext& context() const { return ctx_; }
  // Information-set key used for (public node, local hand of player p).
  static std::string key(int node, int hand) { return "n" + std::to_string(node) + "h" + std::to_string(hand); }

  struct Pair {
    int h0, h1;
    double p;
  };
  const std::vector<Pair>& pairs() const { return pairs_; }

 private:
  const poker::SubgameContext& ctx_;
  std::vector<Pair> pairs_;
};

class HoldemSubgameState final : public State {
 public:
  explicit HoldemSubgameState(const HoldemSubgameGame& g) : g_(&g) {}
  bool is_terminal() const override;
  int current_player() const override;
  std::vector<Action> legal_actions() const override;
  std::vector<std::pair<Action, double>> chance_outcomes() const override;
  std::unique_ptr<State> child(Action a) const override;
  std::string information_state_key(int player) const override;
  std::array<double, 2> returns() const override;
  std::string action_to_string(Action a) const override;
  std::string to_string() const override;
  std::string public_state_key() const override { return "n" + std::to_string(node_); }

 private:
  const HoldemSubgameGame* g_;
  int hand_[2] = {-1, -1};
  int node_ = -1;  // -1 = before the deal
};

}  // namespace aegis
