// Kuhn poker: 3-card deck (J=0, Q=1, K=2), ante 1, a single bet of 1.
// Encodings match OpenSpiel's kuhn_poker so the two implementations can be
// compared node-for-node: chance deals player 0's card then player 1's card;
// actions are Pass = 0, Bet = 1.
#pragma once

#include "core/extensive_game/game.h"

namespace aegis {

class KuhnGame final : public Game {
 public:
  std::unique_ptr<State> initial_state() const override;
  std::string name() const override { return "kuhn_poker"; }

  // Known closed-form equilibrium value for player 0: -1/18.
  static constexpr double kGameValueP0 = -1.0 / 18.0;
};

class KuhnState final : public State {
 public:
  KuhnState() = default;
  bool is_terminal() const override;
  int current_player() const override;
  std::vector<Action> legal_actions() const override;
  std::vector<std::pair<Action, double>> chance_outcomes() const override;
  std::unique_ptr<State> child(Action a) const override;
  std::string information_state_key(int player) const override;
  std::array<double, 2> returns() const override;
  std::string action_to_string(Action a) const override;
  std::string to_string() const override;
  std::string public_state_key() const override;

 private:
  int cards_[2] = {-1, -1};
  std::vector<Action> bets_;  // betting history (0 = pass, 1 = bet)
};

}  // namespace aegis
