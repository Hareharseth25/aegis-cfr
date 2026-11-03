// Coin Toss (Brown & Sandholm, "Safe and Nested Subgame Solving", 2017).
//
// A fair coin is flipped; only player 0 sees it. Player 0 chooses Sell or
// Play. Sell pays player 0 +sell_value on Heads and -sell_value on Tails.
// After Play, player 1 (who did not see the coin) guesses Heads or Tails:
// a correct guess pays player 0 -1, a wrong guess pays +1.
//
// In equilibrium player 1 guesses Heads with probability 1/4 and the game
// value is 0. The "Play" subgame is the canonical example in which solving a
// subgame from its local ranges alone (naive / unsafe re-solving) produces a
// strategy that is exploitable in the full game, because the subgame's
// equilibrium depends on what player 0 could have earned by selling.
#pragma once

#include "core/extensive_game/game.h"

namespace aegis {

class CoinTossGame final : public Game {
 public:
  explicit CoinTossGame(double sell_value = 0.5) : sell_value_(sell_value) {}
  std::unique_ptr<State> initial_state() const override;
  std::string name() const override { return "coin_toss"; }
  double sell_value() const { return sell_value_; }

 private:
  double sell_value_;
};

class CoinTossState final : public State {
 public:
  explicit CoinTossState(double sell_value) : sell_value_(sell_value) {}
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

  static constexpr Action kHeads = 0, kTails = 1;  // chance outcomes / guesses
  static constexpr Action kSell = 0, kPlay = 1;     // player 0's actions

 private:
  double sell_value_;
  int coin_ = -1;
  int p0_action_ = -1;
  int guess_ = -1;
};

}  // namespace aegis
