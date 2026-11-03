// Leduc hold'em (Southey et al. 2005), with parameters for larger variants.
//
// Default rules match OpenSpiel's leduc_poker exactly so the implementations
// can be cross-checked node-for-node:
//  * deck of num_ranks x num_suits cards; card id c has rank c / num_suits
//    (default 3 ranks x 2 suits: J J Q Q K K)
//  * ante 1 each; one private card each; betting round 1 (raise size 2);
//    one public card; betting round 2 (raise size 4)
//  * at most 2 raises per round; player 0 acts first in both rounds
//  * fold is only legal when facing a bet; Fold = 0, Call/Check = 1, Raise = 2
//  * showdown: pairing the public card wins, otherwise the higher private rank
//    (OpenSpiel compares (high, low) of private+public, which is equivalent);
//    equal ranks split the pot
#pragma once

#include "core/extensive_game/game.h"

namespace aegis {

struct LeducConfig {
  int num_ranks = 3;
  int num_suits = 2;
  double ante = 1.0;
  double raise_round1 = 2.0;
  double raise_round2 = 4.0;
  int max_raises = 2;
};

class LeducGame final : public Game {
 public:
  explicit LeducGame(LeducConfig cfg = {});
  std::unique_ptr<State> initial_state() const override;
  std::string name() const override;
  const LeducConfig& config() const { return cfg_; }
  int deck_size() const { return cfg_.num_ranks * cfg_.num_suits; }

 private:
  LeducConfig cfg_;
};

class LeducState final : public State {
 public:
  explicit LeducState(const LeducConfig& cfg);
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

  static constexpr Action kFold = 0;
  static constexpr Action kCall = 1;
  static constexpr Action kRaise = 2;

 private:
  bool card_used(int c) const;
  int hand_rank(int player) const;
  std::string history_string() const;

  const LeducConfig* cfg_;
  int private_[2] = {-1, -1};
  int public_ = -1;
  int round_ = 1;
  int player_ = kChancePlayer;
  double contrib_[2];
  double stakes_;
  int raises_ = 0;
  bool folded_[2] = {false, false};
  bool finished_ = false;
  std::vector<Action> hist_[2];  // betting history per round
};

}  // namespace aegis
