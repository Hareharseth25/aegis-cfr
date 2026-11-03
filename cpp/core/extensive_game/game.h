// The extensive-form imperfect-information game G = (N, H, Z, P, I, A, σ_c, u).
//
//  * A State is a history h ∈ H. Terminal histories (Z) report utilities u.
//  * current_player() is P(h): a player index or kChancePlayer.
//  * chance_outcomes() gives σ_c(h, *) at chance histories.
//  * information_state_key(p) identifies the information set I ∈ I_p that
//    contains h from player p's point of view. Two histories with equal keys
//    MUST be strategically indistinguishable to p (same legal actions and, for
//    perfect recall, the same own-action history). GameTree verifies both.
//
// Games are two-player zero-sum unless stated otherwise; GameTree checks the
// zero-sum property explicitly.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace aegis {

using Action = int;
inline constexpr int kChancePlayer = -1;
inline constexpr int kTerminalPlayer = -2;

class State {
 public:
  virtual ~State() = default;

  virtual bool is_terminal() const = 0;
  bool is_chance() const { return current_player() == kChancePlayer; }
  virtual int current_player() const = 0;

  // Legal actions at a decision history, in a fixed canonical order.
  virtual std::vector<Action> legal_actions() const = 0;
  // (outcome, probability) pairs at a chance history. Probabilities sum to 1.
  virtual std::vector<std::pair<Action, double>> chance_outcomes() const = 0;

  virtual std::unique_ptr<State> child(Action a) const = 0;

  // Key of player p's information set at this history. Defined for every
  // non-terminal history and both players (the "augmented" information set
  // when p is not acting), which subgame re-solving needs.
  virtual std::string information_state_key(int player) const = 0;

  // Terminal utilities u_0(z), u_1(z).
  virtual std::array<double, 2> returns() const = 0;

  virtual std::string action_to_string(Action a) const = 0;
  virtual std::string to_string() const = 0;

  // Identifier of the public state (all information common to both players).
  // Used to group histories into subgames. Default: not supported.
  virtual std::string public_state_key() const { return ""; }
};

class Game {
 public:
  virtual ~Game() = default;
  virtual std::unique_ptr<State> initial_state() const = 0;
  virtual std::string name() const = 0;
  virtual int num_players() const { return 2; }
};

}  // namespace aegis
