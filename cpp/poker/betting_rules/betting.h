// Heads-up No-Limit Hold'em betting rules and public-tree construction for
// turn and river subgames.
//
// Chip accounting (all amounts in big blinds):
//  * At the subgame root the pot is P and both players have contributed P/2.
//  * invested[p] is what p has added since the root; commit[p] = P/2 + invested[p].
//  * Zero-sum utilities: a fold by f pays the other player commit[f] (f loses
//    commit[f]); at showdown (equal commits C) the winner gets +C, the loser
//    -C, a tie 0.
//
// Bet sizing (pot fractions):
//  * bet of fraction f into pot_now:         amount = f * pot_now
//  * raise of fraction f facing to_call:     raise_to = opp_street_bet + f * (pot_now + to_call)
//    i.e. call first, then add f times the resulting pot.
//  * min bet = min_bet (1 BB); min raise increment = max(last raise increment, min_bet);
//    a size at or above the remaining stack becomes all-in; duplicates merge.
//  * max_raises bounds the number of raises after the first bet on a street.
#pragma once

#include <string>
#include <vector>

#include "poker/cards/cards.h"
#include "util/check.h"

namespace aegis::poker {

struct StreetSizes {
  std::vector<double> bet_fractions{0.33, 0.75, 1.5};
  std::vector<double> raise_fractions{1.0};
  bool allow_all_in = true;
  int max_raises = 3;
};

struct TreeConfig {
  std::vector<Card> board;        // 4 cards (turn subgame) or 5 (river)
  double pot = 100.0;             // pot at the subgame root
  double effective_stack = 150.0; // remaining stack of each player at the root
  int first_player = 0;           // player acting first on every street (OOP)
  double min_bet = 1.0;
  // Per-player sizes on each street: [street][player], street 0 = turn, 1 = river.
  StreetSizes sizes[2][2];
  // Optional mid-street start (re-solving after actions already taken).
  double initial_street_bet[2] = {0.0, 0.0};
  int initial_player = -1;        // -1 -> first_player
  int initial_raises = 0;
  double initial_last_raise = 0.0;  // last raise increment (for min-raise)
  int initial_checks = 0;           // checks already made this street (0 or 1)
  bool depth_limited = false;     // turn tree: end-of-turn non-all-in nodes become leaves

  void set_sizes(const StreetSizes& turn, const StreetSizes& river) {
    sizes[0][0] = sizes[0][1] = turn;
    sizes[1][0] = sizes[1][1] = river;
  }
  void set_river_sizes(const StreetSizes& s) { sizes[1][0] = sizes[1][1] = s; }
};

enum class ActionKind : uint8_t { kFold, kCheck, kCall, kBet, kRaise, kAllIn, kDeal };

struct ActionLabel {
  ActionKind kind = ActionKind::kCheck;
  double amount = 0.0;    // bet/raise: actor's street total after the action; call: chips added
  double fraction = 0.0;  // bet/raise size as a pot fraction (see header comment)
  double nominal = -1.0;  // configured fraction that generated a bet/raise (-1 if none)
  Card card = -1;         // kDeal
  std::string to_string() const;
};

enum class PNodeType : uint8_t { kDecision, kFold, kShowdown, kChance, kLeaf };

struct PublicNode {
  PNodeType type = PNodeType::kDecision;
  int player = -1;        // decision: actor; fold: the player who folded
  int street = 0;         // 0 = turn, 1 = river
  int parent = -1;
  int first_child = -1;   // children are contiguous: [first_child, first_child + num_children)
  int num_children = 0;
  int board_id = 0;       // index into PublicTree::boards
  double commit[2] = {0, 0};
  double street_bet[2] = {0, 0};
  int raises = 0;
  int checks = 0;
  double last_raise = 0.0;
  ActionLabel action;     // edge from parent
  int depth = 0;
  int decision_index = -1;  // dense index among decision nodes
};

class PublicTree {
 public:
  static PublicTree build(const TreeConfig& cfg);

  const TreeConfig& config() const { return cfg_; }
  int num_nodes() const { return static_cast<int>(nodes.size()); }
  int num_decision_nodes() const { return num_decisions_; }
  // Checked node access (exposed to Python, so bad ids must not be UB).
  const PublicNode& node(int n) const {
    AEGIS_CHECK_MSG(n >= 0 && n < num_nodes(), "node id " << n << " out of range [0, " << num_nodes() << ")");
    return nodes[n];
  }
  double pot(int n) const { return node(n).commit[0] + node(n).commit[1]; }
  double stack_remaining(int n, int p) const {
    AEGIS_CHECK(p == 0 || p == 1);
    return cfg_.effective_stack - (node(n).commit[p] - cfg_.pot / 2.0);
  }
  // Path of action labels from the root to node n (for display).
  std::string path_string(int n) const;
  // Index of the child of `n` reached by action label string, or -1.
  int child_by_label(int n, const std::string& label) const;
  int count_type(PNodeType t) const;

  std::vector<PublicNode> nodes;
  std::vector<std::vector<Card>> boards;  // board_id -> cards (root board first)
  std::vector<CardMask> board_masks;

 private:
  void expand(int n);
  int add_child(int parent, const PublicNode& proto);
  std::vector<std::pair<ActionLabel, PublicNode>> decision_children(const PublicNode& nd) const;
  int board_for(int parent_board, Card c);

  TreeConfig cfg_;
  int num_decisions_ = 0;
};

}  // namespace aegis::poker
