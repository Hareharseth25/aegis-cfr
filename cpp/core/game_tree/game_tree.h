// A fully materialised game tree in flat, contiguous arrays.
//
// Every history becomes a node id (depth-first pre-order, so a parent's id is
// always smaller than its children's). Regrets, strategy sums and policies are
// indexed by a global "action slot" = infoset.offset + action_index, which
// gives contiguous per-information-set storage instead of an object per node.
//
// GameTree::build validates the game as it materialises it:
//  * all histories in an information set have identical legal actions;
//  * perfect recall: all histories in an information set share the acting
//    player's own sequence of (information set, action) pairs;
//  * chance probabilities are non-negative and sum to one;
//  * terminal utilities are zero-sum (recorded in `zero_sum`).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/extensive_game/game.h"

namespace aegis {

enum class NodeType : uint8_t { kTerminal = 0, kChance = 1, kDecision = 2 };

struct InfoSetInfo {
  int player = -1;
  int num_actions = 0;
  int offset = 0;  // first global action slot
  int depth = 0;   // depth of the first history (all histories share it)
  std::string key;
  std::vector<Action> actions;
  std::vector<std::string> action_names;
  std::vector<int> nodes;  // histories h ∈ I
  // Perfect-recall parent sequence: global action slot of this player's
  // previous (infoset, action), or -1 for the empty sequence.
  int parent_sequence = -1;
};

class GameTree {
 public:
  struct BuildOptions {
    bool check_perfect_recall = true;
    int64_t max_nodes = 50'000'000;
  };

  static GameTree build(const Game& game, BuildOptions opts);
  static GameTree build(const Game& game) { return build(game, BuildOptions{}); }

  int num_nodes() const { return static_cast<int>(type.size()); }
  int num_infosets() const { return static_cast<int>(infosets.size()); }
  int num_terminals() const { return num_terminals_; }

  int infoset_index(int player, const std::string& key) const;  // -1 if absent

  // Node arrays (structure of arrays).
  std::vector<NodeType> type;
  std::vector<int8_t> player;        // acting player, kChancePlayer or kTerminalPlayer
  std::vector<int32_t> infoset;      // decision nodes only, else -1
  std::vector<int32_t> child_begin;  // into children/child_action/chance_prob
  std::vector<int32_t> child_count;
  std::vector<int32_t> parent;       // -1 for the root
  std::vector<int32_t> depth;
  std::vector<double> utility[2];    // terminal utilities (0 for non-terminals)

  // Edge arrays.
  std::vector<int32_t> children;
  std::vector<Action> child_action;
  std::vector<double> chance_prob;   // chance edges; 0 elsewhere

  // Augmented information sets (defined at every non-terminal node for both
  // players) and public states, interned to integer ids. Used by subgame
  // re-solving to group root histories.
  std::vector<int32_t> aug_infoset[2];
  std::vector<std::string> aug_keys[2];
  std::vector<int32_t> public_state;
  std::vector<std::string> public_keys;

  // Sequence-form data: for every node, each player's last own action slot on
  // the path (-1 = empty sequence). Supports LP solving and realisation plans.
  std::vector<int32_t> sequence[2];

  std::vector<InfoSetInfo> infosets;
  int total_actions = 0;
  bool zero_sum = true;
  std::string game_name;

  // Slot -> (infoset, action index) lookup.
  std::vector<int32_t> slot_infoset;

 private:
  int num_terminals_ = 0;
  std::unordered_map<std::string, int> infoset_lookup_[2];
  friend class GameTreeBuilder;
};

}  // namespace aegis
