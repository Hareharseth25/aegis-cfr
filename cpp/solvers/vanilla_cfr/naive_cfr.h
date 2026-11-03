// Reference "textbook" CFR with object-per-node storage: it walks State
// objects directly and keeps a heap-allocated node per information set in a
// hash map keyed by the information-state string.
//
// It exists for two reasons:
//  1. as an independent implementation for cross-checking TabularCFR
//     (identical iterates are asserted in tests);
//  2. as the baseline in the memory-layout benchmark (flat contiguous arrays
//     vs. object-per-node + string hashing).
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/extensive_game/game.h"

namespace aegis {

class NaiveCFR {
 public:
  explicit NaiveCFR(const Game& game) : game_(game) {}

  // Vanilla CFR with alternating updates (same convention as TabularCFR).
  void iterate(int n = 1);

  // Average strategy for an information set key of player p.
  std::vector<double> average_strategy(int p, const std::string& key) const;
  size_t num_infosets() const { return nodes_.size(); }
  long long nodes_touched() const { return touched_; }

 private:
  struct Node {
    std::vector<double> regret, inst, sum, sigma;
    int player;
  };
  double walk(const State& s, int i, double r0, double r1, double rc);
  Node& node(const State& s, int p);

  const Game& game_;
  std::unordered_map<std::string, std::unique_ptr<Node>> nodes_;
  int t_ = 0;
  long long touched_ = 0;
};

}  // namespace aegis
