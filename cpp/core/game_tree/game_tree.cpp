#include "core/game_tree/game_tree.h"

#include <cmath>

#include "util/check.h"

namespace aegis {

class GameTreeBuilder {
 public:
  GameTreeBuilder(GameTree& t, GameTree::BuildOptions opts) : t_(t), opts_(opts) {}

  void run(const State& root) {
    int seq[2] = {-1, -1};
    visit(root, -1, 0, seq);
    // Edge arrays were appended in DFS order per node; nothing else to fix up.
  }

 private:
  int intern(std::unordered_map<std::string, int>& map, std::vector<std::string>& keys, const std::string& k) {
    auto it = map.find(k);
    if (it != map.end()) return it->second;
    const int id = static_cast<int>(keys.size());
    keys.push_back(k);
    map.emplace(k, id);
    return id;
  }

  int new_node(int parent, int depth) {
    AEGIS_CHECK_MSG(t_.num_nodes() < opts_.max_nodes, "game tree exceeds max_nodes=" << opts_.max_nodes);
    const int id = t_.num_nodes();
    t_.type.push_back(NodeType::kTerminal);
    t_.player.push_back(kTerminalPlayer);
    t_.infoset.push_back(-1);
    t_.child_begin.push_back(0);
    t_.child_count.push_back(0);
    t_.parent.push_back(parent);
    t_.depth.push_back(depth);
    t_.utility[0].push_back(0.0);
    t_.utility[1].push_back(0.0);
    t_.aug_infoset[0].push_back(-1);
    t_.aug_infoset[1].push_back(-1);
    t_.public_state.push_back(-1);
    t_.sequence[0].push_back(-1);
    t_.sequence[1].push_back(-1);
    return id;
  }

  int visit(const State& s, int parent, int depth, const int seq[2]) {
    const int id = new_node(parent, depth);
    t_.sequence[0][id] = seq[0];
    t_.sequence[1][id] = seq[1];

    if (s.is_terminal()) {
      const auto u = s.returns();
      t_.utility[0][id] = u[0];
      t_.utility[1][id] = u[1];
      if (std::abs(u[0] + u[1]) > 1e-9) t_.zero_sum = false;
      t_.num_terminals_++;
      return id;
    }

    for (int p = 0; p < 2; ++p)
      t_.aug_infoset[p][id] = intern(aug_map_[p], t_.aug_keys[p], s.information_state_key(p));
    t_.public_state[id] = intern(public_map_, t_.public_keys, s.public_state_key());

    const int cp = s.current_player();
    if (cp == kChancePlayer) {
      t_.type[id] = NodeType::kChance;
      t_.player[id] = kChancePlayer;
      const auto outcomes = s.chance_outcomes();
      AEGIS_CHECK_MSG(!outcomes.empty(), "chance node without outcomes: " << s.to_string());
      double total = 0.0;
      for (const auto& [a, p] : outcomes) {
        AEGIS_CHECK_MSG(p >= 0.0, "negative chance probability");
        total += p;
      }
      AEGIS_CHECK_MSG(std::abs(total - 1.0) < 1e-9, "chance probabilities sum to " << total);
      std::vector<int> kids;
      kids.reserve(outcomes.size());
      for (const auto& [a, p] : outcomes) {
        auto c = s.child(a);
        kids.push_back(visit(*c, id, depth + 1, seq));
      }
      t_.child_begin[id] = static_cast<int32_t>(t_.children.size());
      t_.child_count[id] = static_cast<int32_t>(kids.size());
      for (size_t i = 0; i < kids.size(); ++i) {
        t_.children.push_back(kids[i]);
        t_.child_action.push_back(outcomes[i].first);
        t_.chance_prob.push_back(outcomes[i].second);
      }
      return id;
    }

    AEGIS_CHECK_MSG(cp == 0 || cp == 1, "unsupported player id " << cp);
    t_.type[id] = NodeType::kDecision;
    t_.player[id] = static_cast<int8_t>(cp);
    const auto actions = s.legal_actions();
    AEGIS_CHECK_MSG(!actions.empty(), "decision node without actions: " << s.to_string());
    const std::string key = s.information_state_key(cp);

    int iset;
    auto it = t_.infoset_lookup_[cp].find(key);
    if (it == t_.infoset_lookup_[cp].end()) {
      iset = t_.num_infosets();
      InfoSetInfo info;
      info.player = cp;
      info.num_actions = static_cast<int>(actions.size());
      info.offset = t_.total_actions;
      info.depth = depth;
      info.key = key;
      info.actions = actions;
      for (Action a : actions) info.action_names.push_back(s.action_to_string(a));
      info.parent_sequence = seq[cp];
      t_.total_actions += info.num_actions;
      for (int k = 0; k < info.num_actions; ++k) t_.slot_infoset.push_back(iset);
      t_.infosets.push_back(std::move(info));
      t_.infoset_lookup_[cp].emplace(key, iset);
    } else {
      iset = it->second;
      const auto& info = t_.infosets[iset];
      AEGIS_CHECK_MSG(info.actions == actions,
                      "information set '" << key << "' has inconsistent legal actions");
      if (opts_.check_perfect_recall) {
        AEGIS_CHECK_MSG(info.parent_sequence == seq[cp],
                        "perfect recall violated at information set '" << key << "'");
      }
    }
    t_.infoset[id] = iset;
    t_.infosets[iset].nodes.push_back(id);

    const int offset = t_.infosets[iset].offset;
    std::vector<int> kids;
    kids.reserve(actions.size());
    for (size_t k = 0; k < actions.size(); ++k) {
      int child_seq[2] = {seq[0], seq[1]};
      child_seq[cp] = offset + static_cast<int>(k);
      auto c = s.child(actions[k]);
      kids.push_back(visit(*c, id, depth + 1, child_seq));
    }
    t_.child_begin[id] = static_cast<int32_t>(t_.children.size());
    t_.child_count[id] = static_cast<int32_t>(kids.size());
    for (size_t k = 0; k < kids.size(); ++k) {
      t_.children.push_back(kids[k]);
      t_.child_action.push_back(actions[k]);
      t_.chance_prob.push_back(0.0);
    }
    return id;
  }

  GameTree& t_;
  GameTree::BuildOptions opts_;
  std::unordered_map<std::string, int> aug_map_[2];
  std::unordered_map<std::string, int> public_map_;
};

GameTree GameTree::build(const Game& game, BuildOptions opts) {
  GameTree t;
  t.game_name = game.name();
  AEGIS_CHECK_MSG(game.num_players() == 2, "only two-player games are supported");
  auto root = game.initial_state();
  GameTreeBuilder b(t, opts);
  b.run(*root);
  return t;
}

int GameTree::infoset_index(int p, const std::string& key) const {
  AEGIS_CHECK(p == 0 || p == 1);
  auto it = infoset_lookup_[p].find(key);
  return it == infoset_lookup_[p].end() ? -1 : it->second;
}

}  // namespace aegis
