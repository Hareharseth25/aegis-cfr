#include "games/holdem_subgame_efg.h"

#include "poker/hand_evaluator/evaluator.h"
#include "util/check.h"

namespace aegis {

using namespace poker;

HoldemSubgameGame::HoldemSubgameGame(const SubgameContext& ctx) : ctx_(ctx) {
  const auto& H0 = ctx.hands(0);
  const auto& H1 = ctx.hands(1);
  double total = 0.0;
  for (int i = 0; i < H0.size(); ++i)
    for (int j = 0; j < H1.size(); ++j) {
      if (combos().mask[H0.combos[i]] & combos().mask[H1.combos[j]]) continue;
      const double w = H0.root_weight[i] * H1.root_weight[j];
      if (w <= 0.0) continue;
      pairs_.push_back({i, j, w});
      total += w;
    }
  AEGIS_CHECK(total > 0.0);
  for (auto& p : pairs_) p.p /= total;
}

std::unique_ptr<State> HoldemSubgameGame::initial_state() const { return std::make_unique<HoldemSubgameState>(*this); }

bool HoldemSubgameState::is_terminal() const {
  if (node_ < 0) return false;
  const auto t = g_->context().tree().nodes[node_].type;
  return t == PNodeType::kFold || t == PNodeType::kShowdown;
}

int HoldemSubgameState::current_player() const {
  if (node_ < 0) return kChancePlayer;
  const auto& nd = g_->context().tree().nodes[node_];
  switch (nd.type) {
    case PNodeType::kDecision:
      return nd.player;
    case PNodeType::kChance:
      return kChancePlayer;
    case PNodeType::kLeaf:
      AEGIS_CHECK_MSG(false, "depth-limited trees cannot be expressed as a finite EFG");
      return kTerminalPlayer;
    default:
      return kTerminalPlayer;
  }
}

std::vector<Action> HoldemSubgameState::legal_actions() const {
  if (node_ < 0 || is_terminal()) return {};
  const auto& nd = g_->context().tree().nodes[node_];
  if (nd.type != PNodeType::kDecision) return {};
  std::vector<Action> a(nd.num_children);
  for (int k = 0; k < nd.num_children; ++k) a[k] = k;
  return a;
}

std::vector<std::pair<Action, double>> HoldemSubgameState::chance_outcomes() const {
  std::vector<std::pair<Action, double>> out;
  if (node_ < 0) {
    const auto& pairs = g_->pairs();
    for (size_t k = 0; k < pairs.size(); ++k) out.emplace_back(static_cast<Action>(k), pairs[k].p);
    return out;
  }
  const auto& ctx = g_->context();
  const auto& t = ctx.tree();
  const auto& nd = t.nodes[node_];
  AEGIS_CHECK(nd.type == PNodeType::kChance);
  const CardMask hands = combos().mask[ctx.hands(0).combos[hand_[0]]] | combos().mask[ctx.hands(1).combos[hand_[1]]];
  int count = 0;
  for (int k = 0; k < nd.num_children; ++k)
    if (!(card_bit(t.nodes[nd.first_child + k].action.card) & hands)) ++count;
  for (int k = 0; k < nd.num_children; ++k)
    if (!(card_bit(t.nodes[nd.first_child + k].action.card) & hands)) out.emplace_back(k, 1.0 / count);
  return out;
}

std::unique_ptr<State> HoldemSubgameState::child(Action a) const {
  auto s = std::make_unique<HoldemSubgameState>(*this);
  if (node_ < 0) {
    const auto& pr = g_->pairs()[a];
    s->hand_[0] = pr.h0;
    s->hand_[1] = pr.h1;
    s->node_ = 0;
    return s;
  }
  const auto& nd = g_->context().tree().nodes[node_];
  AEGIS_CHECK(a >= 0 && a < nd.num_children);
  s->node_ = nd.first_child + a;
  return s;
}

std::string HoldemSubgameState::information_state_key(int player) const {
  return HoldemSubgameGame::key(node_, hand_[player]);
}

std::array<double, 2> HoldemSubgameState::returns() const {
  if (!is_terminal()) return {0.0, 0.0};
  const auto& ctx = g_->context();
  const auto& t = ctx.tree();
  const auto& nd = t.nodes[node_];
  if (nd.type == PNodeType::kFold) {
    const int f = nd.player;
    std::array<double, 2> u{};
    u[f] = -nd.commit[f];
    u[1 - f] = nd.commit[f];
    return u;
  }
  const CardMask bm = t.board_masks[nd.board_id];
  const HandValue v0 = evaluate(bm | combos().mask[ctx.hands(0).combos[hand_[0]]]);
  const HandValue v1 = evaluate(bm | combos().mask[ctx.hands(1).combos[hand_[1]]]);
  if (v0 == v1) return {0.0, 0.0};
  const double c = nd.commit[0];
  return v0 > v1 ? std::array<double, 2>{c, -c} : std::array<double, 2>{-c, c};
}

std::string HoldemSubgameState::action_to_string(Action a) const {
  if (node_ < 0) return "deal" + std::to_string(a);
  const auto& t = g_->context().tree();
  return t.nodes[t.nodes[node_].first_child + a].action.to_string();
}

std::string HoldemSubgameState::to_string() const {
  return "hands=" + std::to_string(hand_[0]) + "," + std::to_string(hand_[1]) + " node=" + std::to_string(node_);
}

}  // namespace aegis
