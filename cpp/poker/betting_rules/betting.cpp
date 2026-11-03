#include "poker/betting_rules/betting.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "util/check.h"

namespace aegis::poker {

namespace {
constexpr double kEps = 1e-9;
std::string fmt(double x) {
  std::ostringstream os;
  os.precision(4);
  os << x;
  return os.str();
}
}  // namespace

std::string ActionLabel::to_string() const {
  switch (kind) {
    case ActionKind::kFold:
      return "Fold";
    case ActionKind::kCheck:
      return "Check";
    case ActionKind::kCall:
      return "Call";
    case ActionKind::kBet:
      return "Bet " + fmt(amount);
    case ActionKind::kRaise:
      return "Raise " + fmt(amount);
    case ActionKind::kAllIn:
      return "AllIn " + fmt(amount);
    case ActionKind::kDeal:
      return "Deal " + card_to_string(card);
  }
  return "?";
}

PublicTree PublicTree::build(const TreeConfig& cfg) {
  AEGIS_CHECK_MSG(cfg.board.size() == 4 || cfg.board.size() == 5, "board must have 4 (turn) or 5 (river) cards");
  AEGIS_CHECK_MSG(popcount(mask_of(cfg.board)) == static_cast<int>(cfg.board.size()), "duplicate board cards");
  AEGIS_CHECK_MSG(cfg.pot > 0.0, "pot must be positive");
  AEGIS_CHECK_MSG(cfg.effective_stack >= 0.0, "negative stack");
  AEGIS_CHECK(cfg.first_player == 0 || cfg.first_player == 1);
  for (int p = 0; p < 2; ++p)
    AEGIS_CHECK_MSG(cfg.initial_street_bet[p] >= 0.0 && cfg.initial_street_bet[p] <= cfg.effective_stack + kEps,
                    "invalid initial street bet");
  PublicTree t;
  t.cfg_ = cfg;
  t.boards.push_back(cfg.board);
  t.board_masks.push_back(mask_of(cfg.board));
  PublicNode root;
  root.type = PNodeType::kDecision;
  root.street = cfg.board.size() == 4 ? 0 : 1;
  root.player = cfg.initial_player >= 0 ? cfg.initial_player : cfg.first_player;
  for (int p = 0; p < 2; ++p) {
    root.street_bet[p] = cfg.initial_street_bet[p];
    root.commit[p] = cfg.pot / 2.0 + cfg.initial_street_bet[p];
  }
  root.raises = cfg.initial_raises;
  root.checks = cfg.initial_checks;
  root.last_raise = cfg.initial_last_raise;
  root.action.kind = ActionKind::kCheck;
  t.nodes.push_back(root);
  t.expand(0);
  for (auto& n : t.nodes)
    if (n.type == PNodeType::kDecision) n.decision_index = t.num_decisions_++;
  return t;
}

int PublicTree::board_for(int parent_board, Card c) {
  std::vector<Card> b = boards[parent_board];
  b.push_back(c);
  boards.push_back(b);
  board_masks.push_back(mask_of(b));
  return static_cast<int>(boards.size()) - 1;
}

std::vector<std::pair<ActionLabel, PublicNode>> PublicTree::decision_children(const PublicNode& nd) const {
  std::vector<std::pair<ActionLabel, PublicNode>> out;
  const int p = nd.player, o = 1 - p;
  const StreetSizes& sz = cfg_.sizes[nd.street][p];
  const double stack_p = cfg_.effective_stack - (nd.commit[p] - cfg_.pot / 2.0);
  const double stack_o = cfg_.effective_stack - (nd.commit[o] - cfg_.pot / 2.0);
  const double to_call = nd.street_bet[o] - nd.street_bet[p];
  const double pot_now = nd.commit[0] + nd.commit[1];
  AEGIS_CHECK(to_call >= -kEps);

  auto next_player = [&](PublicNode c) {
    c.type = PNodeType::kDecision;
    c.player = o;
    return c;
  };
  // Marks a street as complete; the caller's expand() resolves what follows.
  auto street_end = [&](PublicNode c) {
    c.type = PNodeType::kChance;  // placeholder, resolved in expand()
    c.player = -1;
    return c;
  };

  if (to_call <= kEps) {
    // Check.
    {
      PublicNode c = nd;
      c.checks = nd.checks + 1;
      ActionLabel a;
      a.kind = ActionKind::kCheck;
      out.emplace_back(a, c.checks >= 2 ? street_end(c) : next_player(c));
    }
    if (stack_p > kEps && stack_o > kEps) {
      std::vector<std::pair<double, ActionLabel>> bets;
      auto add_bet = [&](double amt, double nominal) {
        amt = std::max(amt, std::min(cfg_.min_bet, stack_p));
        ActionLabel a;
        if (amt >= stack_p - kEps) {
          amt = stack_p;
          a.kind = ActionKind::kAllIn;
        } else {
          a.kind = ActionKind::kBet;
        }
        a.amount = amt;
        a.fraction = amt / pot_now;
        a.nominal = a.kind == ActionKind::kAllIn ? -1.0 : nominal;
        for (auto& [x, lab] : bets)
          if (std::abs(x - amt) < 1e-7) return;
        bets.emplace_back(amt, a);
      };
      for (double f : sz.bet_fractions) {
        AEGIS_CHECK_MSG(f > 0.0, "bet fractions must be positive");
        add_bet(f * pot_now, f);
      }
      if (sz.allow_all_in) add_bet(stack_p, -1.0);
      std::sort(bets.begin(), bets.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
      for (auto& [amt, a] : bets) {
        PublicNode c = nd;
        c.street_bet[p] = nd.street_bet[p] + amt;
        c.commit[p] = nd.commit[p] + amt;
        c.last_raise = amt;
        out.emplace_back(a, next_player(c));
      }
    }
    return out;
  }

  // Facing a bet: fold, call, raises.
  {
    PublicNode c = nd;
    c.type = PNodeType::kFold;
    c.player = p;
    ActionLabel a;
    a.kind = ActionKind::kFold;
    out.emplace_back(a, c);
  }
  {
    PublicNode c = nd;
    const double add = std::min(to_call, stack_p);
    c.street_bet[p] = nd.street_bet[p] + add;
    c.commit[p] = nd.commit[p] + add;
    ActionLabel a;
    a.kind = ActionKind::kCall;
    a.amount = add;
    out.emplace_back(a, street_end(c));
  }
  if (nd.raises < sz.max_raises && stack_p > to_call + kEps && stack_o > kEps) {
    std::vector<std::pair<double, ActionLabel>> raises;
    const double min_inc = std::max(nd.last_raise, cfg_.min_bet);
    auto add_raise = [&](double raise_to, double nominal) {
      raise_to = std::max(raise_to, nd.street_bet[o] + min_inc);
      double add = raise_to - nd.street_bet[p];
      ActionLabel a;
      if (add >= stack_p - kEps) {
        add = stack_p;
        raise_to = nd.street_bet[p] + stack_p;
        a.kind = ActionKind::kAllIn;
      } else {
        a.kind = ActionKind::kRaise;
      }
      a.amount = raise_to;
      a.fraction = (raise_to - nd.street_bet[o]) / (pot_now + to_call);
      a.nominal = a.kind == ActionKind::kAllIn ? -1.0 : nominal;
      for (auto& [x, lab] : raises)
        if (std::abs(x - raise_to) < 1e-7) return;
      raises.emplace_back(raise_to, a);
    };
    for (double f : sz.raise_fractions) {
      AEGIS_CHECK_MSG(f > 0.0, "raise fractions must be positive");
      add_raise(nd.street_bet[o] + f * (pot_now + to_call), f);
    }
    if (sz.allow_all_in) add_raise(nd.street_bet[p] + stack_p, -1.0);
    std::sort(raises.begin(), raises.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    for (auto& [raise_to, a] : raises) {
      PublicNode c = nd;
      const double add = raise_to - nd.street_bet[p];
      c.street_bet[p] = raise_to;
      c.commit[p] = nd.commit[p] + add;
      c.last_raise = raise_to - nd.street_bet[o];
      c.raises = nd.raises + 1;
      out.emplace_back(a, next_player(c));
    }
  }
  return out;
}

int PublicTree::add_child(int parent, const PublicNode& proto) {
  PublicNode c = proto;
  c.parent = parent;
  c.depth = nodes[parent].depth + 1;
  c.first_child = -1;
  c.num_children = 0;
  nodes.push_back(c);
  return static_cast<int>(nodes.size()) - 1;
}

void PublicTree::expand(int n) {
  PublicNode nd = nodes[n];
  if (nd.type == PNodeType::kFold || nd.type == PNodeType::kShowdown || nd.type == PNodeType::kLeaf) return;

  if (nd.type == PNodeType::kChance) {
    // A street just ended (placeholder from decision_children) or a real
    // chance node. Resolve placeholders first.
    if (nd.num_children == 0 && nd.action.kind != ActionKind::kDeal && nd.first_child == -1 && nd.player == -1) {
      const double s0 = stack_remaining(n, 0), s1 = stack_remaining(n, 1);
      const bool all_in = s0 <= kEps || s1 <= kEps;
      if (nd.street == 1) {
        nodes[n].type = PNodeType::kShowdown;
        return;
      }
      if (!all_in && cfg_.depth_limited) {
        nodes[n].type = PNodeType::kLeaf;
        return;
      }
      // Deal the river.
      const int first = static_cast<int>(nodes.size());
      const CardMask bm = board_masks[nd.board_id];
      std::vector<int> kids;
      for (Card c = 0; c < kNumCards; ++c) {
        if (bm & card_bit(c)) continue;
        PublicNode k;
        k.street = 1;
        k.board_id = board_for(nd.board_id, c);
        k.commit[0] = nd.commit[0];
        k.commit[1] = nd.commit[1];
        k.street_bet[0] = k.street_bet[1] = 0.0;
        k.raises = 0;
        k.checks = 0;
        k.last_raise = 0.0;
        k.action.kind = ActionKind::kDeal;
        k.action.card = c;
        if (all_in) {
          k.type = PNodeType::kShowdown;
          k.player = -1;
        } else {
          k.type = PNodeType::kDecision;
          k.player = cfg_.first_player;
        }
        kids.push_back(add_child(n, k));
      }
      nodes[n].first_child = first;
      nodes[n].num_children = static_cast<int>(kids.size());
      for (int k : kids) expand(k);
      return;
    }
    return;
  }

  // Decision node.
  auto kids = decision_children(nd);
  AEGIS_CHECK(!kids.empty());
  const int first = static_cast<int>(nodes.size());
  std::vector<int> ids;
  for (auto& [label, proto] : kids) {
    PublicNode c = proto;
    c.action = label;
    if (c.type == PNodeType::kChance) {
      c.first_child = -1;
      c.num_children = 0;
    }
    ids.push_back(add_child(n, c));
  }
  nodes[n].first_child = first;
  nodes[n].num_children = static_cast<int>(ids.size());
  for (int id : ids) expand(id);
}

std::string PublicTree::path_string(int n) const {
  node(n);  // bounds check
  std::vector<std::string> parts;
  while (n > 0) {
    parts.push_back(nodes[n].action.to_string());
    n = nodes[n].parent;
  }
  std::string s;
  for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
    if (!s.empty()) s += " / ";
    s += *it;
  }
  return s.empty() ? "root" : s;
}

int PublicTree::child_by_label(int n, const std::string& label) const {
  node(n);  // bounds check
  const auto& nd = nodes[n];
  for (int k = 0; k < nd.num_children; ++k)
    if (nodes[nd.first_child + k].action.to_string() == label) return nd.first_child + k;
  return -1;
}

int PublicTree::count_type(PNodeType t) const {
  int c = 0;
  for (const auto& n : nodes) c += n.type == t;
  return c;
}

}  // namespace aegis::poker
