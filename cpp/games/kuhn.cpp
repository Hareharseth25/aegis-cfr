#include "games/kuhn.h"

#include "util/check.h"

namespace aegis {

std::unique_ptr<State> KuhnGame::initial_state() const { return std::make_unique<KuhnState>(); }

int KuhnState::current_player() const {
  if (cards_[0] < 0 || cards_[1] < 0) return kChancePlayer;
  if (is_terminal()) return kTerminalPlayer;
  return static_cast<int>(bets_.size()) % 2;
}

bool KuhnState::is_terminal() const {
  if (cards_[1] < 0) return false;
  const auto n = bets_.size();
  if (n < 2) return false;
  // pp, bp, bb (showdown/fold after a bet) end after two actions; pb needs one more.
  if (n == 2) return !(bets_[0] == 0 && bets_[1] == 1);
  return n == 3;
}

std::vector<Action> KuhnState::legal_actions() const {
  if (is_terminal() || current_player() == kChancePlayer) return {};
  return {0, 1};
}

std::vector<std::pair<Action, double>> KuhnState::chance_outcomes() const {
  AEGIS_CHECK(current_player() == kChancePlayer);
  std::vector<std::pair<Action, double>> out;
  if (cards_[0] < 0) {
    for (int c = 0; c < 3; ++c) out.emplace_back(c, 1.0 / 3.0);
  } else {
    for (int c = 0; c < 3; ++c)
      if (c != cards_[0]) out.emplace_back(c, 0.5);
  }
  return out;
}

std::unique_ptr<State> KuhnState::child(Action a) const {
  auto s = std::make_unique<KuhnState>(*this);
  if (cards_[0] < 0) {
    AEGIS_CHECK(a >= 0 && a < 3);
    s->cards_[0] = a;
  } else if (cards_[1] < 0) {
    AEGIS_CHECK(a >= 0 && a < 3 && a != cards_[0]);
    s->cards_[1] = a;
  } else {
    AEGIS_CHECK(!is_terminal() && (a == 0 || a == 1));
    s->bets_.push_back(a);
  }
  return s;
}

std::string KuhnState::information_state_key(int player) const {
  AEGIS_CHECK(player == 0 || player == 1);
  std::string key;
  key += cards_[player] < 0 ? '?' : static_cast<char>('0' + cards_[player]);
  for (Action b : bets_) key += (b == 0 ? 'p' : 'b');
  return key;
}

std::string KuhnState::public_state_key() const {
  std::string key = "pub:";
  for (Action b : bets_) key += (b == 0 ? 'p' : 'b');
  return key;
}

std::array<double, 2> KuhnState::returns() const {
  if (!is_terminal()) return {0.0, 0.0};
  const int winner_showdown = cards_[0] > cards_[1] ? 0 : 1;
  double u0 = 0.0;
  const std::string h = information_state_key(0).substr(1);
  if (h == "pp") {
    u0 = winner_showdown == 0 ? 1.0 : -1.0;
  } else if (h == "bp") {
    u0 = 1.0;  // player 1 folds
  } else if (h == "pbp") {
    u0 = -1.0;  // player 0 folds
  } else {  // "bb" or "pbb": called bet, showdown for 2
    u0 = winner_showdown == 0 ? 2.0 : -2.0;
  }
  return {u0, -u0};
}

std::string KuhnState::action_to_string(Action a) const {
  if (current_player() == kChancePlayer) {
    static const char* names[] = {"J", "Q", "K"};
    return std::string("Deal:") + names[a];
  }
  return a == 0 ? "Pass" : "Bet";
}

std::string KuhnState::to_string() const {
  std::string s = "cards=";
  s += cards_[0] < 0 ? '?' : static_cast<char>('0' + cards_[0]);
  s += cards_[1] < 0 ? '?' : static_cast<char>('0' + cards_[1]);
  s += " hist=";
  for (Action b : bets_) s += (b == 0 ? 'p' : 'b');
  return s;
}

}  // namespace aegis
