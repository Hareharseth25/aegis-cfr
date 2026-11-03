#include "games/coin_toss.h"

#include "util/check.h"

namespace aegis {

std::unique_ptr<State> CoinTossGame::initial_state() const {
  return std::make_unique<CoinTossState>(sell_value_);
}

bool CoinTossState::is_terminal() const {
  return p0_action_ == kSell || (p0_action_ == kPlay && guess_ >= 0);
}

int CoinTossState::current_player() const {
  if (is_terminal()) return kTerminalPlayer;
  if (coin_ < 0) return kChancePlayer;
  if (p0_action_ < 0) return 0;
  return 1;
}

std::vector<Action> CoinTossState::legal_actions() const {
  const int p = current_player();
  if (p == 0) return {kSell, kPlay};
  if (p == 1) return {kHeads, kTails};
  return {};
}

std::vector<std::pair<Action, double>> CoinTossState::chance_outcomes() const {
  AEGIS_CHECK(current_player() == kChancePlayer);
  return {{kHeads, 0.5}, {kTails, 0.5}};
}

std::unique_ptr<State> CoinTossState::child(Action a) const {
  auto s = std::make_unique<CoinTossState>(*this);
  const int p = current_player();
  AEGIS_CHECK(a == 0 || a == 1);
  if (p == kChancePlayer) s->coin_ = a;
  else if (p == 0) s->p0_action_ = a;
  else if (p == 1) s->guess_ = a;
  else AEGIS_CHECK_MSG(false, "child() on terminal coin toss state");
  return s;
}

std::string CoinTossState::information_state_key(int player) const {
  std::string k;
  if (player == 0) k += coin_ < 0 ? '?' : (coin_ == kHeads ? 'H' : 'T');
  else k += '*';
  k += ':';
  if (p0_action_ >= 0) k += p0_action_ == kSell ? 'S' : 'P';
  return k;
}

std::string CoinTossState::public_state_key() const {
  std::string k = "pub:";
  if (p0_action_ >= 0) k += p0_action_ == kSell ? 'S' : 'P';
  return k;
}

std::array<double, 2> CoinTossState::returns() const {
  if (!is_terminal()) return {0.0, 0.0};
  double u0;
  if (p0_action_ == kSell) u0 = coin_ == kHeads ? sell_value_ : -sell_value_;
  else u0 = guess_ == coin_ ? -1.0 : 1.0;
  return {u0, -u0};
}

std::string CoinTossState::action_to_string(Action a) const {
  const int p = current_player();
  if (p == kChancePlayer) return a == kHeads ? "Heads" : "Tails";
  if (p == 0) return a == kSell ? "Sell" : "Play";
  return a == kHeads ? "GuessHeads" : "GuessTails";
}

std::string CoinTossState::to_string() const {
  return information_state_key(0) + (guess_ >= 0 ? (guess_ == kHeads ? "gH" : "gT") : "");
}

}  // namespace aegis
