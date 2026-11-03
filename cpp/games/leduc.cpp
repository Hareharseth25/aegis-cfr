#include "games/leduc.h"

#include <sstream>

#include "util/check.h"

namespace aegis {

LeducGame::LeducGame(LeducConfig cfg) : cfg_(cfg) {
  AEGIS_CHECK(cfg_.num_ranks >= 2 && cfg_.num_suits >= 1);
  AEGIS_CHECK(cfg_.num_ranks * cfg_.num_suits >= 3);
  AEGIS_CHECK(cfg_.max_raises >= 0);
}

std::unique_ptr<State> LeducGame::initial_state() const { return std::make_unique<LeducState>(cfg_); }

std::string LeducGame::name() const {
  std::ostringstream os;
  os << "leduc(ranks=" << cfg_.num_ranks << ",suits=" << cfg_.num_suits << ",raises=" << cfg_.max_raises << ")";
  return os.str();
}

LeducState::LeducState(const LeducConfig& cfg) : cfg_(&cfg) {
  contrib_[0] = contrib_[1] = cfg.ante;
  stakes_ = cfg.ante;
}

bool LeducState::is_terminal() const { return finished_; }

int LeducState::current_player() const { return finished_ ? kTerminalPlayer : player_; }

bool LeducState::card_used(int c) const { return private_[0] == c || private_[1] == c || public_ == c; }

std::vector<Action> LeducState::legal_actions() const {
  if (finished_ || player_ == kChancePlayer) return {};
  std::vector<Action> a;
  if (stakes_ > contrib_[player_]) a.push_back(kFold);
  a.push_back(kCall);
  if (raises_ < cfg_->max_raises) a.push_back(kRaise);
  return a;
}

std::vector<std::pair<Action, double>> LeducState::chance_outcomes() const {
  AEGIS_CHECK(!finished_ && player_ == kChancePlayer);
  const int deck = cfg_->num_ranks * cfg_->num_suits;
  std::vector<std::pair<Action, double>> out;
  int remaining = 0;
  for (int c = 0; c < deck; ++c)
    if (!card_used(c)) ++remaining;
  for (int c = 0; c < deck; ++c)
    if (!card_used(c)) out.emplace_back(c, 1.0 / remaining);
  return out;
}

int LeducState::hand_rank(int player) const {
  const int r = private_[player] / cfg_->num_suits;
  const int pr = public_ / cfg_->num_suits;
  if (r == pr) return 1000 + r;  // pair
  return r;
}

std::unique_ptr<State> LeducState::child(Action a) const {
  auto s = std::make_unique<LeducState>(*this);
  if (player_ == kChancePlayer) {
    AEGIS_CHECK(!finished_);
    AEGIS_CHECK(a >= 0 && a < cfg_->num_ranks * cfg_->num_suits && !card_used(a));
    if (private_[0] < 0) {
      s->private_[0] = a;
    } else if (private_[1] < 0) {
      s->private_[1] = a;
      s->player_ = 0;
    } else {
      s->public_ = a;
      s->player_ = 0;
    }
    return s;
  }
  AEGIS_CHECK(!finished_);
  const int p = player_;
  const int o = 1 - p;
  const double raise_amount = round_ == 1 ? cfg_->raise_round1 : cfg_->raise_round2;
  auto& h = s->hist_[round_ - 1];
  if (a == kFold) {
    AEGIS_CHECK(stakes_ > contrib_[p]);
    h.push_back(kFold);
    s->folded_[p] = true;
    s->finished_ = true;
    return s;
  }
  if (a == kCall) {
    const bool closes_round = !hist_[round_ - 1].empty();
    s->contrib_[p] = stakes_;
    h.push_back(kCall);
    if (closes_round) {
      if (round_ == 2) {
        s->finished_ = true;
      } else {
        s->round_ = 2;
        s->raises_ = 0;
        s->player_ = kChancePlayer;
      }
    } else {
      s->player_ = o;
    }
    return s;
  }
  AEGIS_CHECK_MSG(a == kRaise && raises_ < cfg_->max_raises, "illegal Leduc action " << a);
  s->stakes_ = stakes_ + raise_amount;
  s->contrib_[p] = s->stakes_;
  s->raises_ = raises_ + 1;
  h.push_back(kRaise);
  s->player_ = o;
  return s;
}

std::string LeducState::history_string() const {
  static const char kChar[] = {'f', 'c', 'r'};
  std::string s;
  for (Action a : hist_[0]) s += kChar[a];
  s += '/';
  for (Action a : hist_[1]) s += kChar[a];
  return s;
}

std::string LeducState::information_state_key(int player) const {
  AEGIS_CHECK(player == 0 || player == 1);
  std::string key = "P";
  key += private_[player] < 0 ? std::string("?") : std::to_string(private_[player]);
  key += ":B";
  key += public_ < 0 ? std::string("-") : std::to_string(public_);
  key += ':';
  key += history_string();
  return key;
}

std::string LeducState::public_state_key() const {
  std::string key = "B";
  key += public_ < 0 ? std::string("-") : std::to_string(public_);
  key += ':';
  key += history_string();
  return key;
}

std::array<double, 2> LeducState::returns() const {
  if (!finished_) return {0.0, 0.0};
  if (folded_[0]) return {-contrib_[0], contrib_[0]};
  if (folded_[1]) return {contrib_[1], -contrib_[1]};
  // Showdown: contributions are equal after a call.
  const int r0 = hand_rank(0), r1 = hand_rank(1);
  if (r0 == r1) return {0.0, 0.0};
  return r0 > r1 ? std::array<double, 2>{contrib_[1], -contrib_[1]}
                 : std::array<double, 2>{-contrib_[0], contrib_[0]};
}

std::string LeducState::action_to_string(Action a) const {
  if (player_ == kChancePlayer) return "Deal:" + std::to_string(a);
  static const char* names[] = {"Fold", "Call", "Raise"};
  return names[a];
}

std::string LeducState::to_string() const {
  std::ostringstream os;
  os << "priv=" << private_[0] << "," << private_[1] << " pub=" << public_ << " hist=" << history_string()
     << " contrib=" << contrib_[0] << "," << contrib_[1];
  return os.str();
}

}  // namespace aegis
