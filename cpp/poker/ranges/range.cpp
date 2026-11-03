#include "poker/ranges/range.h"

#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "util/check.h"

namespace aegis::poker {

namespace {

std::string trim(std::string_view s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return std::string(s.substr(a, b - a));
}

// Adds weight to all combos of a hand class given ranks (r1 >= r2) and a
// suitedness flag: 's', 'o' or 0 (both).
void add_class(Range& r, int r1, int r2, char kind, double weight) {
  if (r1 < r2) std::swap(r1, r2);
  for (int s1 = 0; s1 < 4; ++s1)
    for (int s2 = 0; s2 < 4; ++s2) {
      const Card a = make_card(r1, s1), b = make_card(r2, s2);
      if (a == b) continue;
      if (r1 == r2 && s1 >= s2) continue;  // each pair once
      const bool suited = s1 == s2;
      if (r1 != r2) {
        if (kind == 's' && !suited) continue;
        if (kind == 'o' && suited) continue;
      }
      r[combo_index(a, b)] = weight;
    }
}

struct ClassSpec {
  int r1, r2;  // r1 >= r2
  char kind;   // 's', 'o', 0
};

bool parse_class(std::string_view s, ClassSpec& out) {
  if (s.size() < 2 || s.size() > 3) return false;
  int r1 = rank_from_char(s[0]), r2 = rank_from_char(s[1]);
  if (r1 < 0 || r2 < 0) return false;
  if (r1 < r2) std::swap(r1, r2);
  char kind = 0;
  if (s.size() == 3) {
    kind = static_cast<char>(std::tolower(static_cast<unsigned char>(s[2])));
    if (kind != 's' && kind != 'o') return false;
    if (r1 == r2) return false;
  }
  out = {r1, r2, kind};
  return true;
}

void apply_token(Range& r, const std::string& tok, double weight) {
  std::string t;
  for (char c : tok)
    if (!std::isspace(static_cast<unsigned char>(c))) t += c;
  std::string lower;
  for (char c : t) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (lower == "random" || lower == "all" || lower == "any" || lower == "100%") {
    for (auto& x : r.w) x = weight;
    return;
  }
  // Specific combination such as "AsKh".
  if (t.size() == 4 && std::isalpha(static_cast<unsigned char>(t[1])) && std::islower(static_cast<unsigned char>(t[1])) &&
      std::islower(static_cast<unsigned char>(t[3]))) {
    r[parse_combo(t)] = weight;
    return;
  }
  // Dash range: "22-55", "A2s-A5s".
  const auto dash = t.find('-');
  if (dash != std::string::npos) {
    ClassSpec a, b;
    AEGIS_CHECK_MSG(parse_class(t.substr(0, dash), a) && parse_class(t.substr(dash + 1), b),
                    "bad range token '" << tok << "'");
    if (a.r1 == a.r2) {
      AEGIS_CHECK_MSG(b.r1 == b.r2, "pair range must be pairs: '" << tok << "'");
      for (int x = std::min(a.r1, b.r1); x <= std::max(a.r1, b.r1); ++x) add_class(r, x, x, 0, weight);
      return;
    }
    AEGIS_CHECK_MSG(a.r1 == b.r1 && a.kind == b.kind, "dash range needs same first rank: '" << tok << "'");
    for (int x = std::min(a.r2, b.r2); x <= std::max(a.r2, b.r2); ++x) add_class(r, a.r1, x, a.kind, weight);
    return;
  }
  // Plus range: "TT+", "ATs+".
  if (!t.empty() && t.back() == '+') {
    ClassSpec c;
    AEGIS_CHECK_MSG(parse_class(t.substr(0, t.size() - 1), c), "bad range token '" << tok << "'");
    if (c.r1 == c.r2) {
      for (int x = c.r1; x <= 12; ++x) add_class(r, x, x, 0, weight);
    } else {
      for (int x = c.r2; x < c.r1; ++x) add_class(r, c.r1, x, c.kind, weight);
    }
    return;
  }
  ClassSpec c;
  AEGIS_CHECK_MSG(parse_class(t, c), "bad range token '" << tok << "'");
  add_class(r, c.r1, c.r2, c.kind, weight);
}

}  // namespace

Range Range::full() {
  Range r;
  r.w.fill(1.0);
  return r;
}

Range Range::from_vector(std::span<const double> weights) {
  AEGIS_CHECK(static_cast<int>(weights.size()) == kNumCombos);
  Range r;
  for (int i = 0; i < kNumCombos; ++i) {
    AEGIS_CHECK_MSG(weights[i] >= 0.0 && std::isfinite(weights[i]), "invalid range weight");
    r.w[i] = weights[i];
  }
  return r;
}

Range Range::parse(std::string_view text) {
  Range r;
  std::string s(text);
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (item.empty()) continue;
    double weight = 1.0;
    const auto colon = item.find(':');
    std::string hand = item;
    if (colon != std::string::npos) {
      hand = trim(item.substr(0, colon));
      const std::string ws = trim(item.substr(colon + 1));
      try {
        weight = std::stod(ws);
      } catch (...) {
        AEGIS_CHECK_MSG(false, "bad weight in '" << item << "'");
      }
      AEGIS_CHECK_MSG(weight >= 0.0, "negative weight in '" << item << "'");
    }
    apply_token(r, hand, weight);
  }
  return r;
}

Range& Range::remove(CardMask dead) {
  const auto& C = combos();
  for (int i = 0; i < kNumCombos; ++i)
    if (C.mask[i] & dead) w[i] = 0.0;
  return *this;
}

double Range::total() const {
  double s = 0.0;
  for (double x : w) s += x;
  return s;
}

int Range::num_nonzero() const {
  int n = 0;
  for (double x : w) n += x > 0.0;
  return n;
}

Range& Range::normalize() {
  const double t = total();
  if (t > 0.0)
    for (auto& x : w) x /= t;
  return *this;
}

Range Range::normalized() const {
  Range r = *this;
  r.normalize();
  return r;
}

Range Range::bayes_update(std::span<const double> likelihood) const {
  AEGIS_CHECK(static_cast<int>(likelihood.size()) == kNumCombos);
  Range post;
  double z = 0.0;
  for (int i = 0; i < kNumCombos; ++i) {
    AEGIS_CHECK_MSG(likelihood[i] >= 0.0 && likelihood[i] <= 1.0 + 1e-12, "likelihood must be a probability");
    post.w[i] = w[i] * likelihood[i];
    z += post.w[i];
  }
  AEGIS_CHECK_MSG(z > 0.0, "observed action has zero probability under the assumed policy");
  for (auto& x : post.w) x /= z;
  return post;
}

Range Range::conditioned_on(CardMask known) const {
  Range r = *this;
  r.remove(known);
  r.normalize();
  return r;
}

double Range::blocked_fraction(CardMask known) const {
  const double t = total();
  if (t <= 0.0) return 0.0;
  const auto& C = combos();
  double blocked = 0.0;
  for (int i = 0; i < kNumCombos; ++i)
    if (C.mask[i] & known) blocked += w[i];
  return blocked / t;
}

std::array<double, kNumHandClasses> Range::class_mass() const {
  std::array<double, kNumHandClasses> m{};
  const auto& C = combos();
  for (int i = 0; i < kNumCombos; ++i) m[C.hand_class[i]] += w[i];
  return m;
}

std::array<double, kNumHandClasses> Range::class_frequency(CardMask dead) const {
  std::array<double, kNumHandClasses> m{}, n{};
  const auto& C = combos();
  for (int i = 0; i < kNumCombos; ++i) {
    if (C.mask[i] & dead) continue;
    m[C.hand_class[i]] += w[i];
    n[C.hand_class[i]] += 1.0;
  }
  for (int k = 0; k < kNumHandClasses; ++k) m[k] = n[k] > 0 ? m[k] / n[k] : 0.0;
  return m;
}

std::string Range::to_string() const {
  // Classes whose live combos all share one weight are written compactly;
  // otherwise individual combos are listed.
  const auto& C = combos();
  std::ostringstream os;
  os << std::setprecision(4);
  bool first = true;
  auto emit = [&](const std::string& name, double weight) {
    if (!first) os << ",";
    first = false;
    os << name;
    if (std::abs(weight - 1.0) > 1e-9) os << ":" << weight;
  };
  for (int cls = 0; cls < kNumHandClasses; ++cls) {
    std::vector<int> members;
    for (int i = 0; i < kNumCombos; ++i)
      if (C.hand_class[i] == cls) members.push_back(i);
    bool uniform = true;
    for (int i : members)
      if (std::abs(w[i] - w[members[0]]) > 1e-12) uniform = false;
    if (uniform) {
      if (w[members[0]] > 0.0) emit(hand_class_name(cls), w[members[0]]);
    } else {
      for (int i : members)
        if (w[i] > 0.0) emit(combo_to_string(i), w[i]);
    }
  }
  return os.str();
}

Range random_range_by_strength(std::span<const int> order, Rng& rng) {
  // Recursive splitting: the mass of a contiguous block of strength-sorted
  // hands is divided between its two halves by a uniform random fraction.
  // This produces ranges ranging from polarised to condensed.
  Range r;
  const int n = static_cast<int>(order.size());
  if (n == 0) return r;
  std::vector<double> mass(n, 0.0);
  struct Item {
    int lo, hi;
    double m;
  };
  std::vector<Item> stack{{0, n, 1.0}};
  while (!stack.empty()) {
    Item it = stack.back();
    stack.pop_back();
    if (it.hi - it.lo == 1) {
      mass[it.lo] = it.m;
      continue;
    }
    const int mid = (it.lo + it.hi) / 2;
    const double f = rng.uniform();
    stack.push_back({it.lo, mid, it.m * f});
    stack.push_back({mid, it.hi, it.m * (1.0 - f)});
  }
  for (int k = 0; k < n; ++k) r.w[order[k]] = mass[k];
  // Convert to "probability of holding" weights in [0, 1].
  double mx = 0.0;
  for (double x : r.w) mx = std::max(mx, x);
  if (mx > 0.0)
    for (auto& x : r.w) x /= mx;
  return r;
}

}  // namespace aegis::poker
