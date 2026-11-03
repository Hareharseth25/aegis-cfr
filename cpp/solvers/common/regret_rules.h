// Regret-update and averaging rules shared by every CFR-family solver in AEGIS
// (tabular trees, vectorised poker trees, sampled variants). Keeping one
// definition guarantees that "CFR+" means exactly the same thing on Kuhn,
// Leduc and a Hold'em river subgame.
//
// Iteration t = 1, 2, ... For each updated information set I and action a:
//   r^t(I,a) = v^{σ^t}(I,a) - v^{σ^t}(I)           (counterfactual regret)
//   vanilla : R^t = R^{t-1} + r^t
//   CFR+    : R^t = max(0, R^{t-1} + r^t)             (Tammelin 2014)
//   DCFR    : R^t = (R^{t-1} + r^t) * d_t,  d_t = t^α/(t^α+1) if R >= 0,
//                                               t^β/(t^β+1) otherwise
//             (Brown & Sandholm 2019; β = -∞ zeroes negative regrets)
//   linear  : DCFR(α=1, β=1, γ=1) - iteration t weighted by t (LCFR)
// Average strategy: S^t(I,a) = S^{t-1}(I,a) + w_t * π_i^{σ^t}(I) * σ^t(I,a) with
//   vanilla w_t = 1; CFR+ w_t = max(t - d, 0) (linear averaging, delay d);
//   DCFR / linear w_t = t^γ (equivalent, after normalisation, to discounting
//   the accumulated average by (t/(t+1))^γ each iteration).
// The next strategy is regret matching on R^t (uniform if no positive regret).
#pragma once

#include <cmath>
#include <limits>
#include <string>

namespace aegis {

enum class CFRVariant { kVanilla, kCFRPlus, kDCFR, kLinear };

struct CFRConfig {
  CFRVariant variant = CFRVariant::kVanilla;
  bool alternating = true;  // alternating (player 0 then 1) vs simultaneous updates
  double alpha = 1.5;
  double beta = 0.0;
  double gamma = 2.0;
  int averaging_delay = 0;  // CFR+ only
  // Instantaneous regrets |v(a) - v| <= tie_tolerance * (|v(a)| + |v|) are
  // treated as exact ties (set to 0). Mathematically equal action values that
  // are computed along different arithmetic paths otherwise differ by ~1e-16
  // relative, and regret matching would turn that rounding noise into a pure
  // strategy. 0 disables the snap.
  double tie_tolerance = 1e-12;

  static CFRConfig vanilla(bool alternating = true) {
    CFRConfig c;
    c.variant = CFRVariant::kVanilla;
    c.alternating = alternating;
    return c;
  }
  static CFRConfig cfr_plus(int delay = 0) {
    CFRConfig c;
    c.variant = CFRVariant::kCFRPlus;
    c.alternating = true;
    c.averaging_delay = delay;
    return c;
  }
  static CFRConfig dcfr(double a = 1.5, double b = 0.0, double g = 2.0) {
    CFRConfig c;
    c.variant = CFRVariant::kDCFR;
    c.alternating = true;
    c.alpha = a;
    c.beta = b;
    c.gamma = g;
    return c;
  }
  static CFRConfig linear() {
    CFRConfig c = dcfr(1.0, 1.0, 1.0);
    c.variant = CFRVariant::kLinear;
    return c;
  }

  bool clips_regrets() const { return variant == CFRVariant::kCFRPlus; }
  bool discounts() const { return variant == CFRVariant::kDCFR || variant == CFRVariant::kLinear; }

  // Multiplier applied to accumulated regret R (after adding r^t) at iteration t.
  double regret_discount(int t, bool positive) const {
    if (!discounts()) return 1.0;
    const double e = positive ? alpha : beta;
    if (std::isinf(e)) return e > 0 ? 1.0 : 0.0;
    const double x = std::pow(static_cast<double>(t), e);
    return x / (x + 1.0);
  }

  double average_weight(int t) const {
    switch (variant) {
      case CFRVariant::kVanilla:
        return 1.0;
      case CFRVariant::kCFRPlus:
        return std::max(0, t - averaging_delay);
      case CFRVariant::kDCFR:
      case CFRVariant::kLinear:
        return std::pow(static_cast<double>(t), gamma);
    }
    return 1.0;
  }

  // Applies one regret update in place.
  void update_regret(double& R, double r, int t) const {
    R += r;
    if (clips_regrets()) {
      if (R < 0.0) R = 0.0;
    } else if (discounts()) {
      R *= regret_discount(t, R >= 0.0);
    }
  }

  std::string name() const {
    switch (variant) {
      case CFRVariant::kVanilla:
        return alternating ? "CFR(alt)" : "CFR(sim)";
      case CFRVariant::kCFRPlus:
        return "CFR+";
      case CFRVariant::kDCFR:
        return "DCFR";
      case CFRVariant::kLinear:
        return "LCFR";
    }
    return "?";
  }
};

// Instantaneous regret v_a - v with the tie snap described in CFRConfig.
inline double tie_snapped(double va, double v, double tol) {
  const double d = va - v;
  return std::abs(d) <= tol * (std::abs(va) + std::abs(v)) ? 0.0 : d;
}

// Regret matching: σ(a) prop. to max(R(a), 0); uniform when no action has positive regret.
inline void regret_matching(const double* R, double* sigma, int n) {
  double total = 0.0;
  for (int a = 0; a < n; ++a) total += R[a] > 0.0 ? R[a] : 0.0;
  if (total > 0.0) {
    const double inv = 1.0 / total;
    for (int a = 0; a < n; ++a) sigma[a] = (R[a] > 0.0 ? R[a] : 0.0) * inv;
  } else {
    const double u = 1.0 / n;
    for (int a = 0; a < n; ++a) sigma[a] = u;
  }
}

}  // namespace aegis
