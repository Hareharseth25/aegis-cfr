#include "solvers/subgame_resolver/action_translation.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "util/check.h"

namespace aegis::poker {

double pseudo_harmonic_low_prob(double A, double B, double x) {
  AEGIS_CHECK_MSG(A <= B, "pseudo-harmonic mapping needs A <= B");
  if (x <= A) return 1.0;
  if (x >= B) return 0.0;
  return ((B - x) * (1.0 + A)) / ((B - A) * (1.0 + x));
}

namespace {

bool is_size(ActionKind k) { return k == ActionKind::kBet || k == ActionKind::kRaise || k == ActionKind::kAllIn; }

// Reference child index (0-based among n's children) for abstract child j of m.
int abs_to_ref(const PublicTree& ref, int n, const PublicTree& abs, int m, int j) {
  const auto& rn = ref.nodes[n];
  const auto& a = abs.nodes[abs.nodes[m].first_child + j].action;
  int nearest = -1;
  double best = 1e300;
  for (int r = 0; r < rn.num_children; ++r) {
    const auto& b = ref.nodes[rn.first_child + r].action;
    if (!is_size(a.kind)) {
      if (b.kind == a.kind) return r;
      continue;
    }
    if (!is_size(b.kind)) continue;
    if (a.kind == ActionKind::kAllIn && b.kind == ActionKind::kAllIn) return r;
    if (a.kind == b.kind && a.nominal >= 0 && std::abs(a.nominal - b.nominal) < 1e-9) return r;
    const double d = std::abs(a.fraction - b.fraction);
    if (d < best) {
      best = d;
      nearest = r;
    }
  }
  if (nearest < 0 && is_size(a.kind)) {
    // The reference node allows no bet/raise (e.g. the player already faces an
    // all-in there, while the abstract path had a near-all-in raise leaving a
    // chip behind). Facing an all-in, "raise all-in" is economically a call.
    for (int r = 0; r < rn.num_children; ++r) {
      const auto k = ref.nodes[rn.first_child + r].action.kind;
      if (k == ActionKind::kCall || k == ActionKind::kCheck) return r;
    }
  }
  AEGIS_CHECK_MSG(nearest >= 0, "abstract action '" << a.to_string() << "' has no counterpart in the reference tree at "
                                                    << ref.path_string(n) << " (abstract node " << abs.path_string(m)
                                                    << ")");
  return nearest;
}

struct Component {
  int abs_node;
  std::vector<double> W[2];  // public translation weight x own reach, per hand
};

class Translator {
 public:
  Translator(const SubgameContext& ref, const SubgameContext& abs, const VectorPolicy& ap, TranslationMethod m)
      : ref_(ref), abs_(abs), ap_(ap), method_(m), rt_(ref.tree()), at_(abs.tree()) {
    for (int p = 0; p < 2; ++p)
      AEGIS_CHECK_MSG(ref.hands(p).combos == abs.hands(p).combos,
                      "reference and abstract contexts must use the same hand lists");
    out_.strategy.resize(rt_.num_decision_nodes());
  }

  VectorPolicy run() {
    Component root;
    root.abs_node = 0;
    for (int p = 0; p < 2; ++p) root.W[p].assign(ref_.n(p), 1.0);
    walk(0, {root});
    return std::move(out_);
  }

 private:
  bool has_size_action(int m) const {
    const auto& an = at_.nodes[m];
    for (int j = 0; j < an.num_children; ++j)
      if (is_size(at_.nodes[an.first_child + j].action.kind)) return true;
    return false;
  }

  // With no component left (every abstract counterpart dropped), the subtree
  // falls back to uniform play; counted for diagnostics.
  void uniform_subtree(int n) {
    const auto& rn = rt_.nodes[n];
    if (rn.type == PNodeType::kDecision) {
      const int np = ref_.n(rn.player);
      out_.strategy[rn.decision_index].assign(static_cast<size_t>(rn.num_children) * np, 1.0 / rn.num_children);
      ++fallback_nodes_;
    }
    for (int k = 0; k < rn.num_children; ++k) uniform_subtree(rn.first_child + k);
  }

  void walk(int n, std::vector<Component> M) {
    const auto& rn = rt_.nodes[n];
    if (rn.type == PNodeType::kFold || rn.type == PNodeType::kShowdown) return;
    if (M.empty()) {
      uniform_subtree(n);
      return;
    }
    AEGIS_CHECK_MSG(rn.type != PNodeType::kLeaf, "translation does not support depth-limited trees");
    if (rn.type == PNodeType::kChance) {
      for (int k = 0; k < rn.num_children; ++k) {
        const int rc = rn.first_child + k;
        const Card card = rt_.nodes[rc].action.card;
        std::vector<Component> child;
        for (auto& comp : M) {
          const auto& an = at_.nodes[comp.abs_node];
          if (an.type != PNodeType::kChance) {
            ++dropped_;
            continue;
          }
          int match = -1;
          for (int j = 0; j < an.num_children; ++j)
            if (at_.nodes[an.first_child + j].action.card == card) match = an.first_child + j;
          AEGIS_CHECK(match >= 0);
          Component c = comp;
          c.abs_node = match;
          child.push_back(std::move(c));
        }
        walk(rc, std::move(child));
      }
      return;
    }

    const int p = rn.player;
    const int np = ref_.n(p);
    // Components that cannot continue here (e.g. a bet translated to all-in,
    // so the abstract river is a pure showdown while the reference game still
    // has betting) are dropped; the others carry the strategy.
    {
      std::vector<Component> aligned;
      for (auto& comp : M) {
        const auto& an = at_.nodes[comp.abs_node];
        if (an.type == PNodeType::kDecision && an.player == p) aligned.push_back(std::move(comp));
        else ++dropped_;
      }
      M.swap(aligned);
      if (M.empty()) {
        uniform_subtree(n);
        return;
      }
    }
    // 1. Reference strategy at n.
    auto& sigma = out_.strategy[rn.decision_index];
    sigma.assign(static_cast<size_t>(rn.num_children) * np, 0.0);
    std::vector<double> norm(np, 0.0);
    for (const auto& comp : M)
      for (int i = 0; i < np; ++i) norm[i] += comp.W[p][i];
    std::vector<std::vector<int>> amap(M.size());
    for (size_t k = 0; k < M.size(); ++k) {
      const int m = M[k].abs_node;
      const auto& an = at_.nodes[m];
      const int na = abs_.n(p);
      const double* s = ap_.strategy[an.decision_index].data();
      amap[k].resize(an.num_children);
      for (int j = 0; j < an.num_children; ++j) {
        const int r = abs_to_ref(rt_, n, at_, m, j);
        amap[k][j] = r;
        double* dst = sigma.data() + static_cast<size_t>(r) * np;
        const double* src = s + static_cast<size_t>(j) * na;
        for (int i = 0; i < np; ++i) {
          const double w = norm[i] > 0.0 ? M[k].W[p][i] / norm[i] : 1.0 / M.size();
          dst[i] += w * src[i];
        }
      }
    }
    // 2. Descend into every reference child.
    for (int r = 0; r < rn.num_children; ++r) {
      const int rc = rn.first_child + r;
      std::map<int, Component> next;  // merge by abstract node
      auto add = [&](int abs_child, const std::vector<double>* W, double scale, const double* own_sigma) {
        auto it = next.find(abs_child);
        if (it == next.end()) {
          Component c;
          c.abs_node = abs_child;
          for (int q = 0; q < 2; ++q) c.W[q].assign(ref_.n(q), 0.0);
          it = next.emplace(abs_child, std::move(c)).first;
        }
        for (int q = 0; q < 2; ++q) {
          const int nq = ref_.n(q);
          for (int i = 0; i < nq; ++i) {
            double v = W[q][i] * scale;
            if (q == p && own_sigma) v *= own_sigma[i];
            it->second.W[q][i] += v;
          }
        }
      };
      for (size_t k = 0; k < M.size(); ++k) {
        const int m = M[k].abs_node;
        const auto& an = at_.nodes[m];
        const int na = abs_.n(p);
        const double* s = ap_.strategy[an.decision_index].data();
        bool on_tree = false;
        for (int j = 0; j < an.num_children; ++j) {
          if (amap[k][j] != r) continue;
          on_tree = true;
          add(an.first_child + j, M[k].W, 1.0, s + static_cast<size_t>(j) * na);
        }
        if (!on_tree) {
          // A component whose abstract node cannot express this action (e.g.
          // a raise after a bet that was translated to all-in) is dropped for
          // this branch; the remaining components carry the strategy.
          if (!has_size_action(m) && is_size(rt_.nodes[rc].action.kind)) {
            ++dropped_;
            continue;
          }
          for (const auto& [abs_child, q] : map_offtree_action(rt_, rc, at_, m, method_))
            if (q > 0.0) add(abs_child, M[k].W, q, nullptr);
        }
      }
      std::vector<Component> child;
      for (auto& [key, comp] : next) child.push_back(std::move(comp));
      walk(rc, std::move(child));
    }
  }

  const SubgameContext& ref_;
  const SubgameContext& abs_;
  const VectorPolicy& ap_;
  TranslationMethod method_;
  const PublicTree& rt_;
  const PublicTree& at_;
  VectorPolicy out_;

 public:
  long long dropped_ = 0;
  long long fallback_nodes_ = 0;
};

}  // namespace

std::vector<std::pair<int, double>> map_offtree_action(const PublicTree& ref, int ref_child, const PublicTree& abs,
                                                       int abs_node, TranslationMethod method) {
  const auto& a = ref.nodes[ref_child].action;
  const auto& an = abs.nodes[abs_node];
  if (!is_size(a.kind)) {
    for (int j = 0; j < an.num_children; ++j)
      if (abs.nodes[an.first_child + j].action.kind == a.kind) return {{an.first_child + j, 1.0}};
    AEGIS_CHECK_MSG(false, "no abstract counterpart for " << a.to_string());
  }
  std::vector<std::pair<double, int>> sizes;  // (fraction, abs child)
  for (int j = 0; j < an.num_children; ++j) {
    const auto& b = abs.nodes[an.first_child + j].action;
    if (!is_size(b.kind)) continue;
    if (a.kind == ActionKind::kAllIn && b.kind == ActionKind::kAllIn) return {{an.first_child + j, 1.0}};
    sizes.emplace_back(b.fraction, an.first_child + j);
  }
  AEGIS_CHECK_MSG(!sizes.empty(), "abstract node has no bet/raise to translate '" << a.to_string()
                                                                                    << "' into (raise caps differ?)");
  std::sort(sizes.begin(), sizes.end());
  const double x = a.fraction;
  if (x <= sizes.front().first) return {{sizes.front().second, 1.0}};
  if (x >= sizes.back().first) return {{sizes.back().second, 1.0}};
  for (size_t k = 0; k + 1 < sizes.size(); ++k) {
    const double A = sizes[k].first, B = sizes[k + 1].first;
    if (x >= A && x <= B) {
      if (method == TranslationMethod::kNearest)
        return {{(x - A <= B - x) ? sizes[k].second : sizes[k + 1].second, 1.0}};
      const double pa = pseudo_harmonic_low_prob(A, B, x);
      return {{sizes[k].second, pa}, {sizes[k + 1].second, 1.0 - pa}};
    }
  }
  return {{sizes.back().second, 1.0}};
}

VectorPolicy translate_policy(const SubgameContext& ref, const SubgameContext& abs, const VectorPolicy& abs_policy,
                              TranslationMethod method) {
  Translator t(ref, abs, abs_policy, method);
  return t.run();
}

}  // namespace aegis::poker
