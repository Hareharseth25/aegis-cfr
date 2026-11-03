#include "solvers/subgame_resolver/leaf_values.h"

#include <cmath>
#include <vector>

#include "util/check.h"
#include "util/timer.h"

namespace aegis::poker {

namespace {

// Local reach vector -> 1326 combo vector, zeroing combos that contain `card`.
void to_combo(const HandSpace& H, const std::vector<double>& local, Card card, double* out) {
  std::fill(out, out + kNumCombos, 0.0);
  for (int i = 0; i < H.size(); ++i) {
    if (card >= 0 && (H.cards[i][0] == card || H.cards[i][1] == card)) continue;
    out[H.combos[i]] = local[i];
  }
}

std::vector<Card> live_river_cards(const std::vector<Card>& turn_board) {
  const CardMask bm = mask_of(turn_board);
  std::vector<Card> out;
  for (Card c = 0; c < kNumCards; ++c)
    if (!(bm & card_bit(c))) out.push_back(c);
  return out;
}

}  // namespace

void NeuralLeafProvider::evaluate(const SubgameContext& ctx, std::span<Query> qs) {
  Stopwatch sw;
  const auto& tree = ctx.tree();
  struct Row {
    size_t q;
    Card card;
    double pot, mass[2];
    std::vector<double> r[2];
  };
  std::vector<Row> rows;
  for (size_t k = 0; k < qs.size(); ++k) {
    const auto& nd = tree.nodes[qs[k].node];
    const auto& board = tree.boards[nd.board_id];
    AEGIS_CHECK_MSG(board.size() == 4, "neural leaves are defined at the end of the turn");
    const double pot = nd.commit[0] + nd.commit[1];
    for (Card c : live_river_cards(board)) {
      Row row;
      row.q = k;
      row.card = c;
      row.pot = pot;
      for (int p = 0; p < 2; ++p) {
        row.r[p].assign(kNumCombos, 0.0);
        to_combo(ctx.hands(p), *qs[k].reach[p], c, row.r[p].data());
        double m = 0.0;
        for (double x : row.r[p]) m += x;
        row.mass[p] = m;
      }
      if (row.mass[0] <= 0.0 || row.mass[1] <= 0.0) continue;
      rows.push_back(std::move(row));
    }
  }
  if (rows.empty()) return;
  const int B = static_cast<int>(rows.size());
  std::vector<float> x(static_cast<size_t>(B) * nn::kInputSize);
  for (int b = 0; b < B; ++b) {
    const auto& row = rows[b];
    const auto& nd = tree.nodes[qs[row.q].node];
    std::vector<Card> board = tree.boards[nd.board_id];
    board.push_back(row.card);
    const double stack = tree.stack_remaining(qs[row.q].node, 0);
    nn::build_features(board, row.r[0].data(), row.r[1].data(), row.pot, stack, &x[static_cast<size_t>(b) * nn::kInputSize]);
  }
  auto y = net_.forward(x, B);
  for (int b = 0; b < B; ++b) {
    const auto& row = rows[b];
    float* v0 = &y[static_cast<size_t>(b) * nn::kOutputSize];
    float* v1 = v0 + kNumCombos;
    if (zero_sum_) {
      const float* f = &x[static_cast<size_t>(b) * nn::kInputSize];
      std::vector<double> r0n(kNumCombos), r1n(kNumCombos);
      for (int i = 0; i < kNumCombos; ++i) {
        r0n[i] = f[nn::kBoardFeatures + i];
        r1n[i] = f[nn::kBoardFeatures + kNumCombos + i];
      }
      nn::zero_sum_project(r0n.data(), r1n.data(), v0, v1);
    }
    const auto& nd = tree.nodes[qs[row.q].node];
    const double w = ctx.chance_weight(qs[row.q].node);
    float* v[2] = {v0, v1};
    for (int p = 0; p < 2; ++p) {
      const auto& H = ctx.hands(p);
      auto& out = *qs[row.q].cfv[p];
      const double scale = w * row.pot * row.mass[1 - p];
      for (int i = 0; i < H.size(); ++i) {
        if (H.cards[i][0] == row.card || H.cards[i][1] == row.card) continue;
        out[i] += scale * v[p][H.combos[i]];
      }
    }
    (void)nd;
  }
  stats_.calls++;
  stats_.rows += B;
  stats_.seconds += sw.seconds();
}

int BlueprintLeafProvider::map_leaf(const SubgameContext& ctx, int leaf) {
  auto it = leaf_to_full_.find(leaf);
  if (it != leaf_to_full_.end()) return it->second;
  const auto& dl = ctx.tree();
  std::vector<std::string> labels;
  for (int n = leaf; n > 0; n = dl.nodes[n].parent) labels.push_back(dl.nodes[n].action.to_string());
  int cur = 0;
  for (auto l = labels.rbegin(); l != labels.rend(); ++l) {
    cur = full_.tree().child_by_label(cur, *l);
    AEGIS_CHECK_MSG(cur >= 0, "blueprint tree has no node for depth-limited path " << dl.path_string(leaf));
  }
  AEGIS_CHECK_MSG(full_.tree().nodes[cur].type == PNodeType::kChance,
                  "blueprint node for a leaf must be the river chance node");
  leaf_to_full_[leaf] = cur;
  return cur;
}

void BlueprintLeafProvider::evaluate(const SubgameContext& ctx, std::span<Query> qs) {
  Stopwatch sw;
  for (int p = 0; p < 2; ++p)
    AEGIS_CHECK_MSG(ctx.hands(p).combos == full_.hands(p).combos, "blueprint and depth-limited hand lists differ");
  for (auto& q : qs) {
    const int full_node = map_leaf(ctx, q.node);
    std::vector<double> reach[2] = {*q.reach[0], *q.reach[1]}, out[2];
    full_.subtree_values(blueprint_, full_node, reach, -1, out);
    *q.cfv[0] = out[0];
    *q.cfv[1] = out[1];
  }
  stats_.calls++;
  stats_.rows += static_cast<long long>(qs.size());
  stats_.seconds += sw.seconds();
}

RiverSolveOutput solve_river_values(const std::vector<Card>& board5, const double* r0, const double* r1, double pot,
                                    double stack, const StreetSizes& sizes, int iterations, CFRConfig cfr,
                                    const double* support0, const double* support1) {
  AEGIS_CHECK(board5.size() == 5);
  TreeConfig c;
  c.board = board5;
  c.pot = pot;
  c.effective_stack = stack;
  c.set_river_sizes(sizes);
  auto tree = PublicTree::build(c);
  const Range R0 = Range::from_vector(std::span<const double>(r0, kNumCombos));
  const Range R1 = Range::from_vector(std::span<const double>(r1, kNumCombos));
  // Every live hand is kept (zero-weight ones too): a hand that is currently
  // unreachable still needs a value, because counterfactual regret for the
  // action that would have led here depends on it.
  Range S0, S1;
  if (support0) S0 = Range::from_vector(std::span<const double>(support0, kNumCombos));
  if (support1) S1 = Range::from_vector(std::span<const double>(support1, kNumCombos));
  SubgameContext ctx(tree, R0, R1, 1, {support0 == nullptr, support1 == nullptr},
                     {support0 ? &S0 : nullptr, support1 ? &S1 : nullptr});
  VectorCFR solver(ctx, {cfr});
  solver.iterate(iterations);
  const auto avg = solver.average_policy();
  const auto vals = ctx.evaluate(avg);
  RiverSolveOutput out;
  for (int p = 0; p < 2; ++p) {
    out.cfv[p].assign(kNumCombos, 0.0);
    const auto& H = ctx.hands(p);
    // Reached hands: values under the average strategy (exactly zero-sum).
    // Unreached hands: counterfactual best-response values (their own
    // strategy is arbitrary at equilibrium; the BR value is well defined).
    const auto br = ctx.best_response(avg, p).root_cfv[p];
    for (int i = 0; i < H.size(); ++i)
      out.cfv[p][H.combos[i]] = H.root_weight[i] > 0.0 ? vals.root_cfv[p][i] : br[i];
  }
  out.exploitability_pct_pot = ctx.exploitability(avg).exploitability_pct_pot;
  return out;
}

void OracleLeafProvider::evaluate(const SubgameContext& ctx, std::span<Query> qs) {
  Stopwatch sw;
  const auto& tree = ctx.tree();
  for (auto& q : qs) {
    const auto& nd = tree.nodes[q.node];
    const auto& board = tree.boards[nd.board_id];
    const double pot = nd.commit[0] + nd.commit[1];
    const double stack = tree.stack_remaining(q.node, 0);
    const double w = ctx.chance_weight(q.node);
    for (Card c : live_river_cards(board)) {
      std::vector<double> r[2], sup[2];
      for (int p = 0; p < 2; ++p) {
        r[p].assign(kNumCombos, 0.0);
        to_combo(ctx.hands(p), *q.reach[p], c, r[p].data());
        // Support = every hand in the turn hand list (prior > 0) not blocked by c.
        sup[p].assign(kNumCombos, 0.0);
        to_combo(ctx.hands(p), std::vector<double>(ctx.n(p), 1.0), c, sup[p].data());
      }
      double m0 = 0, m1 = 0;
      for (int i = 0; i < kNumCombos; ++i) {
        m0 += r[0][i];
        m1 += r[1][i];
      }
      if (m0 <= 0.0 || m1 <= 0.0) continue;
      std::vector<Card> b5 = board;
      b5.push_back(c);
      const auto res = solve_river_values(b5, r[0].data(), r[1].data(), pot, stack, sizes_, iterations_, cfr_,
                                          sup[0].data(), sup[1].data());
      for (int p = 0; p < 2; ++p) {
        const auto& H = ctx.hands(p);
        for (int i = 0; i < H.size(); ++i) (*q.cfv[p])[i] += w * res.cfv[p][H.combos[i]];
      }
      stats_.rows++;
    }
  }
  stats_.calls++;
  stats_.seconds += sw.seconds();
}

}  // namespace aegis::poker
