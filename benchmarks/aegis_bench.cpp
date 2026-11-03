// AEGIS performance benchmarks. Prints one JSON object per line:
//   {"bench": ..., "config": ..., metrics...}
//
// Usage: aegis_bench [--quick] [--only NAME]
// Hardware counters (cycles, instructions, cache/branch misses) are recorded
// when the platform allows it (Linux perf_event; macOS kperf requires sudo).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/best_response/best_response.h"
#include "games/leduc.h"
#include "poker/hand_evaluator/evaluator.h"
#include "solvers/mccfr/mccfr.h"
#include "solvers/subgame_resolver/vector_cfr.h"
#include "solvers/vanilla_cfr/cfr.h"
#include "solvers/vanilla_cfr/naive_cfr.h"
#include "util/hw_counters.h"
#include "util/rng.h"
#include "util/thread_pool.h"
#include "util/timer.h"
#include "value_net/mlp.h"

using namespace aegis;
using namespace aegis::poker;

namespace {

bool g_quick = false;
std::string g_only;
HwCounters* g_hw = nullptr;

bool enabled(const char* name) { return g_only.empty() || g_only == name; }

void emit(const char* bench, const std::string& config, const HwSample& s, const std::string& metrics) {
  std::printf("{\"bench\":\"%s\",\"config\":\"%s\",\"wall_s\":%.6f,\"cpu_s\":%.6f,\"max_rss_kb\":%ld", bench,
              config.c_str(), s.wall_seconds, s.user_cpu_seconds, s.max_rss_kb);
  if (s.counters_valid)
    std::printf(",\"cycles\":%llu,\"instructions\":%llu,\"ipc\":%.3f,\"cache_misses\":%llu,\"branch_misses\":%llu",
                static_cast<unsigned long long>(s.cycles), static_cast<unsigned long long>(s.instructions), s.ipc(),
                static_cast<unsigned long long>(s.cache_misses), static_cast<unsigned long long>(s.branch_misses));
  std::printf(",%s}\n", metrics.c_str());
  std::fflush(stdout);
}

std::string kv(const char* k, double v) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "\"%s\":%.6g", k, v);
  return buf;
}

void bench_evaluator() {
  Rng rng(1);
  const int n = g_quick ? 2'000'000 : 20'000'000;
  std::vector<CardMask> hands(1 << 16);
  for (auto& h : hands) {
    CardMask m = 0;
    while (popcount(m) < 7) m |= card_bit(static_cast<Card>(rng.below(52)));
    h = m;
  }
  g_hw->start();
  uint64_t acc = 0;
  for (int i = 0; i < n; ++i) acc += evaluate(hands[i & 0xFFFF]);
  const auto s = g_hw->stop();
  emit("evaluator_7card", "random hands", s,
       kv("evals_per_sec", n / s.wall_seconds) + "," + kv("ns_per_eval", 1e9 * s.wall_seconds / n) + "," +
           kv("checksum", static_cast<double>(acc % 1000003)));
}

void bench_tabular() {
  LeducGame g;
  auto t = GameTree::build(g);
  const int iters = g_quick ? 100 : 500;
  {
    TabularCFR cfr(t, CFRConfig::vanilla(true));
    g_hw->start();
    cfr.iterate(iters);
    const auto s = g_hw->stop();
    emit("tabular_cfr", "leduc flat arrays", s,
         kv("iterations_per_sec", iters / s.wall_seconds) + "," + kv("nodes_per_sec", cfr.nodes_touched() / s.wall_seconds));
  }
  {
    NaiveCFR cfr(g);
    g_hw->start();
    cfr.iterate(iters);
    const auto s = g_hw->stop();
    emit("tabular_cfr", "leduc object-per-node", s,
         kv("iterations_per_sec", iters / s.wall_seconds) + "," + kv("nodes_per_sec", cfr.nodes_touched() / s.wall_seconds));
  }
  for (int ranks : {3, 6, 9}) {
    LeducConfig c;
    c.num_ranks = ranks;
    LeducGame lg(c);
    Stopwatch sw;
    auto lt = GameTree::build(lg);
    const double build = sw.seconds();
    TabularCFR cfr(lt, CFRConfig::cfr_plus());
    g_hw->start();
    cfr.iterate(g_quick ? 5 : 20);
    const auto s = g_hw->stop();
    emit("tabular_scaling", "leduc ranks=" + std::to_string(ranks), s,
         kv("nodes", lt.num_nodes()) + "," + kv("infosets", lt.num_infosets()) + "," + kv("build_s", build) + "," +
             kv("nodes_per_sec", cfr.nodes_touched() / s.wall_seconds));
  }
}

void bench_mccfr() {
  LeducGame g;
  auto t = GameTree::build(g);
  const int n = g_quick ? 20000 : 100000;
  {
    MCCFRConfig c;
    MCCFR m(t, c);
    g_hw->start();
    m.iterate(n);
    const auto s = g_hw->stop();
    emit("mccfr", "external serial", s,
         kv("traversals_per_sec", 2.0 * n / s.wall_seconds) + "," + kv("nodes_per_sec", m.nodes_touched() / s.wall_seconds));
  }
  for (int th : {1, 2, 4, 8}) {
    MCCFRConfig c;
    MCCFR m(t, c);
    const int batches = n / (100 * th) + 1;
    g_hw->start();
    m.iterate_parallel(batches, 100, th);
    const auto s = g_hw->stop();
    emit("mccfr", "external parallel threads=" + std::to_string(th), s,
         kv("threads", th) + "," + kv("traversals_per_sec", 2.0 * batches * 100 * th / s.wall_seconds) + "," +
             kv("nodes_per_sec", m.nodes_touched() / s.wall_seconds));
  }
}

TreeConfig river_cfg() {
  TreeConfig c;
  c.board = parse_cards("AsKh8c4d2s");
  c.pot = 100;
  c.effective_stack = 150;
  StreetSizes s;
  s.bet_fractions = {0.33, 0.75, 1.5};
  s.raise_fractions = {1.0};
  s.max_raises = 3;
  c.set_sizes(s, s);
  return c;
}

void bench_river() {
  auto tree = PublicTree::build(river_cfg());
  struct R {
    const char* name;
    Range r0, r1;
  };
  std::vector<R> ranges = {{"full 1081x1081", Range::full(), Range::full()},
                           {"medium", Range::parse("22+,A2+,K9+,QT+,JT,T9s,98s,87s,76s,65s"),
                            Range::parse("22+,A2+,K7+,Q9+,J9+,T9,98s,87s,76s")},
                           {"narrow", Range::parse("AA,KK,AK,AQs,88,44"), Range::parse("QQ-99,AQ,AJs,KQ")}};
  for (auto& rr : ranges) {
    SubgameContext ctx(tree, rr.r0, rr.r1);
    VectorCFR v(ctx, {CFRConfig::dcfr()});
    const int iters = g_quick ? 100 : 500;
    g_hw->start();
    v.iterate(iters);
    const auto s = g_hw->stop();
    Stopwatch sw;
    const auto e = ctx.exploitability(v.average_policy());
    const double br = sw.seconds();
    emit("river_cfr", std::string("dcfr ") + rr.name, s,
         kv("hands0", ctx.n(0)) + "," + kv("hands1", ctx.n(1)) + "," + kv("nodes", tree.num_nodes()) + "," +
             kv("ms_per_iteration", 1e3 * s.wall_seconds / iters) + "," +
             kv("hand_node_updates_per_sec", v.hand_updates() / s.wall_seconds) + "," +
             kv("exploitability_pct_pot", e.exploitability_pct_pot) + "," + kv("br_eval_ms", 1e3 * br) + "," +
             kv("solver_memory_mb", v.memory_bytes() / 1e6));
  }
}

void bench_turn_threads() {
  TreeConfig c;
  c.board = parse_cards("AsKh8c4d");
  c.pot = 100;
  c.effective_stack = 200;
  StreetSizes ts, rs;
  ts.bet_fractions = {0.5, 1.0};
  ts.raise_fractions = {1.0};
  ts.max_raises = 2;
  rs.bet_fractions = {0.75};
  rs.raise_fractions = {1.0};
  rs.max_raises = 2;
  c.set_sizes(ts, rs);
  auto tree = PublicTree::build(c);
  const int max_t = ThreadPool::hardware_threads();
  for (int th = 1; th <= max_t; th *= 2) {
    SubgameContext ctx(tree, Range::full(), Range::full(), th);
    VectorCFR v(ctx, {CFRConfig::dcfr()});
    const int iters = g_quick ? 3 : 10;
    g_hw->start();
    v.iterate(iters);
    const auto s = g_hw->stop();
    emit("turn_cfr_threads", "full ranges threads=" + std::to_string(th), s,
         kv("threads", th) + "," + kv("nodes", tree.num_nodes()) + "," + kv("ms_per_iteration", 1e3 * s.wall_seconds / iters) +
             "," + kv("hand_node_updates_per_sec", v.hand_updates() / s.wall_seconds) + "," +
             kv("solver_memory_mb", v.memory_bytes() / 1e6));
  }
}

void bench_value_net() {
  Rng rng(3);
  const std::vector<int> hidden = {512, 512};
  std::vector<std::array<int, 2>> shapes;
  int in = nn::kInputSize;
  for (int h : hidden) {
    shapes.push_back({h, in});
    in = h;
  }
  shapes.push_back({nn::kOutputSize, in});
  std::vector<std::vector<float>> W, B;
  for (auto& s : shapes) {
    W.emplace_back(static_cast<size_t>(s[0]) * s[1]);
    for (auto& x : W.back()) x = static_cast<float>(rng.uniform() - 0.5) * 0.02f;
    B.emplace_back(s[0], 0.0f);
  }
  auto net = nn::MLP::from_arrays(W, B, shapes);
  for (int batch : {1, 16, 64, 256, 1024}) {
    std::vector<float> x(static_cast<size_t>(batch) * nn::kInputSize);
    for (auto& v : x) v = static_cast<float>(rng.uniform());
    const int reps = std::max(1, (g_quick ? 2000 : 20000) / batch);
    g_hw->start();
    for (int r = 0; r < reps; ++r) net.forward(x, batch);
    const auto s = g_hw->stop();
    emit("value_net_cpu", "mlp 2x512 batch=" + std::to_string(batch), s,
         kv("batch", batch) + "," + kv("rows_per_sec", static_cast<double>(batch) * reps / s.wall_seconds) + "," +
             kv("ms_per_batch", 1e3 * s.wall_seconds / reps) + "," + kv("parameters", static_cast<double>(net.parameters())));
  }
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--quick")) g_quick = true;
    else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) g_only = argv[++i];
  }
  HwCounters hw;
  g_hw = &hw;
  std::printf("{\"bench\":\"meta\",\"hw_counters\":%s,\"backend\":\"%s\",\"reason\":\"%s\",\"threads\":%d,\"vectorized\":%s}\n",
              hw.available() ? "true" : "false", hw.backend().c_str(), hw.reason().c_str(), ThreadPool::hardware_threads(),
#ifdef AEGIS_NO_VECTORIZE
              "false"
#else
              "true"
#endif
  );
  if (enabled("evaluator")) bench_evaluator();
  if (enabled("tabular")) bench_tabular();
  if (enabled("mccfr")) bench_mccfr();
  if (enabled("river")) bench_river();
  if (enabled("turn")) bench_turn_threads();
  if (enabled("valuenet")) bench_value_net();
  return 0;
}
