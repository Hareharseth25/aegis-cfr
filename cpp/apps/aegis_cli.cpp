// Native command-line front end (no Python needed).
//
//   aegis_cli tabular --game kuhn|leduc|coin_toss --algo cfr|cfr+|dcfr|lcfr|es|os --iters N
//   aegis_cli river --board AsKh8c4d2s --pot 100 --stack 150 --r0 "<range>" --r1 "<range>"
//                   [--bets 0.33,0.75,1.5] [--raises 1.0] [--iters 1000] [--threads 1] [--algo dcfr]
//   aegis_cli equity --hero AdJd --villain "AK,QQ" --board AsKh8c
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>

#include "core/best_response/best_response.h"
#include "games/coin_toss.h"
#include "games/kuhn.h"
#include "games/leduc.h"
#include "poker/equity/equity.h"
#include "solvers/mccfr/mccfr.h"
#include "solvers/subgame_resolver/vector_cfr.h"
#include "solvers/vanilla_cfr/cfr.h"
#include "util/check.h"
#include "util/timer.h"

using namespace aegis;
using namespace aegis::poker;

namespace {

std::map<std::string, std::string> parse_args(int argc, char** argv, int start) {
  std::map<std::string, std::string> a;
  for (int i = start; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) == 0 && i + 1 < argc) a[k.substr(2)] = argv[++i];
  }
  return a;
}

std::string get(const std::map<std::string, std::string>& a, const std::string& k, const std::string& d) {
  auto it = a.find(k);
  return it == a.end() ? d : it->second;
}

std::vector<double> parse_list(const std::string& s) {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) v.push_back(std::stod(item));
  return v;
}

CFRConfig algo(const std::string& name) {
  if (name == "cfr") return CFRConfig::vanilla(true);
  if (name == "cfr+") return CFRConfig::cfr_plus();
  if (name == "lcfr") return CFRConfig::linear();
  return CFRConfig::dcfr();
}

int cmd_tabular(const std::map<std::string, std::string>& a) {
  const std::string gname = get(a, "game", "kuhn");
  std::unique_ptr<Game> g;
  if (gname == "kuhn") g = std::make_unique<KuhnGame>();
  else if (gname == "leduc") g = std::make_unique<LeducGame>();
  else g = std::make_unique<CoinTossGame>();
  auto t = GameTree::build(*g);
  const std::string an = get(a, "algo", "cfr+");
  const int iters = std::stoi(get(a, "iters", "1000"));
  std::printf("%s: %d nodes, %d information sets\n", t.game_name.c_str(), t.num_nodes(), t.num_infosets());
  Stopwatch sw;
  TabularPolicy avg;
  if (an == "es" || an == "os") {
    MCCFRConfig c;
    c.scheme = an == "es" ? MCCFRScheme::kExternal : MCCFRScheme::kOutcome;
    MCCFR m(t, c);
    m.iterate(iters);
    avg = m.average_policy();
  } else {
    TabularCFR cfr(t, algo(an));
    cfr.iterate(iters);
    avg = cfr.average_policy();
  }
  const auto rep = exploitability_report(t, avg);
  std::printf("%s x %d iterations in %.3fs: NashConv %.3e, exploitability %.3e, EV(p0) %.6f\n", an.c_str(), iters,
              sw.seconds(), rep.nash_conv, rep.exploitability, rep.policy_value[0]);
  return 0;
}

int cmd_river(const std::map<std::string, std::string>& a) {
  TreeConfig c;
  c.board = parse_cards(get(a, "board", "AsKh8c4d2s"));
  c.pot = std::stod(get(a, "pot", "100"));
  c.effective_stack = std::stod(get(a, "stack", "150"));
  StreetSizes s;
  s.bet_fractions = parse_list(get(a, "bets", "0.33,0.75,1.5"));
  s.raise_fractions = parse_list(get(a, "raises", "1.0"));
  s.max_raises = std::stoi(get(a, "max-raises", "3"));
  c.set_sizes(s, s);
  auto tree = PublicTree::build(c);
  auto r0 = Range::parse(get(a, "r0", "random")).remove(mask_of(c.board));
  auto r1 = Range::parse(get(a, "r1", "random")).remove(mask_of(c.board));
  SubgameContext ctx(tree, r0, r1, std::stoi(get(a, "threads", "1")));
  VectorCFR v(ctx, {algo(get(a, "algo", "dcfr"))});
  const int iters = std::stoi(get(a, "iters", "1000"));
  std::printf("%d public nodes, %d/%d hands\n", tree.num_nodes(), ctx.n(0), ctx.n(1));
  Stopwatch sw;
  int done = 0;
  for (int cp = 10; done < iters; cp = std::min(iters, cp * 3)) {
    v.iterate(cp - done);
    done = cp;
    const auto e = ctx.exploitability(v.average_policy());
    std::printf("  iter %6d  %.3fs  exploitability %.4f%% pot\n", done, sw.seconds(), e.exploitability_pct_pot);
  }
  const auto pol = v.average_policy();
  const auto& root = tree.nodes[0];
  const int np = ctx.n(root.player);
  const auto reach = ctx.hands(root.player).root_weight;
  double total = 0.0;
  for (double x : reach) total += x;
  std::printf("root (player %d) aggregate strategy:\n", root.player);
  for (int k = 0; k < root.num_children; ++k) {
    double f = 0.0;
    for (int i = 0; i < np; ++i) f += reach[i] * pol.strategy[root.decision_index][static_cast<size_t>(k) * np + i];
    std::printf("  %-12s %6.2f%%\n", tree.nodes[root.first_child + k].action.to_string().c_str(), 100.0 * f / total);
  }
  return 0;
}

int cmd_equity(const std::map<std::string, std::string>& a) {
  const auto board = parse_cards(get(a, "board", ""));
  const int hero = parse_combo(get(a, "hero", "AdJd"));
  const auto villain = Range::parse(get(a, "villain", "random"));
  const auto r = equity_vs_range_exact(hero, villain, board);
  std::printf("equity %.4f%% (win %.4f%%, tie %.4f%%) over %lld boards\n", 100 * r.equity(), 100 * r.win, 100 * r.tie,
              r.samples);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: aegis_cli tabular|river|equity [--options]\n");
    return 2;
  }
  try {
    const std::string cmd = argv[1];
    const auto a = parse_args(argc, argv, 2);
    if (cmd == "tabular") return cmd_tabular(a);
    if (cmd == "river") return cmd_river(a);
    if (cmd == "equity") return cmd_equity(a);
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
