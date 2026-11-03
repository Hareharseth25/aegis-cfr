// Python bindings for the AEGIS C++ core (module `aegis_core`).
//
// Long-running calls release the GIL. Objects that hold references to other
// objects (contexts -> trees, solvers -> contexts, samplers -> solvers) use
// keep_alive so Python can never free a referent while it is in use.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "adaptive/allocation/adaptive_sampler.h"
#include "adaptive/allocation/budget_allocator.h"
#include "adaptive/refinement/action_refinement.h"
#include "core/best_response/best_response.h"
#include "core/game_tree/game_tree.h"
#include "games/coin_toss.h"
#include "games/holdem_subgame_efg.h"
#include "games/kuhn.h"
#include "games/leduc.h"
#include "poker/equity/equity.h"
#include "solvers/mccfr/mccfr.h"
#include "solvers/subgame_resolver/action_translation.h"
#include "solvers/subgame_resolver/leaf_values.h"
#include "solvers/subgame_resolver/resolver.h"
#include "solvers/subgame_resolver/tabular_resolving.h"
#include "solvers/vanilla_cfr/cfr.h"
#include "solvers/vanilla_cfr/naive_cfr.h"
#include "util/hw_counters.h"
#include "util/thread_pool.h"
#include "value_net/data_gen.h"
#include "value_net/mlp.h"

namespace py = pybind11;
using namespace aegis;
using namespace aegis::poker;

namespace {

template <typename T>
py::array_t<T> to_np(const std::vector<T>& v) {
  py::array_t<T> a(static_cast<py::ssize_t>(v.size()));
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

template <typename T, size_t N>
py::array_t<T> to_np(const std::array<T, N>& v) {
  py::array_t<T> a(static_cast<py::ssize_t>(N));
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

template <typename T>
py::array_t<T> to_np2(const std::vector<T>& v, int rows, int cols) {
  py::array_t<T> a({rows, cols});
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

std::vector<double> from_np(const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
  return std::vector<double>(a.data(), a.data() + a.size());
}

// Leaf provider implemented in Python: fn(nodes, reach0, reach1) -> (cfv0, cfv1)
// where reach_p / cfv_p are (num_leaves x n_p) float64 arrays.
class PythonLeafProvider final : public LeafValueProvider {
 public:
  explicit PythonLeafProvider(py::function fn) : fn_(std::move(fn)) {}
  void evaluate(const SubgameContext& ctx, std::span<Query> qs) override {
    py::gil_scoped_acquire gil;
    const int L = static_cast<int>(qs.size());
    const int n0 = ctx.n(0), n1 = ctx.n(1);
    py::array_t<int> nodes(L);
    py::array_t<double> r0({L, n0}), r1({L, n1});
    for (int k = 0; k < L; ++k) {
      nodes.mutable_data()[k] = qs[k].node;
      std::copy(qs[k].reach[0]->begin(), qs[k].reach[0]->end(), r0.mutable_data() + static_cast<size_t>(k) * n0);
      std::copy(qs[k].reach[1]->begin(), qs[k].reach[1]->end(), r1.mutable_data() + static_cast<size_t>(k) * n1);
    }
    py::tuple res = fn_(nodes, r0, r1);
    auto c0 = res[0].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto c1 = res[1].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    AEGIS_CHECK_MSG(c0.size() == static_cast<py::ssize_t>(L) * n0 && c1.size() == static_cast<py::ssize_t>(L) * n1,
                    "python leaf provider returned arrays of the wrong shape");
    for (int k = 0; k < L; ++k) {
      std::copy(c0.data() + static_cast<size_t>(k) * n0, c0.data() + static_cast<size_t>(k + 1) * n0,
                qs[k].cfv[0]->begin());
      std::copy(c1.data() + static_cast<size_t>(k) * n1, c1.data() + static_cast<size_t>(k + 1) * n1,
                qs[k].cfv[1]->begin());
    }
    ++calls;
  }
  long long calls = 0;

 private:
  py::function fn_;
};

py::dict exploit_dict(const SubgameContext::Exploitability& e) {
  py::dict d;
  d["br_value"] = py::make_tuple(e.br_value[0], e.br_value[1]);
  d["policy_value"] = py::make_tuple(e.policy_value[0], e.policy_value[1]);
  d["nash_conv"] = e.nash_conv;
  d["exploitability"] = e.exploitability;
  d["exploitability_pct_pot"] = e.exploitability_pct_pot;
  return d;
}

}  // namespace

PYBIND11_MODULE(aegis_core, m) {
  m.doc() = "AEGIS: adaptive game-theoretic poker engine (C++ core)";
  m.def("hardware_threads", &ThreadPool::hardware_threads);
#ifdef AEGIS_NO_VECTORIZE
  m.attr("vectorized_build") = false;
#else
  m.attr("vectorized_build") = true;
#endif

  // ------------------------------------------------------------------ games
  py::class_<Game, std::shared_ptr<Game>>(m, "Game").def("name", &Game::name);
  py::class_<KuhnGame, Game, std::shared_ptr<KuhnGame>>(m, "KuhnGame").def(py::init<>());
  py::class_<LeducConfig>(m, "LeducConfig")
      .def(py::init<>())
      .def_readwrite("num_ranks", &LeducConfig::num_ranks)
      .def_readwrite("num_suits", &LeducConfig::num_suits)
      .def_readwrite("ante", &LeducConfig::ante)
      .def_readwrite("raise_round1", &LeducConfig::raise_round1)
      .def_readwrite("raise_round2", &LeducConfig::raise_round2)
      .def_readwrite("max_raises", &LeducConfig::max_raises);
  py::class_<LeducGame, Game, std::shared_ptr<LeducGame>>(m, "LeducGame")
      .def(py::init<LeducConfig>(), py::arg("config") = LeducConfig{});
  py::class_<CoinTossGame, Game, std::shared_ptr<CoinTossGame>>(m, "CoinTossGame")
      .def(py::init<double>(), py::arg("sell_value") = 0.5);

  // -------------------------------------------------------------- game tree
  py::enum_<NodeType>(m, "NodeType")
      .value("TERMINAL", NodeType::kTerminal)
      .value("CHANCE", NodeType::kChance)
      .value("DECISION", NodeType::kDecision);
  py::class_<InfoSetInfo>(m, "InfoSetInfo")
      .def_readonly("player", &InfoSetInfo::player)
      .def_readonly("num_actions", &InfoSetInfo::num_actions)
      .def_readonly("offset", &InfoSetInfo::offset)
      .def_readonly("depth", &InfoSetInfo::depth)
      .def_readonly("key", &InfoSetInfo::key)
      .def_readonly("actions", &InfoSetInfo::actions)
      .def_readonly("action_names", &InfoSetInfo::action_names)
      .def_readonly("nodes", &InfoSetInfo::nodes)
      .def_readonly("parent_sequence", &InfoSetInfo::parent_sequence);
  py::class_<GameTree, std::shared_ptr<GameTree>>(m, "GameTree")
      .def_static(
          "build",
          [](const Game& g, bool check_recall) {
            GameTree::BuildOptions o;
            o.check_perfect_recall = check_recall;
            return std::make_shared<GameTree>(GameTree::build(g, o));
          },
          py::arg("game"), py::arg("check_perfect_recall") = true, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("num_nodes", &GameTree::num_nodes)
      .def_property_readonly("num_infosets", &GameTree::num_infosets)
      .def_property_readonly("num_terminals", &GameTree::num_terminals)
      .def_readonly("total_actions", &GameTree::total_actions)
      .def_readonly("zero_sum", &GameTree::zero_sum)
      .def_readonly("game_name", &GameTree::game_name)
      .def_readonly("infosets", &GameTree::infosets)  // full copy: prefer infoset(i) in loops
      .def("infoset", [](const GameTree& t, int i) {
        AEGIS_CHECK_MSG(i >= 0 && i < t.num_infosets(), "infoset " << i << " out of range");
        return t.infosets[i];
      }, py::arg("i"))
      .def_readonly("public_keys", &GameTree::public_keys)
      .def("infoset_index", &GameTree::infoset_index)
      .def_property_readonly("node_type",
                             [](const GameTree& t) {
                               std::vector<int> v;
                               for (auto x : t.type) v.push_back(static_cast<int>(x));
                               return to_np(v);
                             })
      .def_property_readonly("node_player",
                             [](const GameTree& t) {
                               std::vector<int> v(t.player.begin(), t.player.end());
                               return to_np(v);
                             })
      .def_property_readonly("node_infoset", [](const GameTree& t) { return to_np(t.infoset); })
      .def_property_readonly("child_begin", [](const GameTree& t) { return to_np(t.child_begin); })
      .def_property_readonly("child_count", [](const GameTree& t) { return to_np(t.child_count); })
      .def_property_readonly("children", [](const GameTree& t) { return to_np(t.children); })
      .def_property_readonly("child_action", [](const GameTree& t) { return to_np(t.child_action); })
      .def_property_readonly("chance_prob", [](const GameTree& t) { return to_np(t.chance_prob); })
      .def_property_readonly("parent", [](const GameTree& t) { return to_np(t.parent); })
      .def_property_readonly("depth", [](const GameTree& t) { return to_np(t.depth); })
      .def_property_readonly("public_state", [](const GameTree& t) { return to_np(t.public_state); })
      .def("utility", [](const GameTree& t, int p) { return to_np(t.utility[p]); })
      .def("sequence", [](const GameTree& t, int p) { return to_np(t.sequence[p]); })
      .def("aug_infoset", [](const GameTree& t, int p) { return to_np(t.aug_infoset[p]); });

  py::class_<TabularPolicy>(m, "TabularPolicy")
      .def(py::init<const GameTree&>())
      .def_static("uniform", &TabularPolicy::uniform)
      .def_static(
          "random",
          [](const GameTree& t, uint64_t seed, double conc) {
            Rng rng(seed);
            return TabularPolicy::random(t, rng, conc);
          },
          py::arg("tree"), py::arg("seed"), py::arg("concentration") = 1.0)
      .def_property(
          "probs", [](const TabularPolicy& p) { return to_np(p.probs); },
          [](TabularPolicy& p, const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
            p.probs = from_np(a);
          })
      .def("validate", &TabularPolicy::validate, py::arg("tree"), py::arg("tol") = 1e-9)
      .def("at", [](const TabularPolicy& p, const GameTree& t, int I) {
        auto s = p.at(t, I);
        return std::vector<double>(s.begin(), s.end());
      });

  py::class_<BestResponseResult>(m, "BestResponseResult")
      .def_readonly("player", &BestResponseResult::player)
      .def_readonly("value", &BestResponseResult::value)
      .def_readonly("policy", &BestResponseResult::policy)
      .def_readonly("best_action", &BestResponseResult::best_action)
      .def_property_readonly("cf_action_value", [](const BestResponseResult& r) { return to_np(r.cf_action_value); })
      .def_property_readonly("cf_policy_value", [](const BestResponseResult& r) { return to_np(r.cf_policy_value); })
      .def_property_readonly("node_value", [](const BestResponseResult& r) { return to_np(r.node_value); });
  py::class_<ExploitabilityReport>(m, "ExploitabilityReport")
      .def_property_readonly("br_value", [](const ExploitabilityReport& r) { return py::make_tuple(r.br_value[0], r.br_value[1]); })
      .def_property_readonly("policy_value",
                             [](const ExploitabilityReport& r) { return py::make_tuple(r.policy_value[0], r.policy_value[1]); })
      .def_readonly("nash_conv", &ExploitabilityReport::nash_conv)
      .def_readonly("exploitability", &ExploitabilityReport::exploitability);
  m.def("best_response", &best_response, py::call_guard<py::gil_scoped_release>());
  m.def("expected_values", [](const GameTree& t, const TabularPolicy& p) {
    auto v = expected_values(t, p);
    return py::make_tuple(v[0], v[1]);
  });
  m.def("exploitability_report", &exploitability_report, py::call_guard<py::gil_scoped_release>());
  m.def("nash_conv", &nash_conv, py::call_guard<py::gil_scoped_release>());
  m.def("exploitability", &exploitability, py::call_guard<py::gil_scoped_release>());
  m.def("counterfactual_regrets", [](const GameTree& t, const TabularPolicy& p, int i) {
    return to_np(counterfactual_regrets(t, p, i));
  });

  // ---------------------------------------------------------------- solvers
  py::enum_<CFRVariant>(m, "CFRVariant")
      .value("VANILLA", CFRVariant::kVanilla)
      .value("CFR_PLUS", CFRVariant::kCFRPlus)
      .value("DCFR", CFRVariant::kDCFR)
      .value("LINEAR", CFRVariant::kLinear);
  py::class_<CFRConfig>(m, "CFRConfig")
      .def(py::init<>())
      .def_readwrite("variant", &CFRConfig::variant)
      .def_readwrite("alternating", &CFRConfig::alternating)
      .def_readwrite("alpha", &CFRConfig::alpha)
      .def_readwrite("beta", &CFRConfig::beta)
      .def_readwrite("gamma", &CFRConfig::gamma)
      .def_readwrite("averaging_delay", &CFRConfig::averaging_delay)
      .def_readwrite("tie_tolerance", &CFRConfig::tie_tolerance)
      .def_static("vanilla", &CFRConfig::vanilla, py::arg("alternating") = true)
      .def_static("cfr_plus", &CFRConfig::cfr_plus, py::arg("delay") = 0)
      .def_static("dcfr", &CFRConfig::dcfr, py::arg("alpha") = 1.5, py::arg("beta") = 0.0, py::arg("gamma") = 2.0)
      .def_static("linear", &CFRConfig::linear)
      .def("name", &CFRConfig::name);
  py::class_<TabularCFR>(m, "TabularCFR")
      .def(py::init<const GameTree&, CFRConfig>(), py::keep_alive<1, 2>())
      .def("iterate", &TabularCFR::iterate, py::arg("n") = 1, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("iteration", &TabularCFR::iteration)
      .def_property_readonly("nodes_touched", &TabularCFR::nodes_touched)
      .def("average_policy", &TabularCFR::average_policy)
      .def("current_policy", &TabularCFR::current_policy)
      .def_property_readonly("cumulative_regret", [](const TabularCFR& c) { return to_np(c.cumulative_regret()); })
      .def_property_readonly("strategy_sum", [](const TabularCFR& c) { return to_np(c.strategy_sum()); });
  py::class_<NaiveCFR>(m, "NaiveCFR")
      .def(py::init<const Game&>(), py::keep_alive<1, 2>())
      .def("iterate", &NaiveCFR::iterate, py::arg("n") = 1, py::call_guard<py::gil_scoped_release>())
      .def("average_strategy", &NaiveCFR::average_strategy)
      .def_property_readonly("num_infosets", &NaiveCFR::num_infosets)
      .def_property_readonly("nodes_touched", &NaiveCFR::nodes_touched);

  py::enum_<MCCFRScheme>(m, "MCCFRScheme")
      .value("EXTERNAL", MCCFRScheme::kExternal)
      .value("OUTCOME", MCCFRScheme::kOutcome)
      .value("IMPORTANCE", MCCFRScheme::kImportance);
  py::class_<MCCFRConfig>(m, "MCCFRConfig")
      .def(py::init<>())
      .def_readwrite("scheme", &MCCFRConfig::scheme)
      .def_readwrite("regret_matching_plus", &MCCFRConfig::regret_matching_plus)
      .def_readwrite("linear_averaging", &MCCFRConfig::linear_averaging)
      .def_readwrite("os_epsilon", &MCCFRConfig::os_epsilon)
      .def_readwrite("seed", &MCCFRConfig::seed);
  py::class_<SamplingPolicy>(m, "SamplingPolicy");
  py::class_<MCCFR>(m, "MCCFR")
      .def(py::init<const GameTree&, MCCFRConfig>(), py::keep_alive<1, 2>())
      .def("iterate", &MCCFR::iterate, py::arg("n") = 1, py::call_guard<py::gil_scoped_release>())
      .def("iterate_parallel", &MCCFR::iterate_parallel, py::arg("batches"), py::arg("traversals_per_thread"),
           py::arg("threads"), py::call_guard<py::gil_scoped_release>())
      .def("set_sampling_policy", &MCCFR::set_sampling_policy, py::keep_alive<1, 2>())
      .def("average_policy", &MCCFR::average_policy)
      .def("current_policy", &MCCFR::current_policy)
      .def_property_readonly("iteration", &MCCFR::iteration)
      .def_property_readonly("nodes_touched", &MCCFR::nodes_touched)
      .def_property_readonly("infoset_updates", [](const MCCFR& m) { return to_np(m.infoset_updates()); })
      .def_property_readonly("cumulative_regret", [](const MCCFR& m) { return to_np(m.cumulative_regret()); })
      .def("sample_regret_estimate", [](MCCFR& m, int p) { return to_np(m.sample_regret_estimate(p)); });

  py::enum_<AllocationMode>(m, "AllocationMode")
      .value("EXTERNAL", AllocationMode::kExternal)
      .value("RANDOM", AllocationMode::kRandom)
      .value("REACH", AllocationMode::kReach)
      .value("ADAPTIVE", AllocationMode::kAdaptive);
  py::class_<PriorityParams>(m, "PriorityParams")
      .def(py::init<>())
      .def_readwrite("alpha", &PriorityParams::alpha)
      .def_readwrite("beta", &PriorityParams::beta)
      .def_readwrite("gamma", &PriorityParams::gamma)
      .def_readwrite("delta", &PriorityParams::delta)
      .def_readwrite("eps", &PriorityParams::eps)
      .def_readwrite("lambda_", &PriorityParams::lambda)
      .def_readwrite("tau", &PriorityParams::tau)
      .def_readwrite("edge_rate", &PriorityParams::edge_rate)
      .def_readwrite("prio_floor", &PriorityParams::prio_floor);
  py::class_<AdaptiveSampler, SamplingPolicy>(m, "AdaptiveSampler")
      .def(py::init<const GameTree&, const MCCFR&, AllocationMode, PriorityParams>(), py::arg("tree"),
           py::arg("solver"), py::arg("mode"), py::arg("params") = PriorityParams{}, py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>())
      .def("score", &AdaptiveSampler::score)
      .def("reach_estimate", &AdaptiveSampler::reach_estimate)
      .def("uncertainty", &AdaptiveSampler::uncertainty)
      .def("sensitivity", &AdaptiveSampler::sensitivity)
      .def("cost", &AdaptiveSampler::cost)
      .def_property_readonly("visits", [](const AdaptiveSampler& s) { return to_np(s.visits()); });

  py::enum_<ResolveMethod>(m, "ResolveMethod")
      .value("NAIVE", ResolveMethod::kNaive)
      .value("RESOLVE", ResolveMethod::kResolve)
      .value("MAXMARGIN", ResolveMethod::kMaxMargin);
  py::class_<TabularResolveOptions>(m, "TabularResolveOptions")
      .def(py::init<>())
      .def_readwrite("method", &TabularResolveOptions::method)
      .def_readwrite("us", &TabularResolveOptions::us)
      .def_readwrite("iterations", &TabularResolveOptions::iterations)
      .def_readwrite("cfr", &TabularResolveOptions::cfr)
      .def_readwrite("use_cbv", &TabularResolveOptions::use_cbv)
      .def_readwrite("replace_opponent", &TabularResolveOptions::replace_opponent);
  py::class_<TabularResolveResult>(m, "TabularResolveResult")
      .def_readonly("policy", &TabularResolveResult::policy)
      .def_readonly("subgame_infosets", &TabularResolveResult::subgame_infosets)
      .def_readonly("constraint", &TabularResolveResult::constraint)
      .def_readonly("final_margin", &TabularResolveResult::final_margin)
      .def_readonly("gadget_follow", &TabularResolveResult::gadget_follow);
  m.def("resolve_subgame", &resolve_subgame, py::call_guard<py::gil_scoped_release>());
  m.def("nodes_with_public_state", &nodes_with_public_state);
  m.def("public_state_id", &public_state_id);

  // ------------------------------------------------------------------ cards
  m.def("parse_card", &parse_card);
  m.def("card_to_string", &card_to_string);
  m.def("parse_cards", [](const std::string& s) { return parse_cards(s); });
  m.def("combo_index", [](Card a, Card b) { return combo_index(a, b); });
  m.def("combo_to_string", &combo_to_string);
  m.def("parse_combo", [](const std::string& s) { return parse_combo(s); });
  m.def("hand_class_name", &hand_class_name);
  m.def("hand_class_index", [](const std::string& s) { return hand_class_index(s); });
  m.def("combo_cards", [] {
    py::array_t<int> a({kNumCombos, 2});
    for (int i = 0; i < kNumCombos; ++i) {
      a.mutable_data()[2 * i] = combos().cards[i][0];
      a.mutable_data()[2 * i + 1] = combos().cards[i][1];
    }
    return a;
  });
  m.def("combo_hand_class", [] { return to_np(combos().hand_class); });
  m.def("evaluate_hand", [](const std::vector<Card>& cs) {
    AEGIS_CHECK(cs.size() >= 5 && cs.size() <= 7);
    return evaluate(cs.data(), static_cast<int>(cs.size()));
  });
  m.def("evaluate_naive5", [](const std::vector<Card>& cs) {
    AEGIS_CHECK(cs.size() == 5);
    return evaluate_naive5(cs.data());
  });
  m.def("describe_hand", &describe);
  m.def("hand_category", [](HandValue v) { return category_of(v); });
  m.def("board_hand_values", [](const std::vector<Card>& b) { return to_np(board_hand_values(b)); });

  py::class_<Range>(m, "Range")
      .def(py::init<>())
      .def_static("parse", [](const std::string& s) { return Range::parse(s); })
      .def_static("full", &Range::full)
      .def_static("from_array",
                  [](const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
                    auto v = from_np(a);
                    return Range::from_vector(v);
                  })
      .def_property(
          "weights", [](const Range& r) { return to_np(r.w); },
          [](Range& r, const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
            auto v = from_np(a);
            r = Range::from_vector(v);
          })
      .def("remove", [](Range& r, const std::vector<Card>& dead) { r.remove(mask_of(dead)); return r; })
      .def("total", &Range::total)
      .def("num_nonzero", &Range::num_nonzero)
      .def("normalized", &Range::normalized)
      .def("bayes_update",
           [](const Range& r, const py::array_t<double, py::array::c_style | py::array::forcecast>& like) {
             auto v = from_np(like);
             return r.bayes_update(v);
           })
      .def("conditioned_on", [](const Range& r, const std::vector<Card>& known) { return r.conditioned_on(mask_of(known)); })
      .def("blocked_fraction", [](const Range& r, const std::vector<Card>& known) { return r.blocked_fraction(mask_of(known)); })
      .def("class_mass", [](const Range& r) { return to_np(r.class_mass()); })
      .def("class_frequency", [](const Range& r, const std::vector<Card>& dead) { return to_np(r.class_frequency(mask_of(dead))); },
           py::arg("dead") = std::vector<Card>{})
      .def("to_string", &Range::to_string)
      .def("__getitem__", [](const Range& r, int i) { return r[i]; })
      .def("__setitem__", [](Range& r, int i, double x) { r[i] = x; });
  m.def("random_range_by_strength", [](const std::vector<int>& order, uint64_t seed) {
    Rng rng(seed);
    return random_range_by_strength(order, rng);
  });

  py::class_<EquityResult>(m, "EquityResult")
      .def_readonly("win", &EquityResult::win)
      .def_readonly("tie", &EquityResult::tie)
      .def_readonly("lose", &EquityResult::lose)
      .def_readonly("std_error", &EquityResult::std_error)
      .def_readonly("samples", &EquityResult::samples)
      .def("equity", &EquityResult::equity);
  m.def("equity_exact", &equity_exact, py::call_guard<py::gil_scoped_release>());
  m.def("equity_vs_range_exact", &equity_vs_range_exact, py::call_guard<py::gil_scoped_release>());
  m.def("equity_range_vs_range_exact",
        [](const Range& h, const Range& v, const std::vector<Card>& b) {
          RangeEquity r;
          {
            py::gil_scoped_release rel;
            r = equity_range_vs_range_exact(h, v, b);
          }
          return py::make_tuple(r.overall, to_np(r.combo_equity), to_np(r.combo_weight));
        });
  m.def("equity_range_vs_range_mc", &equity_range_vs_range_mc, py::call_guard<py::gil_scoped_release>());

  // ---------------------------------------------------------------- betting
  py::class_<StreetSizes>(m, "StreetSizes")
      .def(py::init<>())
      .def_readwrite("bet_fractions", &StreetSizes::bet_fractions)
      .def_readwrite("raise_fractions", &StreetSizes::raise_fractions)
      .def_readwrite("allow_all_in", &StreetSizes::allow_all_in)
      .def_readwrite("max_raises", &StreetSizes::max_raises);
  py::class_<TreeConfig>(m, "TreeConfig")
      .def(py::init<>())
      .def_readwrite("board", &TreeConfig::board)
      .def_readwrite("pot", &TreeConfig::pot)
      .def_readwrite("effective_stack", &TreeConfig::effective_stack)
      .def_readwrite("first_player", &TreeConfig::first_player)
      .def_readwrite("min_bet", &TreeConfig::min_bet)
      .def_readwrite("depth_limited", &TreeConfig::depth_limited)
      .def_readwrite("initial_player", &TreeConfig::initial_player)
      .def_readwrite("initial_raises", &TreeConfig::initial_raises)
      .def_readwrite("initial_checks", &TreeConfig::initial_checks)
      .def_readwrite("initial_last_raise", &TreeConfig::initial_last_raise)
      .def("set_initial_street_bet", [](TreeConfig& c, double b0, double b1) {
        c.initial_street_bet[0] = b0;
        c.initial_street_bet[1] = b1;
      })
      .def("set_sizes", &TreeConfig::set_sizes)
      .def("set_river_sizes", &TreeConfig::set_river_sizes)
      .def("set_street_sizes", [](TreeConfig& c, int street, int player, const StreetSizes& s) { c.sizes[street][player] = s; })
      .def("get_street_sizes", [](const TreeConfig& c, int street, int player) { return c.sizes[street][player]; });
  py::enum_<ActionKind>(m, "ActionKind")
      .value("FOLD", ActionKind::kFold)
      .value("CHECK", ActionKind::kCheck)
      .value("CALL", ActionKind::kCall)
      .value("BET", ActionKind::kBet)
      .value("RAISE", ActionKind::kRaise)
      .value("ALLIN", ActionKind::kAllIn)
      .value("DEAL", ActionKind::kDeal);
  py::class_<ActionLabel>(m, "ActionLabel")
      .def_readonly("kind", &ActionLabel::kind)
      .def_readonly("amount", &ActionLabel::amount)
      .def_readonly("fraction", &ActionLabel::fraction)
      .def_readonly("nominal", &ActionLabel::nominal)
      .def_readonly("card", &ActionLabel::card)
      .def("to_string", &ActionLabel::to_string)
      .def("__repr__", &ActionLabel::to_string);
  py::enum_<PNodeType>(m, "PNodeType")
      .value("DECISION", PNodeType::kDecision)
      .value("FOLD", PNodeType::kFold)
      .value("SHOWDOWN", PNodeType::kShowdown)
      .value("CHANCE", PNodeType::kChance)
      .value("LEAF", PNodeType::kLeaf);
  py::class_<PublicNode>(m, "PublicNode")
      .def_readonly("type", &PublicNode::type)
      .def_readonly("player", &PublicNode::player)
      .def_readonly("street", &PublicNode::street)
      .def_readonly("parent", &PublicNode::parent)
      .def_readonly("first_child", &PublicNode::first_child)
      .def_readonly("num_children", &PublicNode::num_children)
      .def_readonly("board_id", &PublicNode::board_id)
      .def_property_readonly("commit", [](const PublicNode& n) { return py::make_tuple(n.commit[0], n.commit[1]); })
      .def_property_readonly("street_bet", [](const PublicNode& n) { return py::make_tuple(n.street_bet[0], n.street_bet[1]); })
      .def_readonly("raises", &PublicNode::raises)
      .def_readonly("action", &PublicNode::action)
      .def_readonly("depth", &PublicNode::depth)
      .def_readonly("decision_index", &PublicNode::decision_index);
  py::class_<PublicTree, std::shared_ptr<PublicTree>>(m, "PublicTree")
      .def_static("build", [](const TreeConfig& c) { return std::make_shared<PublicTree>(PublicTree::build(c)); })
      .def_property_readonly("num_nodes", &PublicTree::num_nodes)
      .def_property_readonly("num_decision_nodes", &PublicTree::num_decision_nodes)
      // `nodes` / `boards` copy the whole native vector into a Python list on
      // every access, so use them once per traversal. For single lookups use
      // node(i) / board(i), which copy one element and check the index.
      .def_readonly("nodes", &PublicTree::nodes)
      .def_readonly("boards", &PublicTree::boards)
      .def("node", [](const PublicTree& t, int i) { return t.node(i); }, py::arg("i"))
      .def("board", [](const PublicTree& t, int i) {
        AEGIS_CHECK_MSG(i >= 0 && i < static_cast<int>(t.boards.size()), "board id " << i << " out of range");
        return t.boards[i];
      }, py::arg("i"))
      .def_property_readonly("num_boards", [](const PublicTree& t) { return static_cast<int>(t.boards.size()); })
      .def("config", &PublicTree::config)
      .def("pot", &PublicTree::pot)
      .def("stack_remaining", &PublicTree::stack_remaining)
      .def("path_string", &PublicTree::path_string)
      .def("child_by_label", &PublicTree::child_by_label)
      .def("count_type", &PublicTree::count_type);

  // ------------------------------------------------------- vector solving
  py::class_<HandSpace>(m, "HandSpace")
      .def_readonly("combos", &HandSpace::combos)
      .def_property_readonly("root_weight", [](const HandSpace& h) { return to_np(h.root_weight); })
      .def_property_readonly("cards", [](const HandSpace& h) {
        py::array_t<int> a({h.size(), 2});
        for (int i = 0; i < h.size(); ++i) {
          a.mutable_data()[2 * i] = h.cards[i][0];
          a.mutable_data()[2 * i + 1] = h.cards[i][1];
        }
        return a;
      })
      .def("size", &HandSpace::size);
  py::class_<VectorPolicy>(m, "VectorPolicy")
      .def(py::init<>())
      .def_property_readonly("num_nodes", [](const VectorPolicy& p) { return p.strategy.size(); })
      .def("get", [](const VectorPolicy& p, int decision_index) { return to_np(p.strategy.at(decision_index)); })
      .def("set",
           [](VectorPolicy& p, int decision_index, const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
             p.strategy.at(decision_index) = from_np(a);
           })
      .def("copy", [](const VectorPolicy& p) { return p; });
  py::class_<SubgameContext, std::shared_ptr<SubgameContext>>(m, "SubgameContext")
      .def(py::init([](std::shared_ptr<PublicTree> t, const Range& r0, const Range& r1, int threads,
                       std::array<bool, 2> full) { return std::make_shared<SubgameContext>(*t, r0, r1, threads, full); }),
           py::arg("tree"), py::arg("r0"), py::arg("r1"), py::arg("threads") = 1,
           py::arg("full_support") = std::array<bool, 2>{false, false}, py::keep_alive<1, 2>())
      .def("hands", &SubgameContext::hands, py::return_value_policy::reference_internal)
      .def("n", &SubgameContext::n)
      .def_property_readonly("normalizer", &SubgameContext::normalizer)
      .def("uniform_policy", &SubgameContext::uniform_policy)
      .def("evaluate",
           [](const SubgameContext& c, const VectorPolicy& p) {
             SubgameContext::Values v;
             {
               py::gil_scoped_release rel;
               v = c.evaluate(p);
             }
             return py::make_tuple(py::make_tuple(v.ev[0], v.ev[1]), to_np(v.root_cfv[0]), to_np(v.root_cfv[1]));
           })
      .def("best_response",
           [](const SubgameContext& c, const VectorPolicy& p, int player) {
             SubgameContext::Values v;
             {
               py::gil_scoped_release rel;
               v = c.best_response(p, player);
             }
             return py::make_tuple(v.ev[player], to_np(v.root_cfv[player]));
           })
      .def("exploitability",
           [](const SubgameContext& c, const VectorPolicy& p) {
             SubgameContext::Exploitability e;
             {
               py::gil_scoped_release rel;
               e = c.exploitability(p);
             }
             return exploit_dict(e);
           })
      .def("subtree_values",
           [](const SubgameContext& c, const VectorPolicy& p, int node, const std::vector<double>& r0,
              const std::vector<double>& r1, int br_player) {
             std::vector<double> reach[2] = {r0, r1}, out[2];
             {
               py::gil_scoped_release rel;
               c.subtree_values(p, node, reach, br_player, out);
             }
             return py::make_tuple(to_np(out[0]), to_np(out[1]));
           })
      .def("node_reach", [](const SubgameContext& c, const VectorPolicy& p, int player) {
        auto r = c.node_reach(p, player);
        py::list l;
        for (auto& v : r) l.append(to_np(v));
        return l;
      });

  py::class_<LeafValueProvider>(m, "LeafValueProvider");
  py::class_<PythonLeafProvider, LeafValueProvider>(m, "PythonLeafProvider")
      .def(py::init<py::function>())
      .def_readonly("calls", &PythonLeafProvider::calls);
  py::class_<LeafStats>(m, "LeafStats")
      .def_readonly("calls", &LeafStats::calls)
      .def_readonly("rows", &LeafStats::rows)
      .def_readonly("seconds", &LeafStats::seconds);
  py::class_<nn::MLP>(m, "MLP")
      .def_static("load", &nn::MLP::load)
      .def_static("from_arrays", &nn::MLP::from_arrays)
      .def("save", &nn::MLP::save)
      .def_property_readonly("input_size", &nn::MLP::input_size)
      .def_property_readonly("output_size", &nn::MLP::output_size)
      .def_property_readonly("parameters", &nn::MLP::parameters)
      .def("forward", [](const nn::MLP& net, const py::array_t<float, py::array::c_style | py::array::forcecast>& x) {
        AEGIS_CHECK(x.ndim() == 2);
        const int B = static_cast<int>(x.shape(0));
        std::vector<float> in(x.data(), x.data() + x.size()), out;
        {
          py::gil_scoped_release rel;
          out = net.forward(in, B);
        }
        return to_np2(out, B, net.output_size());
      });
  m.attr("NN_INPUT_SIZE") = nn::kInputSize;
  m.attr("NN_OUTPUT_SIZE") = nn::kOutputSize;
  m.def("build_features", [](const std::vector<Card>& board, const std::vector<double>& r0,
                             const std::vector<double>& r1, double pot, double stack) {
    AEGIS_CHECK(r0.size() == kNumCombos && r1.size() == kNumCombos);
    std::vector<float> f(nn::kInputSize);
    nn::build_features(board, r0.data(), r1.data(), pot, stack, f.data());
    return to_np(f);
  });
  py::class_<NeuralLeafProvider, LeafValueProvider>(m, "NeuralLeafProvider")
      .def(py::init<nn::MLP, bool>(), py::arg("net"), py::arg("zero_sum") = true)
      .def_property_readonly("stats", &NeuralLeafProvider::stats);
  py::class_<BlueprintLeafProvider, LeafValueProvider>(m, "BlueprintLeafProvider")
      .def(py::init<const SubgameContext&, VectorPolicy>(), py::keep_alive<1, 2>())
      .def_property_readonly("stats", &BlueprintLeafProvider::stats);
  py::class_<OracleLeafProvider, LeafValueProvider>(m, "OracleLeafProvider")
      .def(py::init<StreetSizes, int, CFRConfig>(), py::arg("river_sizes"), py::arg("iterations"),
           py::arg("cfr") = CFRConfig::dcfr())
      .def_property_readonly("stats", &OracleLeafProvider::stats);

  py::enum_<GadgetConfig::Mode>(m, "GadgetMode")
      .value("RESOLVE", GadgetConfig::Mode::kResolve)
      .value("MAXMARGIN", GadgetConfig::Mode::kMaxMargin);
  py::class_<GadgetConfig>(m, "GadgetConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &GadgetConfig::enabled)
      .def_readwrite("opponent", &GadgetConfig::opponent)
      .def_readwrite("mode", &GadgetConfig::mode)
      .def_readwrite("constraint", &GadgetConfig::constraint);
  py::class_<VectorCFRConfig>(m, "VectorCFRConfig")
      .def(py::init<>())
      .def(py::init<CFRConfig>())
      .def_readwrite("cfr", &VectorCFRConfig::cfr)
      .def_readwrite("gadget", &VectorCFRConfig::gadget)
      .def_readwrite("root", &VectorCFRConfig::root)
      .def("set_root_reach", [](VectorCFRConfig& c, const std::vector<double>& r0, const std::vector<double>& r1) {
        c.root_reach[0] = r0;
        c.root_reach[1] = r1;
      });
  py::class_<VectorCFR>(m, "VectorCFR")
      .def(py::init([](const SubgameContext& ctx, VectorCFRConfig cfg, LeafValueProvider* leaf) {
             cfg.leaf = leaf;
             return std::make_unique<VectorCFR>(ctx, cfg);
           }),
           py::arg("ctx"), py::arg("config") = VectorCFRConfig{}, py::arg("leaf") = nullptr, py::keep_alive<1, 2>(),
           py::keep_alive<1, 4>())
      .def("iterate", &VectorCFR::iterate, py::arg("n") = 1, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("iteration", &VectorCFR::iteration)
      .def("average_policy", &VectorCFR::average_policy)
      .def("current_policy", &VectorCFR::current_policy)
      .def("root_cfv", [](const VectorCFR& s, int p) { return to_np(s.root_cfv(p)); })
      .def("regrets", [](const VectorCFR& s, int node) { return to_np(s.regrets(node)); })
      .def("strategy_sums", [](const VectorCFR& s, int node) { return to_np(s.strategy_sums(node)); })
      .def("gadget_root_reach", [](const VectorCFR& s) { return to_np(s.gadget_root_reach()); })
      .def_property_readonly("hand_updates", &VectorCFR::hand_updates)
      .def_property_readonly("leaf_queries", &VectorCFR::leaf_queries)
      .def_property_readonly("memory_bytes", &VectorCFR::memory_bytes);

  py::enum_<TranslationMethod>(m, "TranslationMethod")
      .value("PSEUDO_HARMONIC", TranslationMethod::kPseudoHarmonic)
      .value("NEAREST", TranslationMethod::kNearest);
  m.def("pseudo_harmonic_low_prob", &pseudo_harmonic_low_prob);
  m.def("translate_policy", &translate_policy, py::arg("ref"), py::arg("abs"), py::arg("abs_policy"),
        py::arg("method") = TranslationMethod::kPseudoHarmonic, py::call_guard<py::gil_scoped_release>());

  py::class_<VectorResolveOptions>(m, "VectorResolveOptions")
      .def(py::init<>())
      .def_readwrite("method", &VectorResolveOptions::method)
      .def_readwrite("us", &VectorResolveOptions::us)
      .def_readwrite("iterations", &VectorResolveOptions::iterations)
      .def_readwrite("cfr", &VectorResolveOptions::cfr)
      .def_readwrite("use_cbv", &VectorResolveOptions::use_cbv)
      .def_readwrite("replace_opponent", &VectorResolveOptions::replace_opponent)
      .def_readwrite("constraint", &VectorResolveOptions::constraint);
  py::class_<VectorResolveResult>(m, "VectorResolveResult")
      .def_readonly("policy", &VectorResolveResult::policy)
      .def_readonly("constraint", &VectorResolveResult::constraint)
      .def_readonly("margin", &VectorResolveResult::margin)
      .def_readonly("worst_violation", &VectorResolveResult::worst_violation)
      .def_readonly("total_violation", &VectorResolveResult::total_violation)
      .def_readonly("seconds", &VectorResolveResult::seconds)
      .def_property_readonly("root_reach", [](const VectorResolveResult& r) {
        return py::make_tuple(to_np(r.root_reach[0]), to_np(r.root_reach[1]));
      });
  m.def(
      "resolve_subtree",
      [](const SubgameContext& ctx, const VectorPolicy& base, int root, VectorResolveOptions o, LeafValueProvider* leaf) {
        o.leaf = leaf;
        py::gil_scoped_release rel;
        return resolve_subtree(ctx, base, root, o);
      },
      py::arg("ctx"), py::arg("base"), py::arg("root"), py::arg("options"), py::arg("leaf") = nullptr);
  m.def("reach_at", [](const SubgameContext& ctx, const VectorPolicy& p, int node) {
    std::vector<double> out[2];
    reach_at(ctx, p, node, out);
    return py::make_tuple(to_np(out[0]), to_np(out[1]));
  });
  m.def("opponent_constraint", [](const SubgameContext& ctx, const VectorPolicy& p, int root, int opp, bool cbv) {
    py::gil_scoped_release rel;
    return opponent_constraint(ctx, p, root, opp, cbv);
  });
  m.def("best_alternative_constraint", [](const SubgameContext& ctx, const VectorPolicy& p, int d,
                                          const std::vector<int>& allowed, int opp) {
    py::gil_scoped_release rel;
    return best_alternative_constraint(ctx, p, d, allowed, opp);
  });
  m.def("splice_policy", [](const PublicTree& t, VectorPolicy dst, const VectorPolicy& src, int root, int player) {
    splice_policy(t, dst, src, root, player);
    return dst;
  });
  m.def("street_roots", &street_roots);
  m.def("action_in_abstraction", &action_in_abstraction);

  py::class_<HoldemSubgameGame, Game, std::shared_ptr<HoldemSubgameGame>>(m, "HoldemSubgameGame")
      .def(py::init<const SubgameContext&>(), py::keep_alive<1, 2>())
      .def_static("key", &HoldemSubgameGame::key);

  // ------------------------------------------------------- data generation
  py::class_<nn::RiverSampleConfig>(m, "RiverSampleConfig")
      .def(py::init<>())
      .def_readwrite("turn_board", &nn::RiverSampleConfig::turn_board)
      .def_readwrite("pot_stack", &nn::RiverSampleConfig::pot_stack)
      .def_readwrite("river_sizes", &nn::RiverSampleConfig::river_sizes)
      .def_readwrite("iterations", &nn::RiverSampleConfig::iterations)
      .def_readwrite("cfr", &nn::RiverSampleConfig::cfr)
      .def_readwrite("uniform_mix_prob", &nn::RiverSampleConfig::uniform_mix_prob)
      .def_readwrite("seed", &nn::RiverSampleConfig::seed)
      .def_readwrite("threads", &nn::RiverSampleConfig::threads);
  m.def("generate_river_samples", [](const nn::RiverSampleConfig& cfg, int n) {
    nn::RiverSamples s;
    {
      py::gil_scoped_release rel;
      s = nn::generate_river_samples(cfg, n);
    }
    py::dict d;
    d["features"] = to_np2(s.features, s.n, nn::kInputSize);
    d["targets"] = to_np2(s.targets, s.n, nn::kOutputSize);
    d["mask"] = to_np2(s.mask, s.n, nn::kOutputSize);
    d["boards"] = to_np2(s.boards, s.n, 5);
    d["exploitability_pct_pot"] = to_np(s.exploitability_pct_pot);
    d["pot"] = to_np(s.pot);
    d["stack"] = to_np(s.stack);
    return d;
  });
  m.def("solve_river_values", [](const std::vector<Card>& b5, const std::vector<double>& r0, const std::vector<double>& r1,
                                 double pot, double stack, const StreetSizes& sizes, int iterations, CFRConfig cfr) {
    AEGIS_CHECK(r0.size() == kNumCombos && r1.size() == kNumCombos);
    RiverSolveOutput out;
    {
      py::gil_scoped_release rel;
      out = solve_river_values(b5, r0.data(), r1.data(), pot, stack, sizes, iterations, cfr);
    }
    return py::make_tuple(to_np(out.cfv[0]), to_np(out.cfv[1]), out.exploitability_pct_pot);
  });

  // ------------------------------------------------------------- adaptive
  py::enum_<BudgetPolicy>(m, "BudgetPolicy")
      .value("UNIFORM", BudgetPolicy::kUniform)
      .value("WEIGHTED", BudgetPolicy::kWeighted)
      .value("ADAPTIVE", BudgetPolicy::kAdaptive);
  py::class_<BudgetOptions>(m, "BudgetOptions")
      .def(py::init<>())
      .def_readwrite("total_work", &BudgetOptions::total_work)
      .def_readwrite("warmup_iterations", &BudgetOptions::warmup_iterations)
      .def_readwrite("chunk_iterations", &BudgetOptions::chunk_iterations)
      .def_readwrite("refit_every", &BudgetOptions::refit_every)
      .def_readwrite("ucb_kappa", &BudgetOptions::ucb_kappa)
      .def_readwrite("cfr", &BudgetOptions::cfr)
      .def_readwrite("charge_measurements", &BudgetOptions::charge_measurements);
  py::class_<BudgetResult>(m, "BudgetResult")
      .def_readonly("iterations", &BudgetResult::iterations)
      .def_readonly("exploitability", &BudgetResult::exploitability)
      .def_readonly("weighted_exploitability", &BudgetResult::weighted_exploitability)
      .def_readonly("work_used", &BudgetResult::work_used)
      .def_readonly("measurements", &BudgetResult::measurements)
      .def_readonly("seconds", &BudgetResult::seconds)
      .def_readonly("trace", &BudgetResult::trace);
  m.def("allocate_budget", [](const std::vector<std::pair<std::shared_ptr<SubgameContext>, double>>& games,
                              BudgetPolicy policy, const BudgetOptions& o) {
    std::vector<BudgetSubgame> g;
    for (const auto& [ctx, w] : games) g.push_back({ctx.get(), w});
    py::gil_scoped_release rel;
    return allocate_budget(g, policy, o);
  });

  py::class_<RefinementSpot>(m, "RefinementSpot")
      .def(py::init<>())
      .def_readwrite("base", &RefinementSpot::base)
      .def_readwrite("r0", &RefinementSpot::r0)
      .def_readwrite("r1", &RefinementSpot::r1)
      .def_readwrite("reference_bets", &RefinementSpot::reference_bets)
      .def_readwrite("iterations", &RefinementSpot::iterations)
      .def_readwrite("cfr", &RefinementSpot::cfr)
      .def_readwrite("threads", &RefinementSpot::threads)
      .def_readwrite("translation", &RefinementSpot::translation);
  py::class_<CandidateInfo>(m, "CandidateInfo")
      .def_readonly("player", &CandidateInfo::player)
      .def_readonly("node", &CandidateInfo::node)
      .def_readonly("fraction", &CandidateInfo::fraction)
      .def_readonly("hole", &CandidateInfo::hole)
      .def_readonly("hole_pct_pot", &CandidateInfo::hole_pct_pot)
      .def_readonly("reach_mass", &CandidateInfo::reach_mass)
      .def_readonly("gap_low", &CandidateInfo::gap_low)
      .def_readonly("gap_high", &CandidateInfo::gap_high);
  py::class_<AbstractionEval>(m, "AbstractionEval")
      .def_readonly("bets", &AbstractionEval::bets)
      .def_readonly("ref_exploitability_pct", &AbstractionEval::ref_exploitability_pct)
      .def_readonly("ref_nash_conv", &AbstractionEval::ref_nash_conv)
      .def_readonly("abs_exploitability_pct", &AbstractionEval::abs_exploitability_pct)
      .def_readonly("ref_ev0", &AbstractionEval::ref_ev0)
      .def_readonly("abs_nodes", &AbstractionEval::abs_nodes)
      .def_readonly("abs_infosets", &AbstractionEval::abs_infosets)
      .def_readonly("solve_seconds", &AbstractionEval::solve_seconds)
      .def_readonly("eval_seconds", &AbstractionEval::eval_seconds)
      .def_readonly("candidates", &AbstractionEval::candidates);
  py::class_<ActionRefiner, std::unique_ptr<ActionRefiner>>(m, "ActionRefiner")
      .def(py::init<RefinementSpot>())
      .def("evaluate", &ActionRefiner::evaluate, py::arg("bets"), py::arg("compute_candidates") = true,
           py::call_guard<py::gil_scoped_release>());

  // -------------------------------------------------------------- profiling
  py::class_<HwSample>(m, "HwSample")
      .def_readonly("wall_seconds", &HwSample::wall_seconds)
      .def_readonly("user_cpu_seconds", &HwSample::user_cpu_seconds)
      .def_readonly("cycles", &HwSample::cycles)
      .def_readonly("instructions", &HwSample::instructions)
      .def_readonly("cache_misses", &HwSample::cache_misses)
      .def_readonly("branch_misses", &HwSample::branch_misses)
      .def_readonly("max_rss_kb", &HwSample::max_rss_kb)
      .def_readonly("counters_valid", &HwSample::counters_valid)
      .def("ipc", &HwSample::ipc);
  py::class_<HwCounters>(m, "HwCounters")
      .def(py::init<>())
      .def_property_readonly("available", &HwCounters::available)
      .def_property_readonly("backend", &HwCounters::backend)
      .def_property_readonly("reason", &HwCounters::reason)
      .def("start", &HwCounters::start)
      .def("stop", &HwCounters::stop);
}
