# Project checklist

This is the original plan for the project, item by item, with where each part
ended up and how I checked it. "Test" means it's covered by `build/aegis_tests`
or `pytest tests/python`; "Exp" means one of the experiments in
`python/experiments/` (results in `results/` and `docs/RESULTS.md`).

## The seven stages

| # | Item | Where | How it's checked |
|---|---|---|---|
| 1 | Kuhn and Leduc with explicit information sets | `cpp/games/`, `cpp/core/game_tree/` | tree shapes match OpenSpiel (tests) |
| 1 | Vanilla CFR | `cpp/solvers/vanilla_cfr/` | converges to the Kuhn equilibrium; same iterates as OpenSpiel |
| 1 | Exact best response | `cpp/core/best_response/` | legal-strategy test; NashConv matches OpenSpiel |
| 2 | CFR+, DCFR, Linear CFR | `cpp/solvers/common/regret_rules.h` | same iterates as OpenSpiel's CFR+ and DCFR |
| 2 | External / outcome sampling MCCFR | `cpp/solvers/mccfr/` | convergence tests, unbiasedness test |
| 2 | Memory layout | flat arrays vs `NaiveCFR` | same iterates; ~27x throughput (Exp 9) |
| 2 | Profiling | `benchmarks/aegis_bench.cpp`, `cpp/util/hw_counters.*` | Exp 9 |
| 2 | Parallel runs | parallel MCCFR, threaded turn trees | determinism tests |
| 3 | 7-card evaluator | `cpp/poker/hand_evaluator/` | every 5-card hand vs a naive evaluator; category counts for all 133.8M 7-card hands |
| 3 | Exact equity | `cpp/poker/equity/` | brute-force and Monte Carlo cross-checks |
| 3 | Ranges, card removal, blockers, Bayesian updates | `cpp/poker/ranges/` | range tests, Python cross-checks |
| 4 | River solver with arbitrary ranges, stacks and sizes | `cpp/poker/betting_rules/`, `cpp/solvers/subgame_resolver/` | analytic river test; same iterates as tabular CFR |
| 4 | Real mixed strategies | per-hand strategies in the spot API | Exp 4 lists hands that mix |
| 5 | Depth-limited turn search | `leaf_values.*` | oracle-leaf test matches the full tree; Exp 8 |
| 5 | Blueprint values | `BlueprintLeafProvider` | test; Exp 8 |
| 5 | Safe re-solving | `resolver.*`, `tabular_resolving.*`, gadget in `vector_cfr.*` | Coin Toss and Leduc tests; Exp 7 |
| 6 | Regret-guided refinement | `adaptive/refinement/` | Exp B (exp10) |
| 6 | Compute budgets | `adaptive/allocation/budget_allocator.*` | budget tests (incl. tiny/boundary budgets); Exp 12 |
| 6 | Adaptive abstraction | Exp B, Exp C | |
| 7 | Exploitability audits | everywhere; best-response audit in the UI | |
| 7 | Benchmarks and ablations | Exp 9 (SIMD on/off, layout, threads, CPU vs GPU), Exp A ablations | |
| 7 | Visualisations | `frontend/` + `python/aegis/server.py` | server tests |
| 7 | Reproducible experiments | `run_all.py`, fixed seeds, `report.py` | `results/run_log.json` |

## Maths

| Item | Where |
|---|---|
| Extensive-form game interface | `core/extensive_game/game.h` |
| Indistinguishable histories share a policy | information-set keys + consistency and perfect-recall checks in `GameTree` |
| Counterfactual regret, regret matching | `regret_rules.h`, `cfr.cpp` |
| Values weighted by opponent + chance reach, not own reach | `cfr.cpp`, `vector_cfr.cpp` |
| CFR+ with linear averaging, DCFR with separate +/- discounting | `CFRConfig` |
| Compare on identical games and budgets | Exp 3 |
| NashConv and exploitability = NashConv / 2 | `best_response.cpp`, `subgame_context.cpp` |

## Adaptive computation

| Item | Where |
|---|---|
| Priority score S(I) from reach, uncertainty, regret and cost | `adaptive_sampler.*` |
| Use it for sampling with correct importance weights | importance-sampled MCCFR (unbiasedness tested) |
| Exp A: adaptive vs external sampling (variance, convergence, coverage, runtime) | `exp06_adaptive_sampling.py` |
| Exp B: start coarse, add sizes where interpolation changes regret, re-solve | `exp10_refinement.py` |
| Exp C: learned allocation trained on small games, tested on larger unseen ones | `exp11_learned_allocation.py` |

## Hold'em solver

| Item | Where |
|---|---|
| Ranges instead of single hands | `HandSpace`, `Range`, reach vectors |
| Bayesian update after an action | `Range::bayes_update`, reach propagation |
| Card removal | `Range::remove`, inclusion-exclusion in the kernels, masking at chance nodes |
| Separate ranges with pair compatibility | `SubgameContext` |
| JSON spot requests | `python/aegis/spot.py`, `python -m aegis solve` |
| Per-hand strategies, EVs, convergence, strategy profile | `SolvedSpot` |
| Exact exploitability, scoped to the constructed spot | `SubgameContext::exploitability` |

## Re-solving

| Item | Where |
|---|---|
| Unseen bet size: rebuild the local game with the real action | reference trees + `resolve_subtree` (Exp 7) |
| Blueprint counterfactual values as constraints | `opponent_constraint`, `GadgetConfig` |
| Naive vs constrained vs nested | `ResolveMethod`, nested river re-solves |
| A case where naive re-solving makes things worse | Coin Toss, Leduc with good blueprints, coarse-abstraction turn spot |

## Value network

| Item | Where |
|---|---|
| Only after the exact solver is validated | river solver tests come first; Exp 8 uses it for targets |
| Inputs: board, both ranges, pot, stacks (+ check-down values) | `cpp/value_net/mlp.h` |
| Per-hand counterfactual values as output | 2 x 1326 outputs |
| Targets from an expensive solver | `value_net/data_gen.cpp`, on-distribution data in Exp 8 |
| Weighted MSE + zero-sum regulariser | `python/training/train_value_net.py` |
| Measure safety rather than assume it | Exp 8 full-game exploitability |

## Systems

| Item | Where |
|---|---|
| C++20 core, Python/PyTorch experiments, TypeScript UI | `cpp/`, `python/`, `frontend/` |
| Contiguous regret / strategy arrays | `GameTree` slots, `VectorCFR` |
| Hand-indexed vectors, SIMD | vector kernels (auto-vectorised; no-SIMD build for comparison) |
| Cached hand evaluations per board | `SubgameContext`, feature-builder cache |
| Sparse hand lists, zero-reach pruning | `SubgameContext`, `TabularCFR` |
| Batched network inference | `NeuralLeafProvider` |
| Threaded sampling with synchronised updates | parallel MCCFR |
| Hardware counters | `HwCounters` |
| CPU vs SIMD vs threads vs GPU | Exp 9 |
| Input validation at the Python/C++ boundary | checked accessors in C++, validation in `spot.py`/`server.py`, `tests/python/test_review_regressions.py` |

## Experiments

| Experiment | Metric | File |
|---|---|---|
| Kuhn equilibrium | NashConv vs iterations | `exp01_kuhn_equilibrium.py` |
| Leduc comparison | exploitability vs time | `exp02_leduc_comparison.py` |
| CFR+ vs DCFR | convergence curves | `exp03_cfrplus_vs_dcfr.py` |
| Exact river solving | best-response exploitability | `exp04_river_exact.py` |
| Action abstraction | exploitability vs tree size | `exp05_action_abstraction.py` |
| Adaptive sampling | exploitability vs samples / seconds | `exp06_adaptive_sampling.py` |
| Safe vs naive re-solving | full-game exploitability | `exp07_safe_resolving.py` |
| Learned values | error vs solving performance | `exp08_value_network.py` |
| Performance | nodes/s, memory, latency | `exp09_performance.py` |
| Refinement | exploitability vs added sizes | `exp10_refinement.py` |
| Learned allocation | generalisation to larger spots | `exp11_learned_allocation.py` |
| Compute budgets | weighted exploitability vs budget | `exp12_budget_allocation.py` |
| Win rate vs exploitability | sanity check on evaluation | `exp13_winrate_vs_exploitability.py` |

Comparisons use matched budgets, the same game instances, several seeds where
there's sampling, and 95% confidence intervals.

## Web app

| View | Front end | API |
|---|---|---|
| River decision analysis | `frontend/src/solver_lab.ts`, `poker_table/`, `strategy_matrix/`, `regret_visualizer/` | `/api/solve`, `/api/spot/*` (tree nodes load lazily for big trees) |
| Range explorer | `frontend/src/range_explorer/` | `/api/range`, `/api/range/strategy` |
| Convergence lab | `frontend/src/convergence_lab/` | `/api/convergence` |
| Subgame microscope | `frontend/src/subgame_microscope/` | `/api/microscope`, `/api/microscope/whatif` |
| Adaptive computation | `frontend/src/adaptive_visualizer/` | `/api/adaptive` |
| Profiler | `frontend/src/solver_profiler/` | `/api/profiler` |

## Milestones

| Milestone | Done when | Status |
|---|---|---|
| 1 Exact foundation | CFR converges to Kuhn; best responses agree with OpenSpiel | done |
| 2 Solver engineering | reproducible CFR+/DCFR/MCCFR plots; profiling | done |
| 3 Poker maths | evaluator, pot accounting, showdowns, equity all cross-checked | done |
| 4 River engine | per-hand mixed strategies on arbitrary river spots, with exploitability | done |
| 5 Adaptive research | all variants benchmarked under matched budgets, with ablations and held-out spots | done (mixed results, reported as is) |
| 6 Continual re-solving | turn-to-river re-solving, value approximation, measured safety limits | done |
