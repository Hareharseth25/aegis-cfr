# AEGIS

A game-theoretic poker solver I wrote to learn (and test) how equilibrium
computation actually works for imperfect-information games. It's mostly C++20
with a Python layer for experiments and a small TypeScript front end.

The short version: you give it a heads-up Hold'em spot (board, pot, stacks,
bet sizes, both players' ranges) and it computes a mixed strategy for every
hand, the EV of each action, and how exploitable that strategy is. The
exploitability number is exact - it comes from a real best response, not an
estimate - but it's exact *for the spot you constructed*. I'm not claiming
anything about full No-Limit Hold'em.

Around that core there's a fair amount of research scaffolding: the toy games
used to check the maths, several CFR variants, safe subgame re-solving, a small
value network for depth-limited search, and a few experiments on spending
compute more cleverly (some of which didn't work - see below).

## What's in here

- **Toy games** - Kuhn, Leduc (same rules as OpenSpiel's version) and the Coin
  Toss game from the safe subgame solving paper. Small enough that exact answers
  are known, which is the whole point.
- **Solvers** - vanilla CFR, CFR+, DCFR and Linear CFR all share one set of
  update rules (`cpp/solvers/common/regret_rules.h`). There's also external,
  outcome and importance-sampled MCCFR, plus a parallel version.
- **Best response / NashConv** - restricted to the responder's information sets,
  so it can't peek at hidden cards.
- **Poker maths** - 7-card evaluator, 1,326-combo weighted ranges, card removal,
  blockers, Bayesian range updates, exact and Monte Carlo equity.
- **River/turn solver** - works on ranges as vectors over hands instead of
  enumerating hand pairs, with O(n) showdown evaluation. Turn trees parallelise
  over the river card.
- **Re-solving** - naive, Resolve and MaxMargin gadgets; re-solving after bet
  sizes the abstraction didn't include; pseudo-harmonic action translation.
- **Depth-limited search** - leaf values from a neural net, a blueprint, or an
  exact river solve.
- **Adaptive computation experiments** - priority-weighted sampling, bet-size
  refinement, a learned allocator and compute budgets across many subgames.
- **Front end** - a local web app for poking at solves (strategy grids, tree
  navigation, convergence plots, a range explorer and so on).

## How I checked it

I didn't want to trust any poker output until the small games were provably
right, so most of the tests are cross-checks against something independent:

- CFR, CFR+ and DCFR produce the same average strategies as OpenSpiel, iteration
  by iteration, on Kuhn and Leduc. NashConv on random policies matches to 1e-9.
- Kuhn comes out at -1/18 (closed form). Leduc's value, -0.0856064, also comes
  out of a separate sequence-form LP I wrote in Python.
- The hand evaluator is checked against a naive evaluator on all 2.6M five-card
  hands, and against the known category counts for all 133.8M seven-card hands.
- The vector river solver gives the same iterates as the generic tabular CFR
  running on the same spot written out as an explicit game tree. A separate
  PyTorch version of the river solver agrees to 1e-6.
- On a toy polarised river spot it reproduces the textbook answer (bluff 25% of
  the air, call 50%).
- There's also a set of regression tests for bad input (out-of-range ids,
  malformed policies, tiny compute budgets, path tricks on the web server).

Results from all the experiments are written up in
[docs/RESULTS.md](docs/RESULTS.md). That file is generated from the result
files, so the numbers match what was actually run.

Some of the more interesting ones:

- DCFR beats CFR+ on most poker-like games (by 2-5x on deep-stacked river
  spots), but not everywhere. CFR+ wins on Kuhn and on a single-bet-size river.
- Naive subgame re-solving can make a decent strategy much worse (a coarse
  blueprint went from 5% to 13% of the pot exploitable). The safe gadgets don't
  have that problem.
- Choosing which bet size to add by its "translation hole" beat every other
  rule I tried, on all 8 test spots.
- Priority-guided sampling didn't beat plain external sampling at equal compute,
  even though it reduced estimator variance. The learned allocator only tied the
  simple heuristic. I've left both results in as they are.

## Building

You'll need a C++20 compiler, CMake 3.20+, Python 3.10+ (NumPy, SciPy,
PyTorch, matplotlib, pytest) and Node 18+ for the front end. I developed it on
an M3 MacBook with Apple clang 15 and Python 3.13.

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install pybind11 open_spiel   # open_spiel is only used by the tests

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$PWD/.venv/bin/python
cmake --build build -j8
```

Tests:

```bash
./build/aegis_tests --exclude-slow        # ~8 s
./build/aegis_tests --only-slow           # exhaustive evaluator check etc.
.venv/bin/python -m pytest tests/python
```

## Using it

Solve the demo spot (A♠K♥8♣4♦2♠, 100 BB pot, 150 BB stacks, hero holds A♦J♦
and faces a 75 BB bet):

```bash
cd python
../.venv/bin/python -m aegis demo
../.venv/bin/python -m aegis solve my_spot.json --out result.json
../.venv/bin/python -m aegis equity --hero AdJd --villain "AK,KQ,88" --board AsKh8c
```

A spot file looks like this. Ranges can be range notation, a `{combo: weight}`
map, or a list of 1,326 weights. `hero.path` is optional; without it the hero
starts at their first decision.

```json
{
  "board": ["As", "Kh", "8c", "4d", "2s"],
  "pot": 100, "effective_stack": 150, "player_to_act": 0,
  "actions": {"bet_pot_fractions": [0.33, 0.75, 1.5], "raise_pot_fractions": [1.0],
              "allow_all_in": true, "max_raises": 3},
  "range_p0": "AA,KK,88,AK,AQ,A5s,KQs",
  "range_p1": {"AdQd": 1.0, "KhQh": 0.5, "QsQc": 1.0},
  "solver": {"algorithm": "dcfr", "iterations": 1000},
  "hero": {"player": 1, "hand": "AdJd", "path": ["Bet 75"]}
}
```

EVs are reported from the decision being looked at: chips won from the pot
minus chips still to be put in. Whatever was already committed is treated as
sunk, so folding is always worth 0.

There's also a plain C++ command-line tool if you don't want Python:

```bash
./build/aegis_cli tabular --game leduc --algo cfr+ --iters 1000
./build/aegis_cli river --board AsKh8c4d2s --r0 "AA,KK,AK,AQ" --r1 "AQ,KQ,QQ-99" --iters 1000
```

Web app:

```bash
(cd frontend && npm install && npm run build)
cd python && ../.venv/bin/python -m aegis.server --port 8765
# then open http://localhost:8765
```

To rerun every experiment and regenerate `docs/RESULTS.md` (about 30 minutes on
8 cores):

```bash
.venv/bin/python python/experiments/run_all.py
```

## Layout

```
cpp/
  core/          game interface, flat game trees, policies, best response
  games/         Kuhn, Leduc, Coin Toss, Hold'em subgame as an explicit tree
  poker/         cards, evaluator, ranges, equity, No-Limit betting trees
  solvers/       CFR family, MCCFR, vector solver, re-solving, translation
  adaptive/      sampling priorities, compute budgets, bet-size refinement
  value_net/     MLP inference and training-data generation
  bindings/      pybind11 module
python/
  aegis/         Python API, CLI, HTTP server, OpenSpiel bridge
  experiments/   the 13 experiments + report generator
  models/, training/, evaluation/
frontend/        the web app (Vite + TypeScript, no framework)
benchmarks/      aegis_bench
tests/           C++ and Python tests
docs/            architecture notes, results, checklist, paper references
```

## Notes and limitations

- Hold'em exploitability is only meaningful within the spot you built (ranges +
  bet sizes). Kuhn, Leduc and Coin Toss are solved as complete games.
- The value network is small and trained on ~20k positions from one turn board
  family. It works, but it's clearly worse than solving the river exactly.
- Hardware performance counters on macOS need root (`sudo ./build/aegis_bench`).
  Without them the benchmarks just record timings.
- The web server is meant for local use. It validates its inputs, but it has no
  auth or rate limiting, so don't expose it to a network.
- The papers this is based on are listed in [docs/papers](docs/papers/README.md).
  More detail on the maths is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
