# Architecture notes

How the maths maps onto the code, roughly in the order things are built on top
of each other. File paths are relative to the repo root.

## 1. Games and game trees

`cpp/core/extensive_game/game.h` is the usual extensive-form interface: a
`State` is a history, `current_player()` is P(h) (a player or chance),
`chance_outcomes()` gives the chance probabilities, `returns()` gives terminal
utilities, and `information_state_key(p)` names player p's information set at
that history. The key is defined for both players at every non-terminal
history (not just the one acting), because subgame re-solving needs it.

`cpp/core/game_tree/` turns a game into flat arrays (depth-first, parents
before children). While building, it checks that:

- all histories in an information set have the same legal actions,
- the game has perfect recall (every history in an information set has the
  same own-action sequence for the acting player),
- chance probabilities are non-negative and sum to 1,
- payoffs are zero-sum (CFR refuses to run otherwise).

Regrets, strategy sums and policies live in one flat array indexed by
"action slot" (`infoset.offset + action`). No object per node. The
`NaiveCFR` class does it the textbook way with a hash map of node objects; it's
there as a second implementation to compare against, and as the slow baseline
in the benchmarks (the flat version is about 27x faster on Leduc).

Games: Kuhn, Leduc (rules match OpenSpiel so the trees line up node for node),
Coin Toss, and `holdem_subgame_efg`, which writes a Hold'em turn/river spot out
as an explicit game tree (chance deals hand pairs in proportion to
r0(h0) * r1(h1) when the hands don't overlap).

## 2. CFR and friends

All the regret update rules live in `cpp/solvers/common/regret_rules.h`, and
every solver in the project uses them, so "CFR+" means the same thing on Kuhn
as it does on a river spot:

```
r_t(I,a) = v(I,a) - v(I)                      instantaneous counterfactual regret
vanilla:  R_t = R_{t-1} + r_t
CFR+:     R_t = max(0, R_{t-1} + r_t)
DCFR:     R_t = (R_{t-1} + r_t) * d_t,  d_t = t^a/(t^a+1) if R >= 0, t^b/(t^b+1) otherwise
LCFR:     DCFR with (a, b, g) = (1, 1, 1)
next strategy: regret matching on R_t (uniform if nothing is positive)
average:  S_t(I,a) += w_t * pi_i(I) * sigma_t(I,a)
          w_t = 1 (CFR), max(t - d, 0) (CFR+), t^g (DCFR / LCFR)
```

`TabularCFR` (`cpp/solvers/vanilla_cfr/`) weights counterfactual values by the
opponent's and chance's reach; the player's own reach only goes into the
average strategy. Instantaneous regrets are summed over every history in an
information set first and the update rule is applied once afterwards. That
matters for CFR+: clipping per history would be wrong.

One numerical detail: two actions with mathematically equal values can come
out ~1e-16 apart because they're computed along different paths, and regret
matching will happily turn that into a pure strategy. `tie_tolerance` (1e-12
relative) treats differences that small as ties. It's switched off in the
OpenSpiel comparison tests so the arithmetic matches exactly.

What checks this:

- the closed-form Kuhn equilibrium family is reproduced exactly,
- CFR, CFR+ and DCFR match OpenSpiel's average policies iteration by
  iteration on Kuhn and Leduc (`tests/python/test_openspiel_oracle.py`),
- `python/evaluation/sequence_form.py` solves Kuhn and Leduc as a
  sequence-form LP and gets the same game values (Leduc: -0.0856064).

## 3. Best response and NashConv

`cpp/core/best_response/` computes a best response over the responder's own
information sets: at each one it picks the action with the highest
(opponent + chance reach) weighted value, with values below computed under the
best response. It never looks at hidden cards.

```
NashConv(sigma) = sum_i [ u_i(BR_i(sigma_-i), sigma_-i) - u_i(sigma) ]
exploitability  = NashConv / 2
```

Tests check that the best response is a legal strategy (evaluating it the
normal way gives the same value) and that random strategies never beat it.
`counterfactual_regrets()` gives exact regrets at a fixed strategy, which is
what the sampling tests compare against.

## 4. Monte Carlo CFR

`cpp/solvers/mccfr/`:

- **External sampling**: sample chance and opponent actions, explore all of the
  traverser's. The sampling probability cancels against the opponent reach, so
  the regret estimate is just `v(h,a) - v(h)`.
- **Outcome sampling**: one trajectory, epsilon-uniform exploration at the
  traverser's nodes, importance weights on the way back up.
- **Importance-sampled external sampling**: same as external sampling, but
  chance/opponent actions can be drawn from any distribution q supplied by a
  `SamplingPolicy`. Each sampled value is multiplied by p(a)/q(a) and regrets
  are scaled by W(h) = pi_-i(h) / q_-i(h). As long as q > 0 wherever p > 0 the
  estimate is unbiased. With q = p it's bit-for-bit identical to external
  sampling, which is tested.
- **Parallel external sampling**: each batch, threads run traversals against a
  frozen strategy into private buffers, then the buffers are summed in a fixed
  order. Deterministic for a given seed.

## 5. Poker maths

- `poker/cards`: card id = 4 * rank + suit; combos indexed b(b-1)/2 + a.
- `poker/hand_evaluator`: bit-parallel 7-card evaluator. Checked against a
  naive 5-card evaluator on every 5-card hand (all 7,462 distinct classes come
  out right) and against the known category counts for all 133.8M 7-card
  hands, plus a pure-Python cross-check.
- `poker/ranges`: 1,326-weight vectors, range notation parser, card removal,
  Bayesian updates (r(h|a) proportional to r(h) * sigma(a|h)), blockers, and a
  random range generator used for training data.
- `poker/equity`: exact enumeration (hand vs hand, hand vs range, range vs
  range) and Monte Carlo with standard errors.

## 6. The vector solver

`poker/betting_rules/betting.cpp` builds the public betting tree: bets as
fractions of the pot, raises as "call then raise f times the new pot",
min-bet/min-raise, all-in merging, raise caps, river chance nodes for turn
trees, optional depth-limit leaves, and starting mid-street. Payoffs are
zero-sum relative to the start of the subgame (a fold by player f pays out
f's commitment; a showdown is +/- the shared commitment).

`solvers/subgame_resolver/subgame_context.cpp` keeps a hand list per player
(only combos actually in the range), cached hand strengths and sort orders per
board, and the terminal kernels:

```
fold:      cfv(h) = u * (R - R[c1] - R[c2] + r(h))          inclusion-exclusion on the two cards
showdown:  cfv(h) = C * (mass weaker than h - mass stronger than h)    two sorted sweeps
chance:    cfv(h) = (1/44) * sum over river cards c not in h of cfv_c(h)
```

It can also evaluate any strategy profile, compute best responses and
NashConv, and get counterfactual values at any node for given reaches.

`vector_cfr.cpp` runs a forward pass (reach), batched leaf queries, then a
backward pass (counterfactual values and regret updates) per player. Turn
trees split the work over river cards on a thread pool. It can also solve just
a subtree from given root reaches, with or without a re-solving gadget.

Checks: same iterates as tabular CFR on the explicit-tree version of the same
river and turn spots (all four update rules), same best-response values, the
analytic polarised river equilibrium, and an independent PyTorch version
(`python/evaluation/torch_river.py`) that agrees to 1e-6.

## 7. Re-solving

- **Naive**: enter the subgame with both players' blueprint reach and solve it
  on its own.
- **Resolve** (Burch et al.): each opponent hand can either "terminate" and
  take a fixed value w(h), or "follow" into the subgame. The default w(h) is
  the opponent's counterfactual best-response value against our blueprint.
- **MaxMargin**: the opponent picks a distribution over its hands that
  maximises v(h) - w(h).
- **After an unexpected bet size**: re-solve below the new action with
  w(h) = the opponent's best value among the actions the blueprint did have.

`tabular_resolving.cpp` does this for any game tree; `resolver.cpp` and the
gadget support in `vector_cfr.cpp` do it for poker. Experiment 7 measures the
result against a full-game best response.

## 8. Action translation

`action_translation.cpp` plays a strategy from a coarse bet abstraction in a
finer reference game. An unfamiliar size x between abstract sizes A and B maps
to A with probability ((B - x)(1 + A)) / ((B - A)(1 + x)) (pseudo-harmonic) or
to the nearest size. Because the mapping is random, later decisions mix the
abstract strategies weighted by each player's own reach in each branch.

## 9. Depth-limited search and the value net

Depth-limited turn trees stop at the end of the turn. A `LeafValueProvider`
fills in both players' counterfactual values at each leaf (averaging over river
cards), once per pass and in one batch: from a neural network (C++ MLP using
Accelerate's SGEMM), from a blueprint strategy, from an exact river solve, or
from a Python callback.

The network (`python/models/value_net.py`, same layout as
`cpp/value_net/mlp.h`) takes the board, both normalised ranges, each hand's
check-down value against the other range, pot share and log(1 + SPR), and
predicts per-hand counterfactual values for both players. Training loss is
masked MSE plus a penalty on the zero-sum residual. Training data comes from
DCFR-solved river spots (`value_net/data_gen.cpp`).

## 10. Adaptive computation

- `adaptive/allocation/adaptive_sampler.*`: the priority score
  S(I) = (e+rho)^a (e+U)^b (e+G)^g / (e+C)^d (reach, uncertainty, regret, cost)
  turned into a sampling distribution q = (1 - lambda) p + lambda psi.
- `adaptive/allocation/budget_allocator.*`: splits a fixed work budget across
  many subgames using fitted convergence curves. Every iteration and every
  exploitability measurement is checked against what's left of the budget.
- `adaptive/refinement/action_refinement.*`: "translation holes", i.e. how
  much a player could gain by using a bet size the abstraction doesn't have.
- `python/experiments/exp11_learned_allocation.py`: a small learned model for
  choosing refinements.

## 11. Systems bits

- static C++ library, pybind11 module (long calls release the GIL),
- a small test harness with no dependencies,
- CMake options for native tuning and a no-SIMD build for comparison,
- a thread pool that keeps results deterministic,
- hardware counters through perf_event on Linux or kperf on macOS (root only).

Python handles experiments, training and the local HTTP server. The front end
is Vite + TypeScript with no framework.
