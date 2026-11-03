# Papers

The three papers I leaned on most. The PDFs aren't committed (arXiv's
licence doesn't cover redistribution), so grab them from the links:

- N. Brown, A. Lerer, S. Gross, T. Sandholm. *Deep Counterfactual Regret
  Minimization.* ICML 2019. <https://arxiv.org/abs/1811.00164>
  I went a different way here: the regret learner stays tabular/vectorised so
  it's easy to inspect, and a network is only used for continuation values at
  depth limits (DeepStack-style). See `cpp/value_net/`,
  `python/models/value_net.py` and Experiment 8.
- N. Brown, T. Sandholm. *Safe and Nested Subgame Solving for
  Imperfect-Information Games.* NeurIPS 2017. <https://arxiv.org/abs/1705.02955>
  Coin Toss (`cpp/games/coin_toss.*`), the Resolve and MaxMargin gadgets
  (`tabular_resolving.*`, `vector_cfr.*`, `resolver.*`) and nested re-solving
  after off-tree bets. Experiment 7.
- N. Brown, T. Sandholm. *Solving Imperfect-Information Games via Discounted
  Regret Minimization.* AAAI 2019. <https://arxiv.org/abs/1809.04040>
  DCFR and Linear CFR in `cpp/solvers/common/regret_rules.h`. Experiment 3
  compares DCFR with CFR+ on the same games.

If you keep local copies in this folder, git ignores them
(`docs/papers/*.pdf` is in `.gitignore`).

Other references used in the code:

- Zinkevich, Johanson, Bowling, Piccione. *Regret Minimization in Games with
  Incomplete Information.* NeurIPS 2007 (vanilla CFR).
- Lanctot, Waugh, Zinkevich, Bowling. *Monte Carlo Sampling for Regret
  Minimization in Extensive Games.* NeurIPS 2009 (external and outcome
  sampling).
- Tammelin. *Solving Large Imperfect Information Games Using CFR+.*
  arXiv:1407.5042, 2014; Tammelin, Burch, Johanson, Bowling, IJCAI 2015.
- Burch, Johanson, Bowling. *Solving Imperfect Information Games Using
  Decomposition.* AAAI 2014 (CFR-D and the re-solving gadget).
- Moravčík, Schmid, Ha, Hladík, Gaukrodger. *Refining Subgames in Large
  Imperfect Information Games.* AAAI 2016 (MaxMargin).
- Moravčík et al. *DeepStack: Expert-Level Artificial Intelligence in Heads-Up
  No-Limit Poker.* Science 2017 (continual re-solving, value networks).
- Brown, Sandholm. *Superhuman AI for heads-up no-limit poker: Libratus beats
  top professionals.* Science 2018.
- Brown, Sandholm, Amos. *Depth-Limited Solving for Imperfect-Information
  Games.* NeurIPS 2018 (blueprint values at depth limits).
- Ganzfried, Sandholm. *Action Translation in Extensive-Form Games with Large
  Action Spaces.* IJCAI 2013 (pseudo-harmonic mapping).
- Koller, Megiddo, von Stengel. *Fast Algorithms for Finding Randomized
  Strategies in Game Trees.* STOC 1994 (sequence-form LP).
- Kuhn. *A Simplified Two-Person Poker.* 1950; Southey et al. *Bayes' Bluff.*
  UAI 2005 (Leduc hold'em).
- Lanctot et al. *OpenSpiel: A Framework for Reinforcement Learning in Games.*
  arXiv:1908.09453. Only used in the tests, as an independent reference.
