// Solver Lab - river decision analysis: solve a spot, walk its game tree and
// inspect the hero's mixed strategy with exact diagnostics.

import { renderTable } from "./poker_table/table";
import { renderConvergence } from "./regret_visualizer/regret";
import { api, HeroView, NodeStrategy, SolveResult, TreeNode } from "./shared/api";
import { actionBars, barChart } from "./shared/charts";
import { clear, fmt, h, slotColor, spinner, splitCards } from "./shared/dom";
import { renderMatrix } from "./strategy_matrix/matrix";

const DEFAULTS = {
  board: "As Kh 8c 4d 2s",
  pot: 100,
  stack: 150,
  bets: "0.33, 0.75, 1.5",
  raises: "1.0",
  p0: "AA,KK,88,44,22,AK,AQ,AJ,AT,A9s-A2s,KQ,KJs,QJs,JTs,T9s,98s,87s,76s,65s,54s,K8s,Q8s:0.5,J8s:0.5",
  p1: "AQ,AJ,AT,A9-A2,KQ,KJ,KT,K9s,QQ-99,77-55,33,QJs,JTs,T9s,98s,87s",
  hero: "AdJd",
  heroPlayer: 1,
  iterations: 1000,
  algorithm: "dcfr",
};

interface State {
  result?: SolveResult;
  nodes: Map<number, TreeNode>;  // lazily loaded public-tree nodes
  generation: number;            // bumps on every render; stale responses are dropped
  node: number;
  heroPlayer: number;
  hero: string;
  selectedClass?: string;
  tab: string;
}

export function mountSolverLab(root: HTMLElement): void {
  const st: State = { nodes: new Map(), generation: 0, node: 0, heroPlayer: DEFAULTS.heroPlayer, hero: DEFAULTS.hero, tab: "frequency" };
  const field = (label: string, input: HTMLElement) => h("label", { class: "field" }, label, input);
  const inp = (v: string | number, type = "text") => h("input", { type, value: String(v) }) as HTMLInputElement;
  const board = inp(DEFAULTS.board), pot = inp(DEFAULTS.pot, "number"), stack = inp(DEFAULTS.stack, "number");
  const bets = inp(DEFAULTS.bets), raises = inp(DEFAULTS.raises), iters = inp(DEFAULTS.iterations, "number");
  const hero = inp(DEFAULTS.hero);
  const heroP = h("select", {}, h("option", { value: "0" }, "Player 0 (OOP)"), h("option", { value: "1", selected: true }, "Player 1 (IP)")) as HTMLSelectElement;
  const algo = h("select", {}, ...["dcfr", "cfr+", "lcfr", "cfr"].map((a) => h("option", { value: a, selected: a === DEFAULTS.algorithm }, a.toUpperCase()))) as HTMLSelectElement;
  const r0 = h("textarea", {}, DEFAULTS.p0) as HTMLTextAreaElement;
  const r1 = h("textarea", {}, DEFAULTS.p1) as HTMLTextAreaElement;
  const solveBtn = h("button", {}, "Solve spot");
  const status = h("div", { class: "small muted" });
  const tableEl = h("div", { class: "card" });
  const pathEl = h("div", { class: "treepath" });
  const heroEl = h("div", { class: "card" });
  const summaryEl = h("div", { class: "card" });
  const tabsEl = h("div", { class: "chips" });
  const diagEl = h("div", { class: "card" });

  root.append(
    h("h1", {}, "River decision analysis"),
    h("p", { class: "lede" }, "A heads-up No-Limit river spot, solved with vector-form CFR over both players' ranges (all 1,326 combos). "
      + "You get a mixed strategy for every hand. Exploitability is an exact best response in this spot "
      + "(these ranges and bet sizes), not in full Hold'em."),
    h("div", { class: "card stack" },
      h("div", { class: "grid3" }, field("Board (5 cards, or 4 for a turn spot)", board), field("Pot (BB)", pot), field("Effective stack (BB)", stack)),
      h("div", { class: "grid3" }, field("Bet sizes (pot fractions)", bets), field("Raise sizes (pot fractions)", raises), field("Iterations", iters)),
      h("div", { class: "grid2" }, field("Range player 0 (acts first)", r0), field("Range player 1", r1)),
      h("div", { class: "grid3" }, field("Hero hand", hero), field("Hero seat", heroP), field("Algorithm", algo)),
      h("div", { class: "row" }, solveBtn, status)),
  );
  const results = h("div", { class: "stack", style: "margin-top:16px" });
  root.append(results);

  solveBtn.addEventListener("click", async () => {
    solveBtn.setAttribute("disabled", "");
    clear(status);
    status.append(spinner("Solving..."));
    st.hero = hero.value.trim();
    st.heroPlayer = Number(heroP.value);
    try {
      const res = await api.post<SolveResult>("/api/solve", {
        board: splitCards(board.value), pot: Number(pot.value), effective_stack: Number(stack.value),
        actions: { bet_pot_fractions: bets.value.split(",").map(Number).filter((x) => x > 0),
          raise_pot_fractions: raises.value.split(",").map(Number).filter((x) => x > 0), allow_all_in: true, max_raises: 3 },
        range_p0: r0.value, range_p1: r1.value,
        solver: { algorithm: algo.value, iterations: Number(iters.value) },
        // No path: the server starts the hero at its first decision in this tree.
        hero: { player: st.heroPlayer, hand: st.hero },
      });
      st.result = res;
      st.nodes = new Map(res.tree.map((n) => [n.id, n]));
      st.node = res.hero?.node ?? 0;
      clear(status);
      status.append(`Solved in ${fmt(res.total_seconds, 2)} s (CFR ${fmt(res.solve_seconds, 2)} s) | ${res.num_nodes} public nodes | `
        + `${res.hands[0]}x${res.hands[1]} hands | exploitability ${fmt(res.exploitability.exploitability_pct_pot, 4)}% of pot`);
      if (res.hero_error) status.append(h("div", { class: "error" }, res.hero_error));
      go(st.node);
    } catch (e) {
      clear(status);
      status.append(h("span", { class: "error" }, String(e)));
    } finally {
      solveBtn.removeAttribute("disabled");
    }
  });

  // The demo spot solves in well under a second: show a computed result immediately.
  queueMicrotask(() => solveBtn.click());

  function facing(nd: TreeNode): number {
    if (!nd.street_bet || nd.type !== "DECISION") return 0;
    return Math.max(0, nd.street_bet[1 - nd.player] - nd.street_bet[nd.player]);
  }

  // Fetch any of `ids` not yet loaded (large trees are sent as a prefix).
  async function ensureNodes(ids: number[]): Promise<void> {
    const res = st.result;
    if (!res) return;
    const missing = [...new Set(ids)].filter((i) => !st.nodes.has(i));
    for (let k = 0; k < missing.length; k += 400) {
      const chunk = missing.slice(k, k + 400);
      const got = await api.get<{ nodes: TreeNode[] }>(`/api/spot/${res.id}/tree?ids=${chunk.join(",")}`);
      for (const n of got.nodes) st.nodes.set(n.id, n);
    }
  }

  async function ancestors(n: number): Promise<number[]> {
    const chain: number[] = [];
    let cur = n;
    while (cur >= 0) {
      await ensureNodes([cur]);
      chain.unshift(cur);
      cur = (st.nodes.get(cur) as TreeNode).parent;
    }
    return chain;
  }

  function go(n: number): void {
    st.node = n;
    renderAll().catch((e) => {
      clear(heroEl);
      heroEl.append(h("p", { class: "error" }, `Could not load node: ${String(e)}`));
    });
  }

  async function renderAll(): Promise<void> {
    const res = st.result;
    if (!res) return;
    const gen = ++st.generation;
    const chain = await ancestors(st.node);
    const nd = st.nodes.get(st.node) as TreeNode;
    await ensureNodes(nd.children);
    const node = await api.get<NodeStrategy>(`/api/spot/${res.id}/node/${st.node}`);
    if (gen !== st.generation) return;  // the user moved on while we were waiting
    clear(results);
    results.append(tableEl, h("div", { class: "card stack" }, h("h3", {}, "Game tree"), pathEl), h("div", { class: "grid2" }, heroEl, summaryEl),
      h("div", { class: "card stack" }, h("h3", {}, "Diagnostics"), tabsEl, diagEl));
    const stackLeft = res.effective_stack - (Math.max(...nd.commit) - res.pot / 2);
    renderTable(tableEl, { board: nd.board, hero: splitCards(st.hero), pot: nd.pot, facing: facing(nd), stack: stackLeft,
      toAct: nd.type === "DECISION" ? `Player ${nd.player}${nd.player === st.heroPlayer ? " (hero)" : ""}` : nd.type.toLowerCase() });
    renderPath(chain, nd);
    renderSummary(res);
    await renderHero(res, node, gen);
    if (gen !== st.generation) return;
    renderTabs(res, node);
  }

  function renderPath(chain: number[], nd: TreeNode): void {
    clear(pathEl);
    chain.forEach((n, i) => {
      if (i) pathEl.append(h("span", { class: "muted" }, ">"));
      pathEl.append(h("span", { class: "crumb", onclick: () => go(n) }, n === 0 ? "root" : (st.nodes.get(n) as TreeNode).action));
    });
    if (nd.children.length) {
      pathEl.append(h("span", { class: "muted", style: "margin-left:10px" }, "next:"));
      for (const c of nd.children) {
        const child = st.nodes.get(c);
        pathEl.append(h("span", { class: "chip", onclick: () => go(c) }, child ? child.action : `node ${c}`));
      }
    }
  }

  function renderSummary(res: SolveResult): void {
    clear(summaryEl);
    const e = res.exploitability;
    summaryEl.append(
      h("h3", {}, "Equilibrium quality"),
      h("table", { class: "data" },
        h("tr", {}, h("th", {}, "NashConv"), h("td", {}, `${fmt(e.nash_conv, 5)} BB`)),
        h("tr", {}, h("th", {}, "Exploitability"), h("td", {}, `${fmt(e.exploitability_pct_pot, 5)}% of pot`)),
        h("tr", {}, h("th", {}, "Value P0 / P1"), h("td", {}, `${fmt(e.policy_value[0], 3)} / ${fmt(e.policy_value[1], 3)} BB`)),
        h("tr", {}, h("th", {}, "BR value P0 / P1"), h("td", {}, `${fmt(e.br_value[0], 3)} / ${fmt(e.br_value[1], 3)} BB`)),
        h("tr", {}, h("th", {}, "Time"), h("td", {}, `${fmt(res.total_seconds, 3)} s total, ${fmt(res.solve_seconds, 3)} s in CFR, ${fmt(res.memory_mb, 1)} MB`))),
      h("p", { class: "small muted" }, res.note));
  }

  async function renderHero(res: SolveResult, node: NodeStrategy, gen: number): Promise<void> {
    clear(heroEl);
    heroEl.append(h("h3", {}, `Hero ${st.hero} at ${node.path || "root"}`));
    if (node.type !== "DECISION") { heroEl.append(h("p", { class: "muted" }, "Terminal node.")); return; }
    if (node.player !== st.heroPlayer) {
      heroEl.append(h("p", { class: "muted small" }, `Player ${node.player} (villain) acts here - aggregate range frequencies:`));
      const bars = h("div", {});
      actionBars(bars, node.actions.map((a, k) => ({ label: a, value: node.aggregate_frequency[k], color: slotColor(k) })));
      heroEl.append(bars);
      return;
    }
    try {
      const path = node.path === "root" ? "" : node.path.split(" / ").join("|");
      const hv = await api.get<HeroView>(`/api/spot/${res.id}/hero?player=${st.heroPlayer}&hand=${encodeURIComponent(st.hero)}&path=${encodeURIComponent(path)}`);
      if (gen !== st.generation) return;
      const bars = h("div", {});
      actionBars(bars, (hv.actions ?? []).map((a, k) => ({
        label: a, value: (hv.strategy ?? [])[k], color: slotColor(k),
        sub: hv.action_ev_from_here?.[k] !== null && hv.action_ev_from_here?.[k] !== undefined ? `${fmt(hv.action_ev_from_here[k] as number, 2)} BB` : undefined,
      })));
      heroEl.append(h("div", { class: "small muted" }, "Computed mixed strategy (average strategy of the solve):"), bars,
        h("table", { class: "data", style: "margin-top:10px" },
          h("tr", {}, h("th", {}, "Action"), h("th", {}, "Frequency"), h("th", {}, "EV from here")),
          ...(hv.actions ?? []).map((a, k) => h("tr", {}, h("td", {}, a), h("td", {}, `${(100 * (hv.strategy ?? [])[k]).toFixed(2)}%`),
            h("td", {}, `${fmt(hv.action_ev_from_here?.[k] ?? null, 2)} BB`)))),
        h("p", { class: "small muted" }, `Hand EV ${fmt(hv.ev_from_here ?? null, 2)} BB from here: chips won from the pot minus chips still to put in. `
          + `The ${fmt(hv.committed, 0)} BB already committed is sunk, so folding is 0. `
          + `Hero's cards remove ${(100 * hv.opponent_range_blocked_fraction).toFixed(1)}% of the opponent's range here (blockers).`));
    } catch (e) {
      heroEl.append(h("p", { class: "error small" }, String(e)));
    }
  }

  function renderTabs(res: SolveResult, node: NodeStrategy): void {
    clear(tabsEl);
    const tabs: [string, string][] = [["frequency", "Action frequency"], ["ev", "Range-by-range EV"], ["regret", "Regret convergence"], ["audit", "Best-response audit"]];
    for (const [k, label] of tabs) tabsEl.append(h("span", { class: `chip ${st.tab === k ? "active" : ""}`, onclick: () => { st.tab = k; renderTabs(res, node); } }, label));
    clear(diagEl);
    if (st.tab === "frequency") {
      if (node.type !== "DECISION") { diagEl.append(h("p", { class: "muted" }, "Terminal node.")); return; }
      const bars = h("div", {});
      actionBars(bars, node.actions.map((a, k) => ({ label: a, value: node.aggregate_frequency[k], color: slotColor(k) })));
      const mat = h("div", {});
      renderMatrix(mat, node.class_grid, node.actions, (cls) => { st.selectedClass = cls; renderTabs(res, node); }, st.selectedClass);
      const detail = h("div", { class: "scroll" });
      if (st.selectedClass) {
        const rows = node.combos.map((c, i) => ({ c, i })).filter(({ c }) => classOf(c) === st.selectedClass && node.reach[node.combos.indexOf(c)] > 0);
        detail.append(h("table", { class: "data" }, h("tr", {}, h("th", {}, "Combo"), h("th", {}, "Reach"), ...node.actions.map((a) => h("th", {}, a))),
          ...rows.map(({ c, i }) => h("tr", {}, h("td", {}, c), h("td", {}, fmt(node.reach[i], 3)), ...node.strategy[i].map((p) => h("td", {}, `${(100 * p).toFixed(1)}%`))))));
      }
      diagEl.append(h("div", { class: "grid2" }, h("div", {}, h("h3", {}, `Player ${node.player} range frequencies`), bars,
        h("p", { class: "small muted" }, "Click a hand class to list its combinations."), detail), mat));
    } else if (st.tab === "ev") {
      const evs = node.action_ev_from_here;
      if (!evs) { diagEl.append(h("p", { class: "muted" }, "Select a decision node.")); return; }
      const idx = node.combos.map((_, i) => i).filter((i) => node.reach[i] > 0)
        .sort((a, b) => Math.max(...evs[b].map((x) => x ?? -1e9)) - Math.max(...evs[a].map((x) => x ?? -1e9)));
      diagEl.append(h("p", { class: "small muted" }, "EV of each action for every hand the acting player can have here, in BB measured from this decision "
        + "(chips won minus chips still to put in; folding = 0)."),
        h("div", { class: "scroll" }, h("table", { class: "data" },
          h("tr", {}, h("th", {}, "Combo"), ...node.actions.map((a) => h("th", {}, a)), h("th", {}, "Strategy")),
          ...idx.map((i) => h("tr", {}, h("td", {}, node.combos[i]), ...evs[i].map((v) => h("td", {}, fmt(v, 2))),
            h("td", {}, node.strategy[i].map((p, k) => `${node.actions[k].split(" ")[0]} ${(100 * p).toFixed(0)}%`).join(" | ")))))));
    } else if (st.tab === "regret") {
      const reg = res.hero?.regret_trace?.map((r) => ({ iteration: r.iteration, value: r.avg_positive_regret })) ?? [];
      renderConvergence(diagEl, res.convergence, reg);
    } else {
      api.get<{ exploiter: number; br_value: number; policy_value: number; gain_total: number; top_hands: { combo: string; gain: number }[] }>(
        `/api/spot/${res.id}/audit?player=${st.heroPlayer}`).then((a) => {
        diagEl.append(h("p", { class: "small muted" }, `Best response of player ${a.exploiter} against player ${st.heroPlayer}'s strategy: `
          + `value ${fmt(a.br_value, 4)} vs ${fmt(a.policy_value, 4)} BB under the solved profile - total gain ${fmt(a.gain_total, 5)} BB. `
          + "The best response is restricted to its own information sets (it never sees hidden cards). Hands where it gains most:"));
        const c = h("div", {});
        barChart(c, a.top_hands.map((x) => ({ label: x.combo, value: Math.max(x.gain, 0), color: "var(--s1)" })), { yLabel: "BB gained" });
        diagEl.append(c);
      }).catch((e) => diagEl.append(h("p", { class: "error" }, String(e))));
    }
  }
}

const RANKS = "23456789TJQKA";
function classOf(combo: string): string {
  const a = combo[0], b = combo[2];
  const hi = RANKS.indexOf(a) >= RANKS.indexOf(b) ? a : b, lo = hi === a ? b : a;
  if (a === b) return a + b;
  return hi + lo + (combo[1] === combo[3] ? "s" : "o");
}
