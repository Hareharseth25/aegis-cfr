// Subgame microscope: information sets, action probabilities and best-response
// counterfactual action values of a solved tabular game; edit one information
// set's strategy and see the counterfactual consequences.

import { api } from "../shared/api";
import { actionBars } from "../shared/charts";
import { clear, fmt, h, slotColor, spinner } from "../shared/dom";

interface Info {
  id: number;
  key: string;
  player: number;
  actions: string[];
  strategy: number[];
  depth: number;
  num_histories: number;
  br_action_values: number[];
  br_best: number;
  public_state: string;
}
interface Resp { game: string; nodes: number; infosets: Info[]; nash_conv: number; values: number[] }
interface WhatIf {
  before: { nash_conv: number; value_p0: number; br_opp: number };
  after: { nash_conv: number; value_p0: number; br_opp: number };
  opponent_br_changes: { key: string; before: string; after: string }[];
}

export function mountMicroscope(root: HTMLElement): void {
  const game = h("select", {}, h("option", { value: "kuhn" }, "Kuhn poker"), h("option", { value: "leduc" }, "Leduc hold'em"), h("option", { value: "coin_toss" }, "Coin Toss")) as HTMLSelectElement;
  const filter = h("input", { type: "text", placeholder: "filter information sets (e.g. 'P4:B2' or 'pb')" }) as HTMLInputElement;
  const status = h("div", { class: "small muted" });
  const pubEl = h("div", { class: "card stack" });
  let pubFilter: string | null = null;
  const list = h("div", { class: "card scroll", style: "max-height:560px" });
  const detail = h("div", { class: "card stack" });
  root.append(h("h1", {}, "Subgame microscope"),
    h("p", { class: "lede" }, "Every information set of a game solved with CFR+: its action probabilities and, against the opponent's strategy, the counterfactual value "
      + "of each action (what CFR's regrets are made from). Edit a probability to see the exact exploitability and how the opponent's best response changes."),
    h("div", { class: "card row" }, h("label", { class: "field" }, "Game", game), h("label", { class: "field", style: "flex:1" }, "Filter", filter), status),
    h("p", { class: "small muted" }, "Key notation - Kuhn: own card (0 = J, 1 = Q, 2 = K) then history (p = pass, b = bet). Leduc: P<private card>:B<public card or ->:<round 1>/<round 2> "
      + "with f/c/r = fold/call/raise; card id c has rank c/2 (J, Q, K). Coin Toss: H/T/* then S/P."),
    pubEl,
    h("div", { class: "grid2", style: "margin-top:16px" }, list, detail));
  const renderPublic = () => {
    if (!data) return;
    clear(pubEl);
    // Public states (everything both players observe) in tree order: each
    // groups the information sets that differ only in private cards.
    const states = [...new Set(data.infosets.map((i) => i.public_state))];
    pubEl.append(h("h3", {}, `Public game tree - ${states.length} public decision states (click to focus)`));
    const chips = h("div", { class: "chips" });
    chips.append(h("span", { class: `chip ${pubFilter === null ? "active" : ""}`, onclick: () => { pubFilter = null; renderPublic(); renderList(); } }, "all"));
    for (const st of states.slice(0, 120)) {
      const n = data.infosets.filter((i) => i.public_state === st).length;
      chips.append(h("span", { class: `chip ${pubFilter === st ? "active" : ""}`, onclick: () => { pubFilter = st; renderPublic(); renderList(); } },
        `${st || "root"} (${n})`));
    }
    pubEl.append(chips);
  };
  let data: Resp | null = null;
  const renderList = () => {
    if (!data) return;
    clear(list);
    const f = filter.value.trim();
    const rows = data.infosets.filter((i) => (!f || i.key.includes(f)) && (pubFilter === null || i.public_state === pubFilter)).slice(0, 400);
    list.append(h("table", { class: "data" }, h("tr", {}, h("th", {}, "Information set"), h("th", {}, "P"), h("th", {}, "Strategy")),
      ...rows.map((i) => h("tr", { style: "cursor:pointer", onclick: () => select(i) }, h("td", {}, i.key), h("td", {}, String(i.player)),
        h("td", {}, i.actions.map((a, k) => `${a} ${(100 * i.strategy[k]).toFixed(0)}%`).join(" | "))))));
  };
  const select = (i: Info) => {
    clear(detail);
    const bars = h("div", {});
    actionBars(bars, i.actions.map((a, k) => ({ label: a, value: i.strategy[k], color: slotColor(k), sub: fmt(i.br_action_values[k], 4) })));
    const sliders = i.actions.map((_, k) => h("input", { type: "range", min: "0", max: "100", value: String(Math.round(100 * i.strategy[k])) }) as HTMLInputElement);
    const res = h("div", {});
    const apply = h("button", {}, "Apply change");
    apply.addEventListener("click", async () => {
      clear(res);
      res.append(spinner("Recomputing best responses..."));
      const strat = sliders.map((s) => Number(s.value) + 1e-9);
      try {
        const w = await api.post<WhatIf>("/api/microscope/whatif", { game: game.value, infoset: i.id, strategy: strat });
        clear(res);
        res.append(h("table", { class: "data" }, h("tr", {}, h("th", {}, ""), h("th", {}, "Before"), h("th", {}, "After")),
          h("tr", {}, h("td", {}, "NashConv"), h("td", {}, fmt(w.before.nash_conv, 5)), h("td", {}, fmt(w.after.nash_conv, 5))),
          h("tr", {}, h("td", {}, "Value (player 0)"), h("td", {}, fmt(w.before.value_p0, 5)), h("td", {}, fmt(w.after.value_p0, 5))),
          h("tr", {}, h("td", {}, `Opponent BR value`), h("td", {}, fmt(w.before.br_opp, 5)), h("td", {}, fmt(w.after.br_opp, 5)))),
        h("p", { class: "small muted" }, w.opponent_br_changes.length ? `Opponent best response changes at ${w.opponent_br_changes.length} information set(s):` : "The opponent's best response does not change."),
        h("div", { class: "scroll" }, h("table", { class: "data" }, ...w.opponent_br_changes.slice(0, 30).map((c) => h("tr", {}, h("td", {}, c.key), h("td", {}, `${c.before} -> ${c.after}`))))));
      } catch (e) {
        clear(res);
        res.append(h("p", { class: "error" }, String(e)));
      }
    });
    const cfTable = h("table", { class: "data" }, h("tr", {}, h("th", {}, "Action"), h("th", {}, "Probability"), h("th", {}, "Counterfactual value")),
      ...i.actions.map((a, k) => h("tr", {}, h("td", {}, a + (k === i.br_best ? " (best response)" : "")), h("td", {}, `${(100 * i.strategy[k]).toFixed(2)}%`), h("td", {}, fmt(i.br_action_values[k], 5)))));
    detail.append(h("h2", {}, `${i.key}`), h("div", { class: "small muted" }, `player ${i.player} | depth ${i.depth} | ${i.num_histories} histories in this information set`),
      h("h3", {}, "Average strategy"), bars, cfTable,
      h("p", { class: "small muted" }, "Counterfactual value = Σ over histories in the set of opponent-and-chance reach x value of the action (best response below). "
        + "At equilibrium every action played with positive probability has the same, maximal value."),
      h("h3", {}, "What if this information set played differently?"),
      ...i.actions.map((a, k) => h("label", { class: "field" }, `${a} weight`, sliders[k])), apply, res);
  };
  const load = async () => {
    clear(status);
    status.append(spinner("Solving with CFR+..."));
    try {
      data = await api.get<Resp>(`/api/microscope?game=${game.value}`);
      clear(status);
      status.append(`${data.nodes} histories | ${data.infosets.length} information sets | NashConv ${fmt(data.nash_conv, 6)}`);
      pubFilter = null;
      renderPublic();
      renderList();
      select(data.infosets[0]);
    } catch (e) {
      clear(status);
      status.append(h("span", { class: "error" }, String(e)));
    }
  };
  game.addEventListener("change", load);
  filter.addEventListener("input", renderList);
  load();
}
