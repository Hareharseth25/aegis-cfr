// Adaptive computation visualiser: where does the priority-guided sampler
// spend its traversals compared with uniform external sampling, and did the
// extra work reduce exploitability?

import { api } from "../shared/api";
import { hideTooltip, lineChart, showTooltip } from "../shared/charts";
import { clear, fmt, h, spinner } from "../shared/dom";

interface InfoStat { key: string; player: number; depth: number; visits: number; score: number; reach: number; uncertainty: number; sensitivity: number; cost: number }
interface Method { curve: { iteration: number; nodes: number; exploitability: number }[]; infosets: InfoStat[] }
interface Resp { game: string; methods: Record<string, Method> }

export function mountAdaptive(root: HTMLElement): void {
  const game = h("select", {}, h("option", { value: "leduc" }, "Leduc hold'em"), h("option", { value: "kuhn" }, "Kuhn poker")) as HTMLSelectElement;
  const iters = h("input", { type: "number", value: "30000" }) as HTMLInputElement;
  const lam = h("input", { type: "number", value: "0.3", step: "0.1", min: "0", max: "0.9" }) as HTMLInputElement;
  const go = h("button", {}, "Run both samplers");
  const status = h("div", { class: "small muted" });
  const curve = h("div", { class: "card" });
  const heat = h("div", { class: "card" });
  const top = h("div", { class: "card" });
  root.append(h("h1", {}, "Adaptive computation visualiser"),
    h("p", { class: "lede" }, "Priority S(I) = (ε+ρ)^α (ε+U)^β (ε+G)^γ / (ε+C)^δ guides which chance/opponent branches MCCFR samples; likelihood ratios keep the "
      + "estimates unbiased. Each cell below is a traverser information set: colour shows the change in visit share versus uniform external sampling."),
    h("div", { class: "card row" }, h("label", { class: "field" }, "Game", game), h("label", { class: "field" }, "Iterations", iters), h("label", { class: "field" }, "Mixing λ", lam), go, status),
    h("div", { class: "stack", style: "margin-top:16px" }, curve, h("div", { class: "grid2" }, heat, top)));
  go.addEventListener("click", async () => {
    go.setAttribute("disabled", "");
    clear(status);
    status.append(spinner("Sampling..."));
    try {
      const r = await api.post<Resp>("/api/adaptive", { game: game.value, iterations: Number(iters.value), lambda: Number(lam.value) });
      clear(status);
      const ext = r.methods.external, ada = r.methods.adaptive;
      lineChart(curve, [
        { name: "external sampling (uniform)", color: "var(--s1)", points: ext.curve.map((p) => ({ x: p.nodes, y: p.exploitability })) },
        { name: "adaptive allocation", color: "var(--s4)", points: ada.curve.map((p) => ({ x: p.nodes, y: p.exploitability })) },
      ], { title: "Did the extra work pay off? Exploitability vs nodes touched", xLabel: "nodes touched", yLabel: "chips", logX: true, logY: true, height: 280 });
      const te = ext.infosets.reduce((s, i) => s + i.visits, 0) || 1, ta = ada.infosets.reduce((s, i) => s + i.visits, 0) || 1;
      const diffs = ada.infosets.map((i, k) => ({ i, d: i.visits / ta - ext.infosets[k].visits / te }));
      const maxd = Math.max(...diffs.map((x) => Math.abs(x.d)), 1e-12);
      clear(heat);
      heat.append(h("h3", {}, "Allocation shift by information set (blue: less than uniform, red: more)"));
      const wrap = h("div", { style: "display:grid;grid-template-columns:repeat(auto-fill,minmax(14px,1fr));gap:2px" });
      for (const { i, d } of diffs.sort((a, b) => a.i.depth - b.i.depth)) {
        const t = d / maxd;
        const color = t >= 0 ? `color-mix(in oklab, var(--surface-2) ${(1 - t) * 100}%, var(--s8))` : `color-mix(in oklab, var(--surface-2) ${(1 + t) * 100}%, var(--s1))`;
        const cell = h("div", { style: `aspect-ratio:1;border-radius:3px;background:${color}` });
        cell.addEventListener("mousemove", (ev) => showTooltip(ev as MouseEvent, i.key, [["visit share Δ", `${(100 * d).toFixed(3)} pp`], ["priority S", fmt(i.score, 4)],
          ["reach ρ", fmt(i.reach, 4)], ["uncertainty U", fmt(i.uncertainty, 4)], ["regret G", fmt(i.sensitivity, 4)], ["cost C", fmt(i.cost, 1)]]));
        cell.addEventListener("mouseleave", hideTooltip);
        wrap.append(cell);
      }
      heat.append(wrap, h("div", { class: "legend" }, "diverging scale: grey = same as uniform"));
      clear(top);
      const best = [...ada.infosets].sort((a, b) => b.score - a.score).slice(0, 15);
      top.append(h("h3", {}, "Highest-priority information sets"), h("table", { class: "data" },
        h("tr", {}, h("th", {}, "Infoset"), h("th", {}, "S"), h("th", {}, "ρ"), h("th", {}, "U"), h("th", {}, "G"), h("th", {}, "C")),
        ...best.map((i) => h("tr", {}, h("td", {}, i.key), h("td", {}, fmt(i.score, 3)), h("td", {}, fmt(i.reach, 3)), h("td", {}, fmt(i.uncertainty, 3)), h("td", {}, fmt(i.sensitivity, 3)), h("td", {}, fmt(i.cost, 1))))),
      h("p", { class: "small muted" }, "Experiment A (results tab) reports the matched-budget comparison over 8 seeds with confidence intervals."));
    } catch (e) {
      clear(status);
      status.append(h("span", { class: "error" }, String(e)));
    } finally {
      go.removeAttribute("disabled");
    }
  });
}
