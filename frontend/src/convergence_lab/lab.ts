// Regret-convergence laboratory: run CFR, CFR+, DCFR, LCFR and MCCFR on the
// same game with exact exploitability measurement.

import { api } from "../shared/api";
import { lineChart } from "../shared/charts";
import { clear, fmt, h, slotColor, spinner } from "../shared/dom";

const ALGOS: [string, string][] = [["cfr", "CFR"], ["cfr+", "CFR+"], ["dcfr", "DCFR"], ["lcfr", "LCFR"], ["cfr_sim", "CFR (simultaneous)"], ["es", "External-sampling MCCFR"], ["os", "Outcome-sampling MCCFR"]];
// The vector river solver only does full-traversal, alternating-update CFR variants.
const RIVER_ALGOS = new Set(["cfr", "cfr+", "dcfr", "lcfr"]);
const supported = (game: string, algo: string) => game !== "river" || RIVER_ALGOS.has(algo);

interface Resp {
  work_unit: string;
  series: { algorithm: string; unit: string; points: { iteration: number; seconds: number; exploitability: number; nodes?: number; avg_positive_regret?: number }[] }[];
}

export function mountConvergenceLab(root: HTMLElement): void {
  const game = h("select", {}, ...[["kuhn", "Kuhn poker"], ["leduc", "Leduc hold'em"], ["leduc6", "Leduc (6 ranks)"], ["river", "No-Limit river (demo spot)"]]
    .map(([v, l]) => h("option", { value: v, selected: v === "leduc" }, l))) as HTMLSelectElement;
  const iters = h("input", { type: "number", value: "1000" }) as HTMLInputElement;
  const chosen = new Set(["cfr", "cfr+", "dcfr", "lcfr"]);
  let last: Resp | null = null;
  const chips = h("div", { class: "chips" });
  const renderChips = () => {
    clear(chips);
    for (const k of [...chosen]) if (!supported(game.value, k)) chosen.delete(k);
    for (const [k, l] of ALGOS) {
      if (!supported(game.value, k)) continue;
      chips.append(h("span", { class: `chip ${chosen.has(k) ? "active" : ""}`, onclick: () => { chosen.has(k) ? chosen.delete(k) : chosen.add(k); renderChips(); } }, l));
    }
  };
  renderChips();
  const xmode = h("select", {}, h("option", { value: "iteration" }, "x: iterations"), h("option", { value: "seconds" }, "x: seconds"), h("option", { value: "nodes" }, "x: work")) as HTMLSelectElement;
  game.addEventListener("change", () => { renderChips(); last = null; clear(chart); clear(regret); clear(table); });
  const go = h("button", {}, "Run");
  const status = h("div", { class: "small muted" });
  const chart = h("div", { class: "card" });
  const regret = h("div", { class: "card" });
  const table = h("div", { class: "card" });
  root.append(h("h1", {}, "Regret-convergence laboratory"),
    h("p", { class: "lede" }, "Same game and budget for every algorithm. Exploitability is the exact best-response gain against the average strategy. "
      + "Sampling methods run 50x more (cheaper) iterations, so compare them by seconds or nodes touched."),
    h("div", { class: "card stack" }, h("div", { class: "row" }, h("label", { class: "field" }, "Game", game), h("label", { class: "field" }, "Iterations", iters),
      h("label", { class: "field" }, "Axis", xmode)), chips, h("div", { class: "row" }, go, status)),
    h("div", { class: "stack", style: "margin-top:16px" }, chart, h("div", { class: "grid2" }, regret, table)));
  const draw = () => {
    if (!last) return;
    const key = xmode.value as "iteration" | "seconds" | "nodes";
    const xLabel = key === "nodes" ? last.work_unit : key;
    // Only plot points that actually carry the chosen x value.
    lineChart(chart, last.series.map((s) => ({
      name: ALGOS.find(([k]) => k === s.algorithm)?.[1] ?? s.algorithm,
      color: slotColor(ALGOS.findIndex(([k]) => k === s.algorithm)),
      points: s.points.filter((p) => p[key] !== undefined).map((p) => ({ x: p[key] as number, y: p.exploitability })),
    })), { title: "Exploitability vs " + xLabel, xLabel, yLabel: last.series[0]?.unit ?? "", logX: true, logY: true, height: 320 });
    const withRegret = last.series.filter((s) => s.points.some((p) => p.avg_positive_regret !== undefined));
    lineChart(regret, withRegret.map((s) => ({
      name: ALGOS.find(([k]) => k === s.algorithm)?.[1] ?? s.algorithm,
      color: slotColor(ALGOS.findIndex(([k]) => k === s.algorithm)),
      points: s.points.map((p) => ({ x: p.iteration, y: p.avg_positive_regret ?? NaN })),
    })), { title: "Average positive cumulative regret Σ+R/t", xLabel: "iteration", logX: true, logY: true, height: 260 });
    clear(table);
    table.append(h("h3", {}, "Final values"), h("table", { class: "data" }, h("tr", {}, h("th", {}, "Algorithm"), h("th", {}, "Iterations"), h("th", {}, "Seconds"), h("th", {}, "Exploitability")),
      ...last.series.map((s) => { const p = s.points[s.points.length - 1]; return h("tr", {}, h("td", {}, ALGOS.find(([k]) => k === s.algorithm)?.[1] ?? s.algorithm), h("td", {}, String(p.iteration)), h("td", {}, fmt(p.seconds, 3)), h("td", {}, `${fmt(p.exploitability, 6)} ${s.unit}`)); })));
  };
  xmode.addEventListener("change", draw);
  go.addEventListener("click", async () => {
    go.setAttribute("disabled", "");
    clear(status);
    status.append(spinner("Running solvers..."));
    try {
      if (!chosen.size) throw new Error("pick at least one algorithm");
      last = await api.post<Resp>("/api/convergence", { game: game.value, algorithms: [...chosen], iterations: Number(iters.value) });
      clear(status);
      draw();
    } catch (e) {
      clear(status);
      status.append(h("span", { class: "error" }, String(e)));
    } finally {
      go.removeAttribute("disabled");
    }
  });
}
