// Convergence diagnostics: exploitability of the average strategy (exact best
// response) and the hero information set's average positive regret.

import type { ConvergencePoint } from "../shared/api";
import { lineChart } from "../shared/charts";
import { h } from "../shared/dom";

export function renderConvergence(el: HTMLElement, conv: ConvergencePoint[],
  regret?: { iteration: number; value: number | null }[], unit = "% pot"): void {
  el.innerHTML = "";
  const a = h("div", {});
  el.append(a);
  lineChart(a, [{
    name: "exploitability",
    color: "var(--s1)",
    points: conv.map((p) => ({ x: p.iteration, y: (p.exploitability_pct_pot ?? p.exploitability ?? NaN) as number })),
  }], { title: "Exploitability of the average strategy (exact best response)", xLabel: "iteration", yLabel: unit, logX: true, logY: true, height: 240 });
  if (regret && regret.some((r) => r.value !== null && (r.value as number) > 0)) {
    const b = h("div", {});
    el.append(b);
    lineChart(b, [{
      name: "hero average positive regret",
      color: "var(--s2)",
      points: regret.filter((r) => r.value !== null && (r.value as number) > 0).map((r) => ({ x: r.iteration, y: r.value as number })),
    }], { title: "Hero information set: Σ+ regret / t", xLabel: "iteration", yLabel: "chips (counterfactual)", logX: true, logY: true, height: 220 });
  }
}
