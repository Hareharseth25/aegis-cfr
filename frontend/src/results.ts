// Experiment results gallery: figures and headline numbers from results/.

import { api } from "./shared/api";
import { clear, h, spinner } from "./shared/dom";

interface Index { experiments: { name: string; has_results: boolean; figures: string[] }[] }

const TITLES: Record<string, string> = {
  exp01_kuhn_equilibrium: "1 | Exact equilibrium laboratory (Kuhn, Leduc LP)",
  exp02_leduc_comparison: "2 | Leduc: exploitability vs time",
  exp03_cfrplus_vs_dcfr: "3 | CFR+ vs DCFR: where each wins",
  exp04_river_exact: "4 | Exact river solving",
  exp05_action_abstraction: "5 | Action abstraction: exploitability vs tree size",
  exp06_adaptive_sampling: "A | Adaptive trajectory sampling",
  exp07_safe_resolving: "7 | Safe vs naive re-solving",
  exp08_value_network: "8 | Learned continuation values",
  exp09_performance: "9 | Performance scaling",
  exp10_refinement: "B | Dynamic action refinement",
  exp11_learned_allocation: "C | Learned computation allocation",
  exp12_budget_allocation: "12 | Compute-budget allocation",
  exp13_winrate_vs_exploitability: "13 | Win rate vs exploitability (evaluation control)",
};

export function mountResults(root: HTMLElement): void {
  const body = h("div", { class: "stack" });
  root.append(h("h1", {}, "Experiment results"), h("p", { class: "lede" }, "Output of python/experiments/run_all.py. Each figure comes from results/<experiment>/results.json; "
    + "docs/RESULTS.md has the write-up, including the experiments that didn't work out."), body);
  body.append(spinner("Loading..."));
  api.get<Index>("/api/results").then((idx) => {
    clear(body);
    for (const e of idx.experiments) {
      if (!e.figures.length) continue;
      body.append(h("div", { class: "card stack" }, h("h2", {}, TITLES[e.name] ?? e.name),
        h("div", { class: "figure-grid" }, ...e.figures.map((f) => h("a", { href: `/api/figure/${e.name}/${f}`, target: "_blank" }, h("img", { src: `/api/figure/${e.name}/${f}`, alt: `${e.name} ${f}`, loading: "lazy" }))))));
    }
    if (!body.children.length) body.append(h("p", { class: "notice" }, "No results yet - run python/experiments/run_all.py."));
  }).catch((e) => { clear(body); body.append(h("p", { class: "error" }, String(e))); });
}
