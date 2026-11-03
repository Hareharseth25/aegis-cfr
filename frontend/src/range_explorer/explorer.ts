// Interactive range explorer: range notation -> 13x13 frequencies, blocker
// effects of a hero hand, and per-class equity against an opposing range.

import { api, NodeStrategy } from "../shared/api";
import { renderMatrix } from "../strategy_matrix/matrix";
import { hideTooltip, showTooltip } from "../shared/charts";
import { clear, fmt, h, spinner, splitCards } from "../shared/dom";

interface RangeResp {
  class_frequency: number[];
  class_names: string[];
  combos: number;
  text: string;
  blocked_fraction?: number;
  conditioned_class_frequency?: number[];
  combo_blocked?: Record<string, number>;
  hero_equity?: { win: number; tie: number; equity: number };
  range_equity?: number;
  class_equity?: (number | null)[];
}

function seqColor(v: number): string {
  // One-hue sequential ramp (blue): light -> dark by magnitude.
  const t = Math.max(0, Math.min(1, v));
  return `color-mix(in oklab, var(--seq-100) ${(1 - t) * 100}%, var(--seq-700))`;
}

function grid(el: HTMLElement, names: string[], values: (number | null)[], label: (v: number) => string, title: string): void {
  el.innerHTML = "";
  el.append(h("h3", {}, title));
  const m = h("div", { class: "matrix" });
  names.forEach((n, k) => {
    const v = values[k];
    const c = h("div", { class: `mcell ${v === null || !Number.isFinite(v as number) || (v as number) <= 0 ? "empty" : ""}`,
      style: v !== null && Number.isFinite(v as number) && (v as number) > 0 ? `background:${seqColor(v as number)}` : "" },
    h("div", { class: "lbl" }, n));
    c.addEventListener("mousemove", (ev) => showTooltip(ev as MouseEvent, n, [["value", v === null ? "-" : label(v as number)]]));
    c.addEventListener("mouseleave", hideTooltip);
    m.append(c);
  });
  el.append(m, h("div", { class: "legend" }, "light = low, dark = high (single-hue scale)"));
}

export function mountRangeExplorer(root: HTMLElement): void {
  const range = h("textarea", {}, "22+,A2s+,K9s+,QTs+,JTs,T9s,98s,87s,ATo+,KJo+,QJo") as HTMLTextAreaElement;
  const villain = h("textarea", {}, "TT+,AQ+,AJs,KQs") as HTMLTextAreaElement;
  const board = h("input", { type: "text", value: "Ah Kd 7c" }) as HTMLInputElement;
  const hero = h("input", { type: "text", value: "AsQs" }) as HTMLInputElement;
  const go = h("button", {}, "Explore");
  const status = h("div", { class: "small muted" });
  const out = h("div", { class: "grid2", style: "margin-top:16px" });
  root.append(h("h1", {}, "Interactive range explorer"),
    h("p", { class: "lede" }, "A range is a weight for each of the 1,326 combos, with dead cards removed. See how the hero's blockers "
      + "change the opponent's range, and the exact equity of each hand class against another range on this board."),
    h("div", { class: "card stack" },
      h("div", { class: "grid2" }, h("label", { class: "field" }, "Range", range), h("label", { class: "field" }, "Opposing range (for equity)", villain)),
      h("div", { class: "grid2" }, h("label", { class: "field" }, "Board (0, 3, 4 or 5 cards)", board), h("label", { class: "field" }, "Hero hand (blockers)", hero)),
      h("div", { class: "row" }, go, status)), out);
  go.addEventListener("click", async () => {
    clear(status);
    status.append(spinner());
    try {
      const r = await api.post<RangeResp>("/api/range", { range: range.value, villain: villain.value, board: board.value.replace(/\s/g, ""), hero: hero.value.trim() || undefined });
      clear(status);
      status.append(`${r.combos} live combinations` + (r.blocked_fraction !== undefined ? ` | hero blocks ${(100 * r.blocked_fraction).toFixed(1)}% of this range` : "")
        + (r.hero_equity ? ` | hero equity ${(100 * r.hero_equity.equity).toFixed(2)}%` : "") + (r.range_equity !== undefined ? ` | range equity ${(100 * r.range_equity).toFixed(2)}%` : ""));
      clear(out);
      const a = h("div", { class: "card" }), b = h("div", { class: "card" }), c = h("div", { class: "card" });
      grid(a, r.class_names, r.class_frequency, (v) => `${(100 * v).toFixed(0)}% of combos`, "Range frequency by class");
      out.append(a);
      if (r.conditioned_class_frequency) {
        const mx = Math.max(...r.conditioned_class_frequency, 1e-12);
        grid(b, r.class_names, r.conditioned_class_frequency.map((v) => v / mx), (v) => fmt(v, 3), "After hero's blockers (relative)");
        out.append(b);
      }
      if (r.class_equity) {
        grid(c, r.class_names, r.class_equity, (v) => `${(100 * v).toFixed(1)}% equity`, "Equity vs opposing range (exact)");
        out.append(c);
      }
    } catch (e) {
      clear(status);
      status.append(h("span", { class: "error" }, String(e)));
    }
  });
  go.click();

  // ---- Equilibrium strategy: how opposing ranges and blockers change it ----
  const villainB = h("textarea", {}, "QQ+,AK") as HTMLTextAreaElement;
  const river = h("input", { type: "text", value: "Ah Kd 7c 4s 2h" }) as HTMLInputElement;
  const solve = h("button", {}, "Solve strategies");
  const sstatus = h("div", { class: "small muted" });
  const sout = h("div", { class: "stack", style: "margin-top:12px" });
  root.append(h("div", { class: "card stack", style: "margin-top:16px" },
    h("h2", {}, "Optimal mixed strategy of this range"),
    h("p", { class: "small muted" }, "Solves a river spot (pot 100, stacks 150, bets ½ and 1 pot) with the range above as the first player against the "
      + "opposing range, and against an alternative opposing range. Click a hand class to see each combination's frequencies and how much of the "
      + "opposing range it blocks - blockers are why combos of the same class can play differently."),
    h("div", { class: "grid2" }, h("label", { class: "field" }, "River board (5 cards)", river), h("label", { class: "field" }, "Alternative opposing range", villainB)),
    h("div", { class: "row" }, solve, sstatus), sout));
  solve.addEventListener("click", async () => {
    solve.setAttribute("disabled", "");
    clear(sstatus);
    sstatus.append(spinner("Solving two river spots..."));
    try {
      const boardStr = splitCards(river.value).join("");
      const vs = [villain.value, villainB.value].filter((v) => v.trim());
      const solves = await Promise.all(vs.map((v) => api.post<{ exploitability_pct_pot: number; nodes: Record<string, NodeStrategy> }>(
        "/api/range/strategy", { range: range.value, villain: v, board: boardStr })));
      const blocks = await Promise.all(vs.map((v) => api.post<RangeResp>("/api/range", { range: range.value, villain: v, board: boardStr })));
      clear(sstatus);
      sstatus.append(solves.map((x, k) => `vs range ${k + 1}: exploitability ${fmt(x.exploitability_pct_pot, 4)}% pot`).join(" | "));
      clear(sout);
      let selected: string | undefined;
      const draw = () => {
        clear(sout);
        for (const key of ["root", "check_then_facing_bet"]) {
          const row = h("div", { class: "grid2" });
          solves.forEach((sv, k) => {
            const node = sv.nodes[key];
            if (!node) return;
            const cell = h("div", { class: "card" });
            cell.append(h("h3", {}, `${key === "root" ? "First action" : "After check, facing a bet"} - vs opposing range ${k + 1}`));
            const m = h("div", {});
            renderMatrix(m, node.class_grid, node.actions, (c) => { selected = c; draw(); }, selected);
            cell.append(m);
            if (selected) {
              const idx = node.combos.map((c, i) => ({ c, i })).filter(({ c, i }) => classOfCombo(c) === selected && node.reach[i] > 0);
              cell.append(h("table", { class: "data", style: "margin-top:8px" },
                h("tr", {}, h("th", {}, "Combo"), h("th", {}, "Blocks opp."), ...node.actions.map((a) => h("th", {}, a))),
                ...idx.map(({ c, i }) => h("tr", {}, h("td", {}, c), h("td", {}, `${(100 * (blocks[k].combo_blocked?.[c] ?? 0)).toFixed(1)}%`),
                  ...node.strategy[i].map((p) => h("td", {}, `${(100 * p).toFixed(1)}%`))))));
            }
            row.append(cell);
          });
          sout.append(row);
        }
      };
      draw();
    } catch (e) {
      clear(sstatus);
      sstatus.append(h("span", { class: "error" }, String(e)));
    } finally {
      solve.removeAttribute("disabled");
    }
  });
}

const RANKS = "23456789TJQKA";
function classOfCombo(combo: string): string {
  const a = combo[0], b = combo[2];
  if (a === b) return a + b;
  const hi = RANKS.indexOf(a) > RANKS.indexOf(b) ? a : b, lo = hi === a ? b : a;
  return hi + lo + (combo[1] === combo[3] ? "s" : "o");
}
