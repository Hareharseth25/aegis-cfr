// 13x13 hand-class strategy matrix. Each cell is split horizontally by the
// reach-weighted action mix of that class; empty cells are classes the player
// never reaches this node with. Hover shows the mix, click selects the class.

import { hideTooltip, showTooltip } from "../shared/charts";
import { h, slotColor } from "../shared/dom";

export interface GridCell {
  name: string;
  mass: number;
  mix: number[] | null;
}

export function renderMatrix(el: HTMLElement, grid: GridCell[], actions: string[], onSelect?: (cls: string) => void,
  selected?: string): void {
  el.innerHTML = "";
  const m = h("div", { class: "matrix", role: "grid", "aria-label": "Hand class strategy matrix" });
  for (const cell of grid) {
    const c = h("div", { class: `mcell ${cell.mix ? "" : "empty"} ${cell.name === selected ? "selected" : ""}`, role: "gridcell" });
    if (cell.mix) {
      cell.mix.forEach((f, k) => {
        if (f > 0.001) c.append(h("div", { class: "seg", style: `width:${(f * 100).toFixed(2)}%;background:${slotColor(k)}` }));
      });
    }
    c.append(h("div", { class: "lbl" }, cell.name));
    c.addEventListener("mousemove", (ev) => {
      const rows: [string, string][] = cell.mix
        ? actions.map((a, k) => [a, `${(100 * (cell.mix as number[])[k]).toFixed(1)}%`])
        : [["", "not in range here"]];
      showTooltip(ev as MouseEvent, cell.name, rows);
    });
    c.addEventListener("mouseleave", hideTooltip);
    if (onSelect) c.addEventListener("click", () => onSelect(cell.name));
    m.append(c);
  }
  el.append(m);
  const legend = h("div", { class: "legend" });
  actions.forEach((a, k) => legend.append(h("span", {}, h("span", { class: "sw", style: `background:${slotColor(k)}` }), a)));
  el.append(legend);
}
