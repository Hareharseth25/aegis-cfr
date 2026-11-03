// Poker table header: board, hero hand and the betting situation.

import { cardEl, fmt, h } from "../shared/dom";

export interface TableState {
  board: string[];
  hero?: string[];
  pot: number;
  facing?: number;
  stack: number;
  toAct?: string;
}

export function renderTable(el: HTMLElement, s: TableState): void {
  el.innerHTML = "";
  const stat = (label: string, value: string) =>
    h("div", { class: "stat" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value));
  const board = h("div", { class: "stack board" }, h("div", { class: "stat" }, h("div", { class: "label" }, "Board")),
    h("div", { class: "cards" }, ...s.board.map(cardEl)));
  const items: HTMLElement[] = [board];
  if (s.hero?.length) {
    items.push(h("div", { class: "stack" }, h("div", { class: "stat" }, h("div", { class: "label" }, "Hero hand")),
      h("div", { class: "cards" }, ...s.hero.map(cardEl))));
  }
  items.push(stat("Pot", `${fmt(s.pot, 0)} BB`));
  if (s.facing !== undefined && s.facing > 0) items.push(stat("Facing bet", `${fmt(s.facing, 0)} BB`));
  items.push(stat("Effective stack", `${fmt(s.stack, 0)} BB`));
  if (s.toAct) items.push(stat("To act", s.toAct));
  el.append(h("div", { class: "table-felt" }, ...items));
}
