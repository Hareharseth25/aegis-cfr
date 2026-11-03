// Tiny DOM helpers (no framework).

type Attrs = Record<string, string | number | boolean | EventListener | undefined>;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: (Node | string | null | undefined)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v as EventListener);
    else if (k === "class") el.className = String(v);
    else if (k === "html") el.innerHTML = String(v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children) if (c !== null && c !== undefined) el.append(c);
  return el;
}

export function clear(el: Element): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function fmt(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "-";
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return x.toExponential(2);
  return x.toFixed(digits);
}

export function pct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "-";
  return `${(100 * x).toFixed(digits)}%`;
}

// Categorical slots (validated order). Colour by position of the action at a node.
export const SLOT_VARS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
export function slotColor(k: number): string {
  return `var(${SLOT_VARS[k % SLOT_VARS.length]})`;
}

const SUITS: Record<string, string> = { c: "♣", d: "♦", h: "♥", s: "♠" };
export function cardEl(card: string): HTMLElement {
  const rank = card[0].toUpperCase() === "T" ? "10" : card[0].toUpperCase();
  const suit = card[1].toLowerCase();
  return h("div", { class: `pcard ${suit === "d" || suit === "h" ? "red" : ""}`, "aria-label": card },
    h("span", {}, rank), h("span", { class: "suit" }, SUITS[suit] ?? "?"));
}

export function splitCards(s: string): string[] {
  const compact = s.replace(/[\s,]/g, "");
  const out: string[] = [];
  for (let i = 0; i + 1 < compact.length; i += 2) out.push(compact.slice(i, i + 2));
  return out;
}

export function spinner(text = "Working..."): HTMLElement {
  return h("span", { class: "muted" }, h("span", { class: "spinner" }), " ", text);
}
