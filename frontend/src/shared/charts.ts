// Dependency-free SVG charts with hover tooltips.
//  * lineChart: one y-axis, optional log scales, crosshair + tooltip, legend.
//  * Thin 2px lines, recessive hairline grid, text in ink colours.

import { fmt, h } from "./dom";

const NS = "http://www.w3.org/2000/svg";

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

export function tooltip(): HTMLElement {
  return document.getElementById("tooltip") as HTMLElement;
}

export function showTooltip(ev: MouseEvent, title: string, rows: [string, string][]): void {
  const t = tooltip();
  t.innerHTML = "";
  t.append(h("div", { class: "tt-title" }, title));
  for (const [k, v] of rows) {
    const row = h("div", { class: "tt-row" }, h("span", {}, k));
    row.append(h("b", {}, v));
    t.append(row);
  }
  t.hidden = false;
  const pad = 14;
  const w = t.offsetWidth, hgt = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
  if (y + hgt > window.innerHeight - 8) y = ev.clientY - hgt - pad;
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}

export function hideTooltip(): void {
  tooltip().hidden = true;
}

export interface Series {
  name: string;
  color: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
}

export interface LineOpts {
  title?: string;
  xLabel?: string;
  yLabel?: string;
  logX?: boolean;
  logY?: boolean;
  width?: number;
  height?: number;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
}

function ticks(min: number, max: number, log: boolean): number[] {
  if (log) {
    const out: number[] = [];
    for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
      const v = 10 ** e;
      if (v >= min * 0.999 && v <= max * 1.001) out.push(v);
    }
    if (out.length < 2) return [min, max];
    return out;
  }
  const span = max - min || 1;
  const step = 10 ** Math.floor(Math.log10(span / 4));
  const nice = [1, 2, 5, 10].map((m) => m * step).find((s) => span / s <= 6) ?? step * 10;
  const out: number[] = [];
  for (let v = Math.ceil(min / nice) * nice; v <= max + 1e-12; v += nice) out.push(v);
  return out;
}

// Use the container's real width so text keeps its intended size.
function widthOf(container: HTMLElement, fallback = 640): number {
  const w = container.clientWidth;
  return w > 200 ? Math.min(w, 1400) : fallback;
}

export function lineChart(container: HTMLElement, series: Series[], o: LineOpts = {}): void {
  container.innerHTML = "";
  const W = o.width ?? widthOf(container), H = o.height ?? 300;
  const m = { l: 62, r: 16, t: o.title ? 28 : 10, b: 40 };
  const pts = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && (!o.logX || p.x > 0) && (!o.logY || p.y > 0));
  if (!pts.length) {
    container.append(h("p", { class: "muted small" }, "No data yet."));
    return;
  }
  let x0 = Math.min(...pts.map((p) => p.x)), x1 = Math.max(...pts.map((p) => p.x));
  let y0 = Math.min(...pts.map((p) => p.y)), y1 = Math.max(...pts.map((p) => p.y));
  if (x0 === x1) x1 = x0 + 1;
  if (y0 === y1) { y1 = y0 * 1.1 + 1e-9; y0 = y0 * 0.9; }
  if (!o.logY) { const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad; }
  const fx = (x: number) => m.l + ((o.logX ? Math.log10(x) - Math.log10(x0) : x - x0) / (o.logX ? Math.log10(x1) - Math.log10(x0) : x1 - x0)) * (W - m.l - m.r);
  const fy = (y: number) => H - m.b - ((o.logY ? Math.log10(y) - Math.log10(y0) : y - y0) / (o.logY ? Math.log10(y1) - Math.log10(y0) : y1 - y0)) * (H - m.t - m.b);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" });
  if (o.title) { const t = svg("text", { x: m.l, y: 16, class: "title" }); t.textContent = o.title; root.append(t); }
  const yf = o.yFormat ?? ((v: number) => fmt(v, 3));
  const xf = o.xFormat ?? ((v: number) => fmt(v, 0));
  for (const v of ticks(y0, y1, !!o.logY)) {
    root.append(svg("line", { x1: m.l, x2: W - m.r, y1: fy(v), y2: fy(v), class: "gridline" }));
    const t = svg("text", { x: m.l - 6, y: fy(v) + 4, "text-anchor": "end" });
    t.textContent = yf(v);
    root.append(t);
  }
  for (const v of ticks(x0, x1, !!o.logX)) {
    const t = svg("text", { x: fx(v), y: H - m.b + 16, "text-anchor": "middle" });
    t.textContent = xf(v);
    root.append(t);
  }
  root.append(svg("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }));
  if (o.xLabel) { const t = svg("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }); t.textContent = o.xLabel; root.append(t); }
  if (o.yLabel) {
    const t = svg("text", { x: -(m.t + H - m.b) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle" });
    t.textContent = o.yLabel;
    root.append(t);
  }
  for (const s of series) {
    const ps = s.points.filter((p) => Number.isFinite(p.y) && (!o.logY || p.y > 0) && (!o.logX || p.x > 0));
    if (!ps.length) continue;
    const d = ps.map((p, i) => `${i ? "L" : "M"}${fx(p.x).toFixed(1)},${fy(p.y).toFixed(1)}`).join("");
    root.append(svg("path", { d, class: "series", stroke: s.color, "stroke-dasharray": s.dashed ? "5 4" : "" }));
    const last = ps[ps.length - 1];
    root.append(svg("circle", { cx: fx(last.x), cy: fy(last.y), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
  }
  // Hover layer: crosshair + nearest point per series.
  const cross = svg("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" });
  root.append(cross);
  const hit = svg("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
  root.append(hit);
  hit.addEventListener("mousemove", (ev) => {
    const rect = root.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    let best: { x: number; rows: [string, string][] } | null = null;
    const rows: [string, string][] = [];
    let bx = 0;
    for (const s of series) {
      let near = s.points[0], dist = Infinity;
      for (const p of s.points) { const d = Math.abs(fx(p.x) - px); if (d < dist) { dist = d; near = p; } }
      if (!near) continue;
      bx = near.x;
      rows.push([s.name, yf(near.y)]);
    }
    best = { x: bx, rows };
    cross.setAttribute("x1", String(px));
    cross.setAttribute("x2", String(px));
    cross.setAttribute("visibility", "visible");
    showTooltip(ev, `${o.xLabel ?? "x"} ~ ${xf(best.x)}`, best.rows);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTooltip(); });
  container.append(root);
  if (series.length >= 2) {
    const legend = h("div", { class: "legend" });
    for (const s of series) legend.append(h("span", {}, h("span", { class: "sw", style: `background:${s.color}` }), s.name));
    container.append(legend);
  }
}

// Horizontal action-frequency bars (one row per action), labelled directly.
export function actionBars(container: HTMLElement, items: { label: string; value: number; color: string; sub?: string }[]): void {
  container.innerHTML = "";
  for (const it of items) {
    const row = h("div", { class: "actbar" },
      h("span", {}, it.label),
      h("div", { class: "track" }, h("div", { class: "fill", style: `width:${Math.max(0, Math.min(1, it.value)) * 100}%;background:${it.color}` })),
      h("span", { class: "pct" }, `${(100 * it.value).toFixed(1)}%`));
    row.addEventListener("mousemove", (ev) => showTooltip(ev as MouseEvent, it.label, [["frequency", `${(100 * it.value).toFixed(2)}%`], ...(it.sub ? [["EV", it.sub] as [string, string]] : [])]));
    row.addEventListener("mouseleave", hideTooltip);
    container.append(row);
  }
}

// Vertical bar chart with one bar per category (used for audits and profiles).
export function barChart(container: HTMLElement, items: { label: string; value: number; color: string }[], o: { yLabel?: string; logY?: boolean; height?: number; format?: (v: number) => string } = {}): void {
  container.innerHTML = "";
  const W = widthOf(container), H = o.height ?? 260, m = { l: 62, r: 10, t: 12, b: 54 };
  const vals = items.map((i) => i.value).filter((v) => Number.isFinite(v) && (!o.logY || v > 0));
  if (!vals.length) { container.append(h("p", { class: "muted small" }, "No data.")); return; }
  const y1 = Math.max(...vals) * 1.08, y0 = o.logY ? Math.min(...vals) / 2 : 0;
  const fy = (v: number) => H - m.b - (o.logY ? (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)) : (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" });
  const f = o.format ?? ((v: number) => fmt(v, 3));
  for (const v of ticks(Math.max(y0, 1e-12), y1, !!o.logY)) {
    root.append(svg("line", { x1: m.l, x2: W - m.r, y1: fy(v), y2: fy(v), class: "gridline" }));
    const t = svg("text", { x: m.l - 6, y: fy(v) + 4, "text-anchor": "end" });
    t.textContent = f(v);
    root.append(t);
  }
  const bw = (W - m.l - m.r) / items.length;
  items.forEach((it, i) => {
    if (!Number.isFinite(it.value) || (o.logY && it.value <= 0)) return;
    const x = m.l + i * bw + bw * 0.15, top = fy(it.value), base = H - m.b;
    const r = svg("rect", { x, y: top, width: bw * 0.7, height: Math.max(1, base - top), rx: 4, fill: it.color });
    r.addEventListener("mousemove", (ev) => showTooltip(ev, it.label, [["value", f(it.value)]]));
    r.addEventListener("mouseleave", hideTooltip);
    root.append(r);
    const t = svg("text", { x: x + bw * 0.35, y: H - m.b + 14, "text-anchor": "middle" });
    t.textContent = it.label.length > 14 ? `${it.label.slice(0, 13)}...` : it.label;
    root.append(t);
  });
  root.append(svg("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }));
  if (o.yLabel) {
    const t = svg("text", { x: -(m.t + H - m.b) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle" });
    t.textContent = o.yLabel;
    root.append(t);
  }
  container.append(root);
}

// Grouped bars: one group per category, one coloured bar per series, legend below.
export function groupedBars(container: HTMLElement, groups: string[], series: { name: string; color: string; values: number[] }[],
  o: { yLabel?: string; logY?: boolean; height?: number; format?: (v: number) => string } = {}): void {
  container.innerHTML = "";
  const W = widthOf(container), H = o.height ?? 280, m = { l: 62, r: 10, t: 12, b: 34 };
  const vals = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v) && (!o.logY || v > 0));
  if (!vals.length) { container.append(h("p", { class: "muted small" }, "No data.")); return; }
  const y1 = Math.max(...vals) * 1.15, y0 = o.logY ? Math.min(...vals) / 2 : 0;
  const fy = (v: number) => H - m.b - (o.logY ? (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)) : (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const f = o.format ?? ((v: number) => fmt(v, 3));
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" });
  for (const v of ticks(Math.max(y0, 1e-12), y1, !!o.logY)) {
    root.append(svg("line", { x1: m.l, x2: W - m.r, y1: fy(v), y2: fy(v), class: "gridline" }));
    const t = svg("text", { x: m.l - 6, y: fy(v) + 4, "text-anchor": "end" });
    t.textContent = f(v);
    root.append(t);
  }
  const gw = (W - m.l - m.r) / groups.length;
  const bw = (gw * 0.8) / series.length;
  groups.forEach((g, gi) => {
    series.forEach((s, si) => {
      const v = s.values[gi];
      if (!Number.isFinite(v) || (o.logY && v <= 0)) return;
      // 2px surface gap between adjacent bars.
      const x = m.l + gi * gw + gw * 0.1 + si * bw + 1, top = fy(v), base = H - m.b;
      const r = svg("rect", { x, y: top, width: Math.max(1, bw - 2), height: Math.max(1, base - top), rx: 3, fill: s.color });
      r.addEventListener("mousemove", (ev) => showTooltip(ev, `${g} | ${s.name}`, [["value", f(v)]]));
      r.addEventListener("mouseleave", hideTooltip);
      root.append(r);
    });
    const t = svg("text", { x: m.l + gi * gw + gw / 2, y: H - m.b + 16, "text-anchor": "middle" });
    t.textContent = g;
    root.append(t);
  });
  root.append(svg("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }));
  if (o.yLabel) {
    const t = svg("text", { x: -(m.t + H - m.b) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle" });
    t.textContent = o.yLabel;
    root.append(t);
  }
  container.append(root);
  const legend = h("div", { class: "legend" });
  for (const s of series) legend.append(h("span", {}, h("span", { class: "sw", style: `background:${s.color}` }), s.name));
  container.append(legend);
}
