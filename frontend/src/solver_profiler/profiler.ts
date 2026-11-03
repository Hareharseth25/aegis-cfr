// Solver profiler: benchmark results (exp09) plus a live measurement.

import { api } from "../shared/api";
import { barChart, groupedBars, lineChart } from "../shared/charts";
import { clear, fmt, h, spinner } from "../shared/dom";

interface BenchRow { bench: string; config: string; [k: string]: unknown }
interface Resp {
  available: boolean;
  results?: { bench: BenchRow[]; simd_ablation: Record<string, { vectorized_ms: number; scalar_ms: number; speedup: number }>;
    gpu_vs_cpu: { ranges: string; backend: string; dtype: string; ms_per_iter: number; expl: number }[];
    value_net: { backend: string; batch: number; ms: number }[]; "latency_to_0.1pct_pot": Record<string, { iterations: number; solver_seconds: number; hands: number[]; memory_mb: number }> };
  hw_counters: { available: boolean; backend: string; reason: string };
  live_river_ms_per_iteration: number;
  threads: number;
}

export function mountProfiler(root: HTMLElement): void {
  const body = h("div", { class: "stack" });
  root.append(h("h1", {}, "Solver profiler"), h("p", { class: "lede" }, "Throughput, scaling, memory and latency of the C++ core, the SIMD ablation build, and CPU vs GPU (PyTorch MPS) alternatives. "
    + "Run python/experiments/exp09_performance.py to refresh."), body);
  body.append(spinner("Measuring..."));
  api.get<Resp>("/api/profiler").then((r) => {
    clear(body);
    body.append(h("div", { class: "card" }, h("h3", {}, "This machine, right now"),
      h("p", {}, `Full-range river DCFR: ${fmt(r.live_river_ms_per_iteration, 3)} ms per iteration | ${r.threads} hardware threads`),
      h("p", { class: "small muted" }, r.hw_counters.available ? `Hardware counters: ${r.hw_counters.backend}` : `Hardware counters unavailable: ${r.hw_counters.reason}`)));
    if (!r.available || !r.results) { body.append(h("p", { class: "notice" }, "No stored benchmark results yet.")); return; }
    const res = r.results;
    const flat = res.bench.filter((b) => b.bench === "tabular_cfr");
    const c1 = h("div", {});
    barChart(c1, flat.map((b, k) => ({ label: String(b.config).replace("leduc ", ""), value: Number(b.nodes_per_sec), color: k ? "var(--s2)" : "var(--s1)" })), { yLabel: "nodes / second", logY: true, format: (v) => v.toExponential(1) });
    const turn = res.bench.filter((b) => b.bench === "turn_cfr_threads");
    const c2 = h("div", {});
    lineChart(c2, [{ name: "speed-up", color: "var(--s1)", points: turn.map((b) => ({ x: Number(b.threads), y: Number(turn[0].ms_per_iteration) / Number(b.ms_per_iteration) })) },
      { name: "ideal", color: "var(--text-muted)", dashed: true, points: turn.map((b) => ({ x: Number(b.threads), y: Number(b.threads) })) }],
    { title: "Turn+river CFR thread scaling", xLabel: "threads", yLabel: "speed-up", height: 240 });
    body.append(h("div", { class: "grid2" }, h("div", { class: "card" }, h("h3", {}, "Memory layout: flat arrays vs object-per-node (Leduc CFR)"), c1), h("div", { class: "card" }, c2)));
    const gpu = h("div", {});
    const groups = [...new Set(res.gpu_vs_cpu.map((g) => g.ranges))];
    const backends: [string, string, string][] = [["C++", "float64", "var(--s1)"], ["torch-cpu", "float64", "var(--s2)"],
      ["torch-cpu", "float32", "var(--s3)"], ["torch-mps", "float32", "var(--s4)"]];
    const series = backends.map(([b, dt, color]) => ({ name: `${b} (${dt})`, color,
      values: groups.map((g) => res.gpu_vs_cpu.find((x) => x.ranges === g && x.backend === b && x.dtype === dt)?.ms_per_iter ?? NaN) }));
    const simdRows = Object.entries(res.simd_ablation);
    const gpuCard = h("div", { class: "card" }, h("h3", {}, "River DCFR: C++ sweep vs PyTorch dense matmul on CPU and GPU (ms per iteration, log scale)"), gpu,
      h("table", { class: "data" }, h("tr", {}, h("th", {}, "Ranges"), h("th", {}, "Backend"), h("th", {}, "ms/iter"), h("th", {}, "Exploitability after 60 it (% pot)")),
        ...res.gpu_vs_cpu.map((g) => h("tr", {}, h("td", {}, g.ranges), h("td", {}, `${g.backend} (${g.dtype})`), h("td", {}, fmt(g.ms_per_iter, 3)), h("td", {}, fmt(g.expl, 4))))));
    body.append(gpuCard);
    groupedBars(gpu, groups.map((g) => `${g} ranges`), series, { yLabel: "ms / iteration", logY: true, height: 300 });
    body.append(h("div", { class: "grid2" },
      h("div", { class: "card" }, h("h3", {}, "SIMD ablation (auto-vectorisation off vs on)"), h("table", { class: "data" }, h("tr", {}, h("th", {}, "Benchmark"), h("th", {}, "Vectorised ms"), h("th", {}, "Scalar ms"), h("th", {}, "Speed-up")),
        ...simdRows.map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, fmt(v.vectorized_ms, 3)), h("td", {}, fmt(v.scalar_ms, 3)), h("td", {}, `${fmt(v.speedup, 2)}x`))))),
      h("div", { class: "card" }, h("h3", {}, "Latency to 0.1% pot exploitability (river)"), h("table", { class: "data" }, h("tr", {}, h("th", {}, "Ranges"), h("th", {}, "Hands"), h("th", {}, "Iterations"), h("th", {}, "Seconds"), h("th", {}, "Memory")),
        ...Object.entries(res["latency_to_0.1pct_pot"]).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, v.hands.join("x")), h("td", {}, String(v.iterations)), h("td", {}, fmt(v.solver_seconds, 3)), h("td", {}, `${fmt(v.memory_mb, 1)} MB`)))))));
    const vn = h("div", {});
    lineChart(vn, ["C++", "torch-cpu", "torch-mps"].map((b, k) => ({ name: b, color: `var(--s${k + 1})`, points: res.value_net.filter((x) => x.backend === b).map((x) => ({ x: x.batch, y: (x.batch / x.ms) * 1e3 })) })),
      { title: "Value-network inference throughput", xLabel: "batch size", yLabel: "positions / s", logX: true, logY: true, height: 260 });
    body.append(h("div", { class: "card" }, vn));
  }).catch((e) => { clear(body); body.append(h("p", { class: "error" }, String(e))); });
}
