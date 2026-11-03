"""Experiment 9 - Performance scaling: nodes/sec, memory, latency, CPU vs GPU.

  * C++ microbenchmarks (benchmarks/aegis_bench.cpp): evaluator throughput,
    flat contiguous arrays vs object-per-node CFR, river kernel throughput by
    range width, turn-tree thread scaling, parallel MCCFR, value-net inference.
  * SIMD ablation: the same benchmarks built with auto-vectorisation disabled
    (build-novec, -DAEGIS_NO_VECTORIZE=ON).
  * CPU vs GPU: the same river DCFR in PyTorch (dense matmul terminals) on
    CPU and Apple MPS vs the C++ solver; value-network inference in PyTorch
    CPU / MPS vs C++ (Accelerate SGEMM) across batch sizes.
  * Latency: time to reach 0.1% pot exploitability on river spots.
Hardware counters are recorded only when the OS grants access (macOS: sudo).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ROOT, color, plt, save_fig, save_json  # noqa: E402

import aegis  # noqa: E402
from evaluation.torch_river import TorchRiverCFR  # noqa: E402
from models.value_net import ValueNet  # noqa: E402

NAME = "exp09_performance"


def run_bench(build_dir, args=()):
    exe = os.path.join(ROOT, build_dir, "aegis_bench")
    if not os.path.exists(exe):
        return None
    out = subprocess.check_output([exe, *args], text=True)
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def ensure_novec_build():
    bdir = os.path.join(ROOT, "build-novec")
    if not os.path.exists(os.path.join(bdir, "aegis_bench")):
        subprocess.check_call(["cmake", "-S", ROOT, "-B", bdir, "-DCMAKE_BUILD_TYPE=Release", "-DAEGIS_NO_VECTORIZE=ON",
                               "-DAEGIS_BUILD_PYTHON=OFF"], stdout=subprocess.DEVNULL)
        subprocess.check_call(["cmake", "--build", bdir, "-j8", "--target", "aegis_bench"], stdout=subprocess.DEVNULL)
    return "build-novec"


def river(ranges):
    c = aegis.TreeConfig()
    c.board = aegis.parse_cards("AsKh8c4d2s")
    c.pot, c.effective_stack = 100, 150
    s = aegis.StreetSizes()
    s.bet_fractions, s.raise_fractions, s.max_raises = [0.33, 0.75, 1.5], [1.0], 3
    c.set_sizes(s, s)
    tree = aegis.PublicTree.build(c)
    r0 = aegis.Range.parse(ranges[0]).remove(c.board)
    r1 = aegis.Range.parse(ranges[1]).remove(c.board)
    return tree, aegis.SubgameContext(tree, r0, r1)


RANGES = {"full": ("random", "random"),
          "medium": ("22+,A2+,K9+,QT+,JT,T9s,98s,87s,76s,65s", "22+,A2+,K7+,Q9+,J9+,T9,98s,87s,76s"),
          "narrow": ("AA,KK,AK,AQs,88,44", "QQ-99,AQ,AJs,KQ")}


def gpu_vs_cpu(iters=60):
    rows = []
    has_mps = torch.backends.mps.is_available()
    for label, rr in RANGES.items():
        tree, ctx = river(rr)
        cpp = aegis.VectorCFR(ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
        t = time.perf_counter()
        cpp.iterate(iters)
        t_cpp = (time.perf_counter() - t) / iters
        e_cpp = ctx.exploitability(cpp.average_policy())["exploitability_pct_pot"]
        rows.append({"ranges": label, "backend": "C++", "dtype": "float64", "ms_per_iter": 1e3 * t_cpp, "expl": e_cpp,
                     "hands": [ctx.n(0), ctx.n(1)]})
        configs = [("torch-cpu", "cpu", torch.float64), ("torch-cpu", "cpu", torch.float32)]
        if has_mps:
            configs.append(("torch-mps", "mps", torch.float32))
        for name, dev, dt in configs:
            tr = TorchRiverCFR(tree, ctx, dev, dt)
            tr.iterate(3)
            tr.sync()
            t = time.perf_counter()
            tr.iterate(iters - 3)
            tr.sync()
            ms = 1e3 * (time.perf_counter() - t) / (iters - 3)
            e = ctx.exploitability(tr.average_policy())["exploitability_pct_pot"]
            rows.append({"ranges": label, "backend": name, "dtype": str(dt).split(".")[-1], "ms_per_iter": ms,
                         "expl": e, "hands": [ctx.n(0), ctx.n(1)]})
        print(label, [(r["backend"], r["dtype"], round(r["ms_per_iter"], 3), round(r["expl"], 4))
                      for r in rows if r["ranges"] == label], flush=True)
    return rows


def value_net_inference():
    rows = []
    model = ValueNet((512, 512)).eval()
    net = model.to_aegis()
    devices = [("torch-cpu", "cpu")] + ([("torch-mps", "mps")] if torch.backends.mps.is_available() else [])
    for batch in (1, 16, 64, 240, 1024):
        x = np.random.rand(batch, aegis.NN_INPUT_SIZE).astype(np.float32)
        reps = max(3, 2000 // batch)
        t = time.perf_counter()
        for _ in range(reps):
            net.forward(x)
        rows.append({"backend": "C++", "batch": batch, "ms": 1e3 * (time.perf_counter() - t) / reps})
        for name, dev in devices:
            m = model.to(dev)
            xt = torch.tensor(x, device=dev)
            with torch.no_grad():
                m(xt)
                if dev == "mps":
                    torch.mps.synchronize()
                t = time.perf_counter()
                for _ in range(reps):
                    y = m(xt)
                    y.cpu()  # include the transfer back to the solver
                if dev == "mps":
                    torch.mps.synchronize()
            rows.append({"backend": name, "batch": batch, "ms": 1e3 * (time.perf_counter() - t) / reps})
    return rows


def latency(target=0.1):
    out = {}
    for label, rr in RANGES.items():
        tree, ctx = river(rr)
        s = aegis.VectorCFR(ctx, aegis.VectorCFRConfig(aegis.CFRConfig.dcfr()))
        t_solve, it, step = 0.0, 0, 5
        while True:
            t = time.perf_counter()
            s.iterate(step)
            t_solve += time.perf_counter() - t
            it += step
            if ctx.exploitability(s.average_policy())["exploitability_pct_pot"] <= target or it > 20000:
                break
        out[label] = {"iterations": it, "solver_seconds": t_solve, "hands": [ctx.n(0), ctx.n(1)],
                      "memory_mb": s.memory_bytes / 1e6}
    return out


def plots(res):
    # Thread scaling.
    turn = [r for r in res["bench"] if r["bench"] == "turn_cfr_threads"]
    if turn:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        th = [r["threads"] for r in turn]
        sp = [turn[0]["ms_per_iteration"] / r["ms_per_iteration"] for r in turn]
        ax.plot(th, sp, color=color("C++"), marker="o", label="measured speed-up")
        ax.plot(th, th, color="#898781", linestyle="--", label="ideal")
        ax.set_xlabel("threads")
        ax.set_ylabel("speed-up vs 1 thread")
        ax.set_title("Turn+river CFR: parallel scaling over river cards")
        ax.legend()
        save_fig(fig, NAME, "thread_scaling.png")
    # CPU vs GPU river.
    rows = res["gpu_vs_cpu"]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    labels = list(RANGES)
    backends = [("C++", "float64"), ("torch-cpu", "float64"), ("torch-cpu", "float32"), ("torch-mps", "float32")]
    width = 0.2
    for k, (b, dt) in enumerate(backends):
        ys = [next((r["ms_per_iter"] for r in rows if r["ranges"] == lab and r["backend"] == b and r["dtype"] == dt), np.nan)
              for lab in labels]
        name = b if b != "torch-cpu" else "torch-cpu"
        ax.bar(np.arange(len(labels)) + (k - 1.5) * width, ys, width * 0.9, label=f"{b} ({dt})",
               color=color(name), alpha=1.0 if dt == "float64" or b == "torch-mps" else 0.55)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{lab} ranges" for lab in labels])
    ax.set_ylabel("ms per DCFR iteration (log)")
    ax.set_title("River DCFR: C++ sweep vs PyTorch matmul (CPU / GPU)")
    ax.legend(fontsize=8)
    save_fig(fig, NAME, "river_cpu_vs_gpu.png")
    # Value net inference.
    rows = res["value_net"]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for b in ("C++", "torch-cpu", "torch-mps"):
        pts = [(r["batch"], r["batch"] / r["ms"] * 1e3) for r in rows if r["backend"] == b]
        if pts:
            ax.plot(*zip(*pts), color=color(b), marker="o", label=b)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size (positions)")
    ax.set_ylabel("positions / second")
    ax.set_title("Value-network inference throughput")
    ax.legend()
    save_fig(fig, NAME, "value_net_inference.png")


def run():
    res = {"bench": run_bench("build") or [], "bench_novec": run_bench(ensure_novec_build(), ["--only", "river"]) or []}
    res["bench_novec"] += run_bench("build-novec", ["--only", "turn"]) or []
    simd = {}
    for r in res["bench"]:
        if r["bench"] in ("river_cfr", "turn_cfr_threads"):
            match = [x for x in res["bench_novec"] if x["bench"] == r["bench"] and x["config"] == r["config"]]
            if match:
                simd[f"{r['bench']} {r['config']}"] = {"vectorized_ms": r["ms_per_iteration"],
                                                       "scalar_ms": match[0]["ms_per_iteration"],
                                                       "speedup": match[0]["ms_per_iteration"] / r["ms_per_iteration"]}
    res["simd_ablation"] = simd
    print("SIMD ablation:", json.dumps(simd, indent=1), flush=True)
    res["gpu_vs_cpu"] = gpu_vs_cpu()
    res["value_net"] = value_net_inference()
    res["latency_to_0.1pct_pot"] = latency()
    print("latency:", res["latency_to_0.1pct_pot"], flush=True)
    meta = [r for r in res["bench"] if r["bench"] == "meta"]
    res["hardware_counters"] = meta[0] if meta else {}
    save_json(NAME, "results.json", res)
    plots(res)
    return res


if __name__ == "__main__":
    run()
