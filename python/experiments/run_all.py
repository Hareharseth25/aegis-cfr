"""Run every AEGIS experiment sequentially (no concurrent load, so timings are
clean), record the environment, then regenerate docs/RESULTS.md.

    .venv/bin/python python/experiments/run_all.py            # everything
    .venv/bin/python python/experiments/run_all.py exp03 exp07 # a subset
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import environment, save_json  # noqa: E402

ORDER = ["exp01_kuhn_equilibrium", "exp02_leduc_comparison", "exp03_cfrplus_vs_dcfr", "exp04_river_exact",
         "exp05_action_abstraction", "exp06_adaptive_sampling", "exp07_safe_resolving", "exp08_value_network",
         "exp09_performance", "exp10_refinement", "exp11_learned_allocation", "exp12_budget_allocation",
         "exp13_winrate_vs_exploitability"]


def main(argv):
    wanted = [e for e in ORDER if not argv or any(e.startswith(a) for a in argv)]
    log = {"environment": environment(), "runs": {}}
    for name in wanted:
        print(f"\n=== {name} ===", flush=True)
        t = time.perf_counter()
        try:
            importlib.import_module(f"experiments.{name}").run()
            log["runs"][name] = {"ok": True, "seconds": time.perf_counter() - t}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log["runs"][name] = {"ok": False, "error": repr(e), "seconds": time.perf_counter() - t}
        print(f"=== {name}: {log['runs'][name]['seconds']:.0f}s ===", flush=True)
    save_json("", "run_log.json", log)
    from experiments import report
    report.main()


if __name__ == "__main__":
    main(sys.argv[1:])
