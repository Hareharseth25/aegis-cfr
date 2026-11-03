// AEGIS Solver Lab: hash router over the views.

import { mountAdaptive } from "./adaptive_visualizer/adaptive";
import { mountConvergenceLab } from "./convergence_lab/lab";
import { mountRangeExplorer } from "./range_explorer/explorer";
import { mountResults } from "./results";
import { clear, h } from "./shared/dom";
import { mountProfiler } from "./solver_profiler/profiler";
import { mountSolverLab } from "./solver_lab";
import { mountMicroscope } from "./subgame_microscope/microscope";

const VIEWS: [string, string, (el: HTMLElement) => void][] = [
  ["lab", "Solver Lab", mountSolverLab],
  ["ranges", "Range explorer", mountRangeExplorer],
  ["convergence", "Convergence lab", mountConvergenceLab],
  ["microscope", "Subgame microscope", mountMicroscope],
  ["adaptive", "Adaptive computation", mountAdaptive],
  ["profiler", "Profiler", mountProfiler],
  ["results", "Results", mountResults],
];

const view = document.getElementById("view") as HTMLElement;
const tabs = document.getElementById("tabs") as HTMLElement;

function route(): void {
  const key = location.hash.replace("#", "") || "lab";
  const entry = VIEWS.find(([k]) => k === key) ?? VIEWS[0];
  clear(tabs);
  for (const [k, label] of VIEWS) tabs.append(h("a", { href: `#${k}`, class: k === entry[0] ? "active" : "" }, label));
  clear(view);
  entry[2](view);
}

window.addEventListener("hashchange", route);
route();

// Theme toggle: explicit choice wins over the OS setting in both directions.
const toggle = document.getElementById("theme-toggle") as HTMLButtonElement;
const saved = (() => { try { return localStorage.getItem("aegis-theme"); } catch { return null; } })();
if (saved) document.documentElement.dataset.theme = saved;
toggle.addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("aegis-theme", next); } catch { /* storage unavailable */ }
});
