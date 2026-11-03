// JSON API client for the AEGIS server (python -m aegis.server).

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok) throw new Error((data as { error?: string }).error ?? `HTTP ${res.status}`);
  return data as T;
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body: unknown) => request<T>("POST", path, body),
};

export interface ConvergencePoint {
  iteration: number;
  seconds: number;
  exploitability_pct_pot?: number;
  exploitability?: number;
  nash_conv?: number;
  nodes?: number;
  avg_positive_regret?: number;
}

export interface NodeStrategy {
  node: number;
  player: number;
  path: string;
  actions: string[];
  combos: string[];
  strategy: number[][];
  reach: number[];
  aggregate_frequency: number[];
  class_grid: { name: string; mass: number; mix: number[] | null }[];
  action_ev?: number[][];
  action_ev_from_here?: number[][];
  child_actions?: string[];
  children?: number[];
  type?: string;
  pot?: number;
}

export interface TreeNode {
  id: number;
  type: string;
  player: number;
  parent: number;
  children: number[];
  action: string;
  pot: number;
  commit: number[];
  street_bet: number[];
  street: number;
  board: string[];
}

export interface HeroView {
  node: number;
  path: string;
  hand: string;
  player: number;
  pot: number;
  to_act: number;
  actions?: string[];
  strategy?: number[];
  action_ev?: (number | null)[];
  action_ev_from_here?: (number | null)[];
  ev?: number | null;
  ev_from_here?: number | null;
  committed: number;
  range_frequency?: number[];
  opponent_range_blocked_fraction: number;
  regret_trace: { iteration: number; avg_positive_regret: number | null }[];
}

export interface SolveResult {
  id: string;
  board: string[];
  pot: number;
  effective_stack: number;
  exploitability: { exploitability_pct_pot: number; nash_conv: number; br_value: number[]; policy_value: number[] };
  convergence: ConvergencePoint[];
  regret_trace: { iteration: number; mean: number; node: number }[];
  solve_seconds: number;
  total_seconds: number;
  num_nodes: number;
  tree_complete: boolean;
  hands: number[];
  memory_mb: number;
  tree: TreeNode[];
  root: NodeStrategy;
  hero?: HeroView;
  hero_error?: string;
  note: string;
}
