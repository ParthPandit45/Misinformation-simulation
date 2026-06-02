// Centralized dashboard data/constants for reuse across views and utilities.

export const SUMMARY_FALLBACK = {
  p2p_gnutella: {
    gnn_rl: { median: 312, std: 28, supp: 68.4 },
    none: { median: 987, std: 45 },
    passive_degree: { median: 724, std: 38 },
    active_random: { median: 589, std: 52 },
    active_degree: { median: 441, std: 31 },
    episodes: 700,
    budget: 200,
  },
  ca_grqc: {
    gnn_rl: { median: 198, std: 19, supp: 71.2 },
    none: { median: 688, std: 41 },
    passive_degree: { median: 501, std: 35 },
    active_random: { median: 412, std: 48 },
    active_degree: { median: 289, std: 27 },
    episodes: 900,
    budget: 80,
  },
  facebook: {
    gnn_rl: { median: 521, std: 47, supp: 63.1 },
    none: { median: 1412, std: 88 },
    passive_degree: { median: 1089, std: 72 },
    active_random: { median: 891, std: 63 },
    active_degree: { median: 712, std: 55 },
    episodes: 1400,
    budget: 80,
  },
};

export const GRAPH_META = {
  p2p_gnutella: { nodes: 6301, edges: 20777, avgDeg: 6.6, color: "#378ADD", label: "P2P Gnutella", short: "Gnutella" },
  ca_grqc: { nodes: 5242, edges: 14496, avgDeg: 5.5, color: "#1D9E75", label: "CA-GrQc Collab", short: "GrQc" },
  facebook: { nodes: 4039, edges: 88234, avgDeg: 43.7, color: "#D4537E", label: "Facebook Social", short: "Facebook" },
};

export const STRATEGIES = {
  none: { label: "No Intervention", color: "#E24B4A", dash: [0, 0] },
  passive_degree: { label: "Passive Degree (t=5)", color: "#EF9F27", dash: [6, 3] },
  active_random: { label: "Active Random Cure", color: "#97C459", dash: [4, 4] },
  active_degree: { label: "Active Degree Cure", color: "#7F77DD", dash: [8, 3] },
  gnn_rl: { label: "GNN+RL (ours)", color: "#1D9E75", dash: [0, 0] },
};

export const ACTIONS = {
  0: { label: "Wait", color: "#888780" },
  1: { label: "Mute Small", color: "#378ADD" },
  2: { label: "Mute Large", color: "#185FA5" },
  3: { label: "Cure Infected", color: "#E24B4A" },
  4: { label: "Smart Burst", color: "#EF9F27" },
};

export const SUMMARY = JSON.parse(JSON.stringify(SUMMARY_FALLBACK));

function normalizeSummaryGraph(entry) {
  if (!entry || typeof entry !== "object") return null;

  const baselines = entry.baselines && typeof entry.baselines === "object" ? entry.baselines : {};
  const normalized = {};

  normalized.gnn_rl = {
    median: Number(entry.gnn_rl?.median ?? 0),
    std: Number(entry.gnn_rl?.std ?? 0),
    supp: Number(entry.gnn_rl?.supp ?? entry.gnn_rl?.suppression_pct ?? 0),
  };

  ["none", "passive_degree", "active_random", "active_degree"].forEach((strategy) => {
    const strategyEntry = entry[strategy];
    normalized[strategy] = {
      median: Number(strategyEntry?.median ?? baselines[strategy] ?? 0),
      std: Number(strategyEntry?.std ?? 0),
    };
  });

  normalized.episodes = Number(entry.episodes ?? 0);
  normalized.budget = Number(entry.budget ?? entry.budget_per_step ?? 0);
  return normalized;
}

export function applySummaryOverride(payload) {
  if (!payload || typeof payload !== "object") return false;
  let updated = false;
  Object.keys(payload).forEach((graphKey) => {
    const normalized = normalizeSummaryGraph(payload[graphKey]);
    if (!normalized) return;
    SUMMARY[graphKey] = normalized;
    updated = true;
  });
  return updated;
}

export async function loadSummaryOverride(weightsPath = `${import.meta.env.BASE_URL}weights/results.json`) {
  try {
    // Load default results JSON
    const response = await fetch(weightsPath, { cache: "no-store" });
    const payload = response.ok ? await response.json() : {};
    // Load custom results if present
    const customPath = `${import.meta.env.BASE_URL}weights/custom_results.json`;
    let customPayload = {};
    try {
      const customResp = await fetch(customPath, { cache: "no-store" });
      if (customResp.ok) customPayload = await customResp.json();
    } catch (e) {}
    // Load fallback summary_results.json from project root (served via copy script)
    let fallbackPayload = {};
    try {
      const fallbackResp = await fetch(`${import.meta.env.BASE_URL}results/summary_results.json`, { cache: "no-store" });
      if (fallbackResp.ok) fallbackPayload = await fallbackResp.json();
    } catch (e) {}
    const merged = { ...payload, ...customPayload, ...fallbackPayload };
    return applySummaryOverride(merged);
  } catch (e) {
    return false;
  }
}
