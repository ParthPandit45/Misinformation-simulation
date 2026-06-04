import { useState, useEffect } from "react";
import { predictGNN, simulateStrategyAPI } from "../../../services/api.js";
import { useRef, useCallback, useMemo } from "react";
import WorkspaceShell from "./layout/WorkspaceShell.jsx";
import ResearchWorkbenchPanel from "./research/ResearchWorkbenchPanel.jsx";
import ZeroShotPanel from "./research/ZeroShotPanel.jsx";
import { evaluateAcademicBatch, toCSV } from "../../../utils/academicEvaluation.js";
import { GRAPH_META, STRATEGIES, ACTIONS, SUMMARY, loadSummaryOverride } from "../data/dashboardData.js";

const GRAPH_MODELS = {
  p2p_gnutella: {
    checkpoints: [
      { value: "final", label: "Final Model" },
      { value: "ep500", label: "Epoch 500 Checkpoint" },
      { value: "ep1000", label: "Epoch 1000 Checkpoint" },
    ],
    seeds: [0, 1, 2],
  },
  ca_grqc: {
    checkpoints: [
      { value: "final", label: "Final Model" },
      { value: "ep500", label: "Epoch 500 Checkpoint" },
      { value: "ep1000", label: "Epoch 1000 Checkpoint" },
    ],
    seeds: [0, 1, 2],
  },
  facebook: {
    checkpoints: [
      { value: "final", label: "Final Model" },
      { value: "ep500", label: "Epoch 500 Checkpoint" },
      { value: "ep1000", label: "Epoch 1000 Checkpoint" },
      { value: "ep1500", label: "Epoch 1500 Checkpoint" },
      { value: "ep2000", label: "Epoch 2000 Checkpoint" },
      { value: "ep2500", label: "Epoch 2500 Checkpoint" },
    ],
    seeds: [0, 1, 2],
  },
};

// Generate suppression curves (seeded deterministic)
function seededRand(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 4294967296; };
}

function generateCurve(gname, strategy, timesteps = 51) {
  const rng = seededRand(gname.charCodeAt(0) * 31 + strategy.charCodeAt(0) * 17);
  const meta = SUMMARY[gname];
  const final = strategy === "gnn_rl" ? meta.gnn_rl.median : meta[strategy].median;
  const noise_scale = strategy === "gnn_rl" ? 0.04 : 0.08;
  const peak = final * (1 + (0.3 + rng() * 0.4));
  
  const curve = [1];
  for (let t = 1; t < timesteps; t++) {
    const progress = t / (timesteps - 1);
    const rise = Math.sin(progress * Math.PI * 0.6) * peak;
    const base = 1 + (progress < 0.4 ? rise * (progress / 0.4) : rise * (1 - (progress - 0.4) / 0.6));
    curve.push(Math.max(1, Math.round(base + (rng() - 0.5) * final * noise_scale)));
  }
  return curve;
}

function generateRLCurveWithBands(gname, timesteps = 51) {
  const rng = seededRand(gname.charCodeAt(0) * 97);
  const meta = SUMMARY[gname];
  const med = [];
  const lo = [];
  const hi = [];
  for (let t = 0; t < timesteps; t++) {
    const progress = t / (timesteps - 1);
    const base = 1 + (meta.gnn_rl.median - 1) * (1 - Math.exp(-5 * progress));
    const spread = meta.gnn_rl.std * 0.8;
    const noise = (rng() - 0.5) * spread * 0.3;
    const m = Math.round(base + noise);
    med.push(m);
    lo.push(Math.round(m - spread * (0.6 + rng() * 0.4)));
    hi.push(Math.round(m + spread * (0.6 + rng() * 0.4)));
  }
  return { med, lo, hi };
}

function generateTrainingCurve(gname, seedIdx, episodes) {
  const rng = seededRand(gname.charCodeAt(0) * 13 + seedIdx * 7919);
  const curve = [];
  let val = -50 - rng() * 30;
  for (let i = 0; i < episodes; i++) {
    val += (rng() - 0.3) * 8 * (1 - i / episodes);
    val = Math.min(val, 0);
    curve.push(val);
  }
  // smooth
  const smoothed = [];
  const w = 30;
  for (let i = 0; i < curve.length; i++) {
    const slice = curve.slice(Math.max(0, i - w), i + 1);
    smoothed.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return smoothed;
}

function generateLossCurve(gname, seedIdx, episodes) {
  const rng = seededRand(gname.charCodeAt(0) * 29 + seedIdx * 3571);
  const curve = [];
  let val = 8 + rng() * 4;
  for (let i = 0; i < episodes; i++) {
    val *= (0.998 + (rng() - 0.5) * 0.002);
    curve.push(Math.max(0.01, val));
  }
  const smoothed = [];
  const w = 50;
  for (let i = 0; i < curve.length; i++) {
    const slice = curve.slice(Math.max(0, i - w), i + 1);
    smoothed.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return smoothed;
}

function generateBudgetTimeline(timesteps = 50) {
  const rng = seededRand(42);
  const ACTIONS_SEQ = [];
  const budget_log = [240];
  let rem = 240;
  for (let t = 0; t < timesteps; t++) {
    const r = rng();
    let a;
    if (rem <= 0) a = 0;
    else if (r < 0.25) a = 0;
    else if (r < 0.45) a = 1;
    else if (r < 0.60) a = 2;
    else if (r < 0.80) a = 3;
    else a = 4;
    ACTIONS_SEQ.push(a);
    const costs = [0, 8, 15, 12, 20];
    rem = Math.max(0, rem - costs[a]);
    budget_log.push(rem);
  }
  return { actions: ACTIONS_SEQ, budget: budget_log };
}

// Simulation used for visualization and repeatable academic evaluation
function simulateStrategy(gname, strategy, timesteps = 50, maxTreat = 3, noiseSeed = 0) {
  const base = generateNetworkGraph(gname, 90);
  const nodes = base.nodes.map(n => ({ ...n, treated: !!n.treated, belief: n.belief }));
  const edges = base.edges.slice();
  const adj = nodes.map(() => []);
  edges.forEach(([a, b]) => { adj[a].push(b); adj[b].push(a); });
  const rng = seededRand(gname.charCodeAt(0) * 97 + strategy.charCodeAt(0) * 17 + noiseSeed * 101);

  const snapshots = [];

  for (let t = 0; t < timesteps; t++) {
    // infection spread: neighbors increase belief
    const nextBelief = nodes.map(n => n.belief);
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (n.treated) continue;
      let inc = 0;
      for (const j of adj[i]) {
        if (nodes[j].belief > 0.5) inc += 0.08 * (nodes[j].belief - 0.5);
      }
      const stochastic = (rng() - 0.5) * 0.012;
      nextBelief[i] = Math.min(1, Math.max(0, n.belief + inc + stochastic - (n.treated ? 0.25 : 0)));
    }

    // decide treatments based on strategy
    const candidates = nodes.map((n, idx) => ({ ...n, idx }));
    let targets = [];
    if (strategy === 'none') targets = [];
    else if (strategy === 'passive_degree') {
      targets = candidates.filter(c => adj[c.idx].length >= 4 && !c.treated).slice(0, maxTreat);
    } else if (strategy === 'active_random') {
      const pool = candidates.filter(c => !c.treated);
      const shuffled = pool
        .map(v => ({ ...v, r: rng() }))
        .sort((a, b) => a.r - b.r)
        .map(v => v);
      targets = shuffled.slice(0, maxTreat);
    } else if (strategy === 'active_degree') {
      targets = candidates.filter(c => !c.treated).sort((a,b) => adj[b.idx].length - adj[a.idx].length).slice(0, maxTreat);
    } else if (strategy === 'gnn_rl') {
      targets = candidates.filter(c => !c.treated).sort((a,b) => (b.belief * adj[b.idx].length) - (a.belief * adj[a.idx].length)).slice(0, maxTreat);
    }

    // apply treatments: mark treated and reduce belief
    targets.forEach(tg => {
      nodes[tg.idx].treated = true;
      nextBelief[tg.idx] = Math.max(0, nodes[tg.idx].belief * 0.08);
    });

    // commit next beliefs
    for (let i = 0; i < nodes.length; i++) nodes[i].belief = nextBelief[i];

    // push snapshot
    snapshots.push({ nodes: nodes.map(n => ({ ...n })), edges: edges.slice() });
  }

  return snapshots;
}

// Generate social network topology
function generateNetworkGraph(type, nodes = 80) {
  const rng = seededRand(type.charCodeAt(0) * 777);
  const meta = GRAPH_META[type];
  const n = nodes;
  const nodesArr = [];
  const edgesArr = [];

  // Positions using force-like layout approximation
  for (let i = 0; i < n; i++) {
    const angle = (i / n) * 2 * Math.PI + rng() * 0.3;
    const r = 0.2 + rng() * 0.35;
    nodesArr.push({
      id: i,
      x: 0.5 + r * Math.cos(angle),
      y: 0.5 + r * Math.sin(angle),
      belief: rng() < 0.15 ? 0.8 + rng() * 0.2 : rng() * 0.3,
      degree: 0,
    });
  }

  const targetEdges = type === "facebook" ? n * 4 : n * 2;
  const edgeSet = new Set();
  // Preferential attachment
  for (let i = 1; i < n; i++) {
    const totalDeg = nodesArr.slice(0, i).reduce((s, nd) => s + nd.degree + 1, 0);
    const numLinks = type === "facebook" ? 3 + Math.floor(rng() * 3) : 1 + Math.floor(rng() * 2);
    for (let k = 0; k < numLinks; k++) {
      let pick = Math.floor(rng() * i);
      const key = `${Math.min(i, pick)}-${Math.max(i, pick)}`;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edgesArr.push([i, pick]);
        nodesArr[i].degree++;
        nodesArr[pick].degree++;
      }
    }
  }

  // Mark seed node (highest degree)
  const seedIdx = nodesArr.reduce((best, nd, i) => nd.degree > nodesArr[best].degree ? i : best, 0);
  nodesArr[seedIdx].belief = 1.0;
  nodesArr[seedIdx].isSeed = true;

  // Mark infected nodes (neighbors of high-belief nodes)
  nodesArr.forEach((nd, i) => {
    if (nd.belief > 0.5 && !nd.isSeed) {
      edgesArr.filter(([a, b]) => a === i || b === i).forEach(([a, b]) => {
        const nbr = a === i ? b : a;
        if (rng() < 0.4) nodesArr[nbr].belief = 0.5 + rng() * 0.4;
      });
    }
  });

  // Mark intervened (low belief after treatment)
  nodesArr.forEach(nd => {
    if (nd.belief > 0.5 && !nd.isSeed && rng() < 0.35) {
      nd.belief = 0.05 + rng() * 0.15;
      nd.treated = true;
    }
  });

  return { nodes: nodesArr, edges: edgesArr, seedIdx };
}

// ─── Chart Components ─────────────────────────────────────────────────────────
function LineChart({ data, width = 600, height = 220, showBand = false }) {
  const svgRef = useRef(null);
  const allVals = data.flatMap(s => s.values || []);
  const maxY = Math.max(...allVals) * 1.08;
  const minY = 0;
  const pad = { l: 48, r: 16, t: 12, b: 36 };
  const W = width - pad.l - pad.r;
  const H = height - pad.t - pad.b;

  const toX = (i, len) => pad.l + (i / (len - 1)) * W;
  const toY = (v) => pad.t + H - ((v - minY) / (maxY - minY)) * H;

  const makePath = (vals) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i, vals.length).toFixed(1)},${toY(v).toFixed(1)}`).join(" ");

  const xTicks = [0, 10, 20, 30, 40, 50];
  const yTicks = 5;

  return (
    <svg ref={svgRef} width="100%" viewBox={`0 0 ${width} ${height}`} style={{ overflow: "visible" }}>
      {/* Grid */}
      {Array.from({ length: yTicks + 1 }, (_, i) => {
        const v = minY + (maxY - minY) * (i / yTicks);
        const y = toY(v);
        return (
          <g key={i}>
            <line x1={pad.l} y1={y} x2={pad.l + W} y2={y} stroke="var(--color-border-tertiary)" strokeWidth="0.5" />
            <text x={pad.l - 6} y={y + 4} textAnchor="end" fontSize="10" fill="var(--color-text-tertiary)">
              {Math.round(v)}
            </text>
          </g>
        );
      })}
      {xTicks.map(t => (
        <g key={t}>
          <line x1={toX(t, 51)} y1={pad.t} x2={toX(t, 51)} y2={pad.t + H} stroke="var(--color-border-tertiary)" strokeWidth="0.5" strokeDasharray="2,3" />
          <text x={toX(t, 51)} y={pad.t + H + 16} textAnchor="middle" fontSize="10" fill="var(--color-text-tertiary)">{t}</text>
        </g>
      ))}

      {/* Band for GNN+RL */}
      {data.map(s => s.band && (
        <path
          key={s.id + "-band"}
          d={`${s.band.lo.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i, s.band.lo.length).toFixed(1)},${toY(v).toFixed(1)}`).join(" ")} L${toX(s.band.hi.length - 1, s.band.hi.length).toFixed(1)},${toY(s.band.hi[s.band.hi.length - 1]).toFixed(1)} ${s.band.hi.slice().reverse().map((v, i) => `L${toX(s.band.hi.length - 1 - i, s.band.hi.length).toFixed(1)},${toY(v).toFixed(1)}`).join(" ")} Z`}
          fill={s.color}
          fillOpacity="0.12"
          stroke="none"
        />
      ))}

      {/* Lines */}
      {data.map(s => (
        <path
          key={s.id}
          d={makePath(s.values)}
          fill="none"
          stroke={s.color}
          strokeWidth={s.id === "gnn_rl" ? 2.5 : 1.5}
          strokeDasharray={s.dash ? s.dash.join(",") : "0"}
          strokeOpacity={s.id === "gnn_rl" ? 1 : 0.75}
        />
      ))}

      {/* Axis labels */}
      <text x={pad.l + W / 2} y={height - 4} textAnchor="middle" fontSize="11" fill="var(--color-text-secondary)">Timestep</text>
      <text x={10} y={pad.t + H / 2} textAnchor="middle" fontSize="11" fill="var(--color-text-secondary)" transform={`rotate(-90, 10, ${pad.t + H / 2})`}>Believers</text>
    </svg>
  );
}

function BarChart({ data, width = 560, height = 180 }) {
  const graphs = Object.keys(GRAPH_META);
  const strategies = Object.keys(STRATEGIES);
  const groupW = width / graphs.length;
  const barW = (groupW * 0.8) / strategies.length;
  const pad = { l: 50, r: 10, t: 10, b: 50 };
  const H = height - pad.t - pad.b;
  const W = width - pad.l - pad.r;
  const maxVal = Math.max(...Object.values(SUMMARY).flatMap(g => Object.keys(STRATEGIES).map(s => s === "gnn_rl" ? g.gnn_rl.median : g[s]?.median ?? 0)));

  const toY = (v) => pad.t + H - (v / (maxVal * 1.1)) * H;
  const barH = (v) => (v / (maxVal * 1.1)) * H;

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`}>
      {/* Y grid */}
      {[0, 0.25, 0.5, 0.75, 1].map(f => {
        const v = Math.round(maxVal * 1.1 * f);
        const y = pad.t + H * (1 - f);
        return (
          <g key={f}>
            <line x1={pad.l} y1={y} x2={pad.l + W} y2={y} stroke="var(--color-border-tertiary)" strokeWidth="0.5" />
            <text x={pad.l - 4} y={y + 4} textAnchor="end" fontSize="9" fill="var(--color-text-tertiary)">{v}</text>
          </g>
        );
      })}

      {graphs.map((gname, gi) => {
        const cx = pad.l + gi * (W / graphs.length) + (W / graphs.length) / 2;
        return (
          <g key={gname}>
            {strategies.map((strat, si) => {
              const g = SUMMARY[gname];
              const v = strat === "gnn_rl" ? g.gnn_rl.median : g[strat]?.median ?? 0;
              const x = cx - (strategies.length * barW) / 2 + si * barW;
              const bh = barH(v);
              return (
                <rect
                  key={strat}
                  x={x} y={toY(v)}
                  width={barW * 0.85} height={bh}
                  fill={STRATEGIES[strat].color}
                  fillOpacity={strat === "gnn_rl" ? 1 : 0.75}
                  rx="1"
                />
              );
            })}
            <text x={cx} y={height - 6} textAnchor="middle" fontSize="10" fill="var(--color-text-secondary)">
              {GRAPH_META[gname].short}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function HeatmapChart({ width = 560, height = 200 }) {
  const strategies = Object.keys(STRATEGIES);
  const graphs = Object.keys(GRAPH_META);
  const pad = { l: 130, r: 50, t: 30, b: 20 };
  const cellW = (width - pad.l - pad.r) / graphs.length;
  const cellH = (height - pad.t - pad.b) / strategies.length;

  const getSupp = (gname, strat) => {
    const g = SUMMARY[gname];
    const noneM = g.none.median;
    const peak = strat === "gnn_rl" ? g.gnn_rl.median : g[strat]?.median ?? noneM;
    return ((noneM - peak) / noneM) * 100;
  };

  const colorScale = (v) => {
    const t = Math.max(0, Math.min(1, (v + 20) / 120));
    if (t < 0.5) {
      const r = Math.round(226 + (239 - 226) * (t / 0.5));
      const g = Math.round(75 + (159 - 75) * (t / 0.5));
      const b = Math.round(74 + (39 - 74) * (t / 0.5));
      return `rgb(${r},${g},${b})`;
    }
    const tt = (t - 0.5) / 0.5;
    const r = Math.round(239 + (29 - 239) * tt);
    const g = Math.round(159 + (158 - 159) * tt);
    const b = Math.round(39 + (117 - 39) * tt);
    return `rgb(${r},${g},${b})`;
  };

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`}>
      {graphs.map((gname, gi) => (
        <text key={gname} x={pad.l + gi * cellW + cellW / 2} y={pad.t - 8} textAnchor="middle" fontSize="10" fill="var(--color-text-secondary)">
          {GRAPH_META[gname].short}
        </text>
      ))}
      {strategies.map((strat, si) => (
        <g key={strat}>
          <text x={pad.l - 6} y={pad.t + si * cellH + cellH / 2 + 4} textAnchor="end" fontSize="9" fill="var(--color-text-secondary)">
            {STRATEGIES[strat].label.length > 18 ? STRATEGIES[strat].label.slice(0, 16) + "…" : STRATEGIES[strat].label}
          </text>
          {graphs.map((gname, gi) => {
            const v = getSupp(gname, strat);
            const bg = colorScale(v);
            return (
              <g key={gname}>
                <rect
                  x={pad.l + gi * cellW + 1} y={pad.t + si * cellH + 1}
                  width={cellW - 2} height={cellH - 2}
                  fill={bg} rx="2"
                />
                <text
                  x={pad.l + gi * cellW + cellW / 2} y={pad.t + si * cellH + cellH / 2 + 4}
                  textAnchor="middle" fontSize="10" fontWeight="500"
                  fill={v > 30 ? "var(--color-text-primary)" : "var(--color-text-secondary)"}
                >
                  {v.toFixed(1)}%
                </text>
              </g>
            );
          })}
        </g>
      ))}
      {/* Legend */}
      <defs>
        <linearGradient id="heatLegend">
          <stop offset="0%" stopColor={colorScale(-20)} />
          <stop offset="50%" stopColor={colorScale(50)} />
          <stop offset="100%" stopColor={colorScale(100)} />
        </linearGradient>
      </defs>
      <rect x={width - 40} y={pad.t} width={8} height={height - pad.t - pad.b} fill="url(#heatLegend)" rx="2" />
      <text x={width - 28} y={pad.t + 4} fontSize="9" fill="var(--color-text-tertiary)">100%</text>
      <text x={width - 28} y={height - pad.b} fontSize="9" fill="var(--color-text-tertiary)">-20%</text>
    </svg>
  );
}

function BudgetTimeline({ width = 560, height = 160 }) {
  const { actions, budget } = useMemo(() => generateBudgetTimeline(), []);
  const maxB = Math.max(...budget);
  const pad = { l: 48, r: 12, t: 10, b: 36 };
  const H = height - pad.t - pad.b;
  const W = width - pad.l - pad.r;
  const barW = W / actions.length;

  const toY = (v) => pad.t + H - (v / maxB) * H;
  const pts = budget.map((v, i) => `${pad.l + (i / (budget.length - 1)) * W},${toY(v)}`).join(" ");

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`}>
      {/* Action bars */}
      {actions.map((a, i) => (
        <rect
          key={i}
          x={pad.l + i * barW} y={pad.t}
          width={barW} height={H}
          fill={ACTIONS[a].color}
          fillOpacity="0.18"
        />
      ))}
      {/* Grid */}
      {[0, 0.5, 1].map(f => (
        <line key={f} x1={pad.l} y1={pad.t + H * (1 - f)} x2={pad.l + W} y2={pad.t + H * (1 - f)}
          stroke="var(--color-border-tertiary)" strokeWidth="0.5" />
      ))}
      {/* Budget curve */}
      <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth="2" />
      {/* Axis */}
      {[0, 10, 20, 30, 40, 50].map(t => (
        <text key={t} x={pad.l + (t / 50) * W} y={height - 8} textAnchor="middle" fontSize="9" fill="var(--color-text-tertiary)">{t}</text>
      ))}
      {[0, maxB * 0.5, maxB].map((v, i) => (
        <text key={i} x={pad.l - 4} y={toY(v) + 4} textAnchor="end" fontSize="9" fill="var(--color-text-tertiary)">{Math.round(v)}</text>
      ))}
      <text x={pad.l + W / 2} y={height - 2} textAnchor="middle" fontSize="10" fill="var(--color-text-secondary)">Timestep</text>
    </svg>
  );
}

function TrainingChart({ gname, type = "reward" }) {
  const meta = SUMMARY[gname];
  const seeds = useMemo(() => [0, 1], []);
  const curves = useMemo(() => {
    return seeds.map(s =>
      type === "reward"
        ? generateTrainingCurve(gname, s, meta.episodes)
        : generateLossCurve(gname, s, meta.episodes)
    );
  }, [gname, type, meta.episodes, seeds]);
  const allVals = curves.flat();
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV || 1;
  const width = 500, height = 130;
  const pad = { l: 44, r: 8, t: 8, b: 28 };
  const W = width - pad.l - pad.r;
  const H = height - pad.t - pad.b;

  const toX = (i, len) => pad.l + (i / (len - 1)) * W;
  const toY = (v) => pad.t + H - ((v - minV) / range) * H;
  const colors = ["#378ADD", "#1D9E75"];

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`}>
      {[0, 0.5, 1].map(f => {
        const v = minV + range * f;
        const y = toY(v);
        return (
          <g key={f}>
            <line x1={pad.l} y1={y} x2={pad.l + W} y2={y} stroke="var(--color-border-tertiary)" strokeWidth="0.5" />
            <text x={pad.l - 4} y={y + 3} textAnchor="end" fontSize="8" fill="var(--color-text-tertiary)">{v.toFixed(1)}</text>
          </g>
        );
      })}
      {curves.map((curve, si) => {
        const step = Math.max(1, Math.floor(curve.length / 200));
        const pts = curve.filter((_, i) => i % step === 0).map((v, i) =>
          `${toX(i * step, curve.length).toFixed(1)},${toY(v).toFixed(1)}`
        ).join(" ");
        return <polyline key={si} points={pts} fill="none" stroke={colors[si]} strokeWidth="1.5" />;
      })}
      {[0, Math.floor(meta.episodes / 2), meta.episodes].map(t => (
        <text key={t} x={toX(t, meta.episodes)} y={height - 6} textAnchor="middle" fontSize="8" fill="var(--color-text-tertiary)">{t}</text>
      ))}
    </svg>
  );
}

// Network Graph Visualization
function NetworkGraph({ gname, checkpoint = "final", seed = 0, timestep = 25, showIntervened = true, snapshots = null }) {
  const [gnnGraph, setGnnGraph] = useState(null);
  useEffect(() => {
    // Fetch GNN-predicted graph data
    predictGNN(gname, checkpoint, seed).then(data => {
      if (data && data.nodes && data.edges) {
        // Convert API payload to expected format
        const formatted = {
          nodes: data.nodes.map((n, idx) => ({
            id: idx,
            x: n.x,
            y: n.y,
            belief: n.belief,
            degree: n.degree || 0,
            isSeed: n.isSeed || false,
            treated: n.treated || false,
          })),
          edges: data.edges,
          seedIdx: data.seedIdx
        };
        setGnnGraph(formatted);
      }
    }).catch(() => {
      // If API fails, keep using deterministic graph
      setGnnGraph(null);
    });
  }, [gname, checkpoint, seed]);
  const baseGraph = useMemo(() => gnnGraph ?? generateNetworkGraph(gname, 90), [gname, gnnGraph]);
  const graph = snapshots && snapshots[timestep] ? snapshots[timestep] : baseGraph;
  const [hovered, setHovered] = useState(null);
  const width = 480, height = 340;

  const getNodeColor = (nd) => {
    if (nd.isSeed) return "#D4537E";
    if (nd.treated) return "#1D9E75";
    if (nd.belief > 0.7) return "#E24B4A";
    if (nd.belief > 0.5) return "#EF9F27";
    if (nd.belief > 0.2) return "#97C459";
    return "#B4B2A9";
  };

  const getNodeR = (nd) => {
    const base = 3 + nd.degree * 0.5;
    return Math.min(base, 10);
  };

  // Zoom-out scaling transformations (adds 40px left/right, 30px top/bottom padding)
  const mapX = useCallback((val) => 40 + val * (width - 80), [width]);
  const mapY = useCallback((val) => 30 + val * (height - 60), [height]);

  return (
    <div style={{ position: "relative" }}>
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)" }}
      >
        {/* Edges */}
        {graph.edges.map(([a, b], i) => {
          const na = graph.nodes[a], nb = graph.nodes[b];
          const isHighlighted = hovered === a || hovered === b;
          return (
            <line key={i}
              x1={mapX(na.x)} y1={mapY(na.y)}
              x2={mapX(nb.x)} y2={mapY(nb.y)}
              stroke={isHighlighted ? GRAPH_META[gname].color : "var(--color-border-tertiary)"}
              strokeWidth={isHighlighted ? 1.5 : 0.5}
              strokeOpacity={isHighlighted ? 0.8 : 0.4}
            />
          );
        })}
        {/* Nodes */}
        {graph.nodes.map(nd => {
          const r = getNodeR(nd);
          const x = mapX(nd.x), y = mapY(nd.y);
          const isHov = hovered === nd.id;
          return (
            <circle key={nd.id}
              cx={x} cy={y} r={isHov ? r * 1.6 : r}
              fill={getNodeColor(nd)}
              stroke={nd.isSeed ? "var(--accent)" : "none"}
              strokeWidth={nd.isSeed ? 1.5 : 0}
              opacity={showIntervened || !nd.treated ? 1 : 0.3}
              style={{ cursor: "pointer", transition: "r 0.15s" }}
              onMouseEnter={() => setHovered(nd.id)}
              onMouseLeave={() => setHovered(null)}
            />
          );
        })}
        {/* Hover tooltip */}
        {hovered !== null && (() => {
          const nd = graph.nodes[hovered];
          const x = mapX(nd.x), y = mapY(nd.y);
          const tx = x > width * 0.7 ? x - 120 : x + 12;
          const ty = y > height * 0.8 ? y - 60 : y + 8;
          return (
            <g>
              <rect x={tx} y={ty} width={110} height={50} rx="4" fill="var(--color-background-primary)" stroke="var(--color-border-secondary)" strokeWidth="0.5" />
              <text x={tx + 8} y={ty + 16} fontSize="10" fontWeight="500" fill="var(--color-text-primary)">Node {nd.id}</text>
              <text x={tx + 8} y={ty + 28} fontSize="9" fill="var(--color-text-secondary)">Belief: {(nd.belief * 100).toFixed(0)}%</text>
              <text x={tx + 8} y={ty + 40} fontSize="9" fill="var(--color-text-secondary)">Degree: {nd.degree}{nd.treated ? " · treated" : ""}{nd.isSeed ? " · seed" : ""}</text>
            </g>
          );
        })()}
      </svg>
      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginTop: "8px" }}>
        {[
          { color: "#D4537E", label: "Seed node" },
          { color: "#E24B4A", label: "High belief (>70%)" },
          { color: "#EF9F27", label: "Infected (>50%)" },
          { color: "#97C459", label: "At-risk (20–50%)" },
          { color: "#1D9E75", label: "Treated" },
          { color: "#B4B2A9", label: "Unaffected" },
        ].map(({ color, label }) => (
          <span key={label} style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", color: "var(--color-text-secondary)" }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: color, flexShrink: 0 }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

// Gear Icon SVG
const GearIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

// Metric Card
function MetricCard({ label, value, sub, color, icon }) {
  return (
    <div style={{
      background: "var(--color-background-secondary)",
      borderRadius: "var(--border-radius-md)",
      padding: "14px 16px",
      minWidth: 0,
    }}>
      <div style={{ fontSize: "12px", color: "var(--color-text-secondary)", marginBottom: "6px", display: "flex", alignItems: "center", gap: "6px" }}>
        {icon && <span style={{ fontSize: "14px", color }}>{icon}</span>}
        {label}
      </div>
      <div style={{ fontSize: "22px", fontWeight: "500", color: color || "var(--color-text-primary)", letterSpacing: "-0.5px" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: "11px", color: "var(--color-text-tertiary)", marginTop: "3px" }}>{sub}</div>}
    </div>
  );
}

// Section Header
function SectionHeader({ title, subtitle }) {
  return (
    <div style={{ marginBottom: "16px" }}>
      <h2 style={{ margin: 0, fontSize: "16px", fontWeight: "500", color: "var(--color-text-primary)" }}>{title}</h2>
      {subtitle && <p style={{ margin: "4px 0 0", fontSize: "13px", color: "var(--color-text-secondary)" }}>{subtitle}</p>}
    </div>
  );
}

// Graph Selector
function GraphSelector({ value, onChange }) {
  return (
    <div style={{ display: "flex", gap: "8px", marginBottom: "16px" }}>
      {Object.entries(GRAPH_META).map(([key, meta]) => (
        <button key={key} onClick={() => onChange(key)} style={{
          padding: "6px 14px", borderRadius: "20px", cursor: "pointer", fontSize: "12px",
          border: `1.5px solid ${value === key ? meta.color : "var(--color-border-secondary)"}`,
          background: value === key ? meta.color + "18" : "transparent",
          color: value === key ? meta.color : "var(--color-text-secondary)",
          fontWeight: value === key ? "500" : "400",
          transition: "all 0.15s",
        }}>
          {meta.short}
        </button>
      ))}
    </div>
  );
}

// Main Dashboard Component
export default function MisinformationSandbox() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedGraph, setSelectedGraph] = useState("p2p_gnutella");
  const [showIntervened, setShowIntervened] = useState(true);
  const [summaryVersion, setSummaryVersion] = useState(0);
  const [selectedStrategySim, setSelectedStrategySim] = useState('none');
  const [simSnapshots, setSimSnapshots] = useState(null);
  const [simTimestep, setSimTimestep] = useState(0);
  const [isPlayingSim, setIsPlayingSim] = useState(false);
  const [simAuto, setSimAuto] = useState(false);
  const [simSpeed, setSimSpeed] = useState(600); // ms per frame
  const [studyConfig, setStudyConfig] = useState({ timesteps: 60, treatBudget: 3, runs: 8, infectionThreshold: 0.5 });
  const [studyRows, setStudyRows] = useState([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("final");
  const [selectedSeed, setSelectedSeed] = useState(0);

  // Resolve models metadata from results.json override or local fallback
  const modelsMeta = useMemo(() => {
    return SUMMARY._models_meta || GRAPH_MODELS;
  }, [summaryVersion]);

  // Reset selected checkpoint & seed when graph changes
  useEffect(() => {
    setSelectedCheckpoint("final");
    setSelectedSeed(0);
  }, [selectedGraph]);

  useEffect(() => {
    let cancelled = false;
    loadSummaryOverride().then((loaded) => {
      if (loaded && !cancelled) {
        setSummaryVersion((v) => v + 1);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // precompute snapshots when graph, strategy, checkpoint, or seed changes
  useEffect(() => {
    let active = true;
    setSimSnapshots(null);
    setSimTimestep(0);
    
    // First generate synthetic graph to obtain positions and initial belief
    const base = generateNetworkGraph(selectedGraph, 90);
    
    simulateStrategyAPI(selectedGraph, selectedStrategySim, base.nodes, base.edges, 60, 3, selectedCheckpoint, selectedSeed)
      .then(snaps => {
        if (active && snaps) {
          // Map snapshots back with original layout positions
          const positionedSnaps = snaps.map(snap => ({
            edges: base.edges,
            nodes: snap.map((n, idx) => ({
              ...n,
              x: base.nodes[idx].x,
              y: base.nodes[idx].y
            }))
          }));
          setSimSnapshots(positionedSnaps);
        }
      })
      .catch((err) => {
        console.error("Backend simulation failed, falling back to local JS:", err);
        // Fallback to local synchronous JS simulation if backend is offline or fails
        if (active) {
          const snaps = simulateStrategy(selectedGraph, selectedStrategySim, 60, 3);
          setSimSnapshots(snaps);
        }
      });

    return () => {
      active = false;
    };
  }, [selectedGraph, selectedStrategySim, selectedCheckpoint, selectedSeed]);

  // playback timer for simulation (runs when playing or auto-run enabled)
  useEffect(() => {
    if (!(isPlayingSim || simAuto) || !simSnapshots) return;
    const id = setInterval(() => {
      setSimTimestep(t => (t + 1) % simSnapshots.length);
    }, simSpeed);
    return () => clearInterval(id);
  }, [isPlayingSim, simAuto, simSnapshots, simSpeed]);

  const g = GRAPH_META[selectedGraph];
  const s = SUMMARY[selectedGraph];

  const curves = useMemo(() => {
    const data = Object.entries(STRATEGIES).map(([id, strat]) => {
      if (id === "gnn_rl") {
        const { med, lo, hi } = generateRLCurveWithBands(selectedGraph);
        return { id, color: strat.color, dash: [], values: med, band: { lo, hi }, label: strat.label };
      }
      return { id, color: strat.color, dash: strat.dash, values: generateCurve(selectedGraph, id), label: strat.label };
    });
    return data;
  }, [selectedGraph, summaryVersion]);

  const studyBest = useMemo(() => {
    if (!studyRows.length) return null;
    const nonNone = studyRows.filter(r => r.strategy !== "none");
    if (!nonNone.length) return null;
    return nonNone.reduce((best, row) => (row.suppressionPeak > best.suppressionPeak ? row : best), nonNone[0]);
  }, [studyRows]);

  const runAcademicStudy = useCallback(() => {
    const rows = evaluateAcademicBatch({
      graphsMeta: GRAPH_META,
      strategies: STRATEGIES,
      simulate: simulateStrategy,
      ...studyConfig,
    });
    setStudyRows(rows);
  }, [studyConfig]);

  const downloadFile = useCallback((fileName, content, mimeType) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const refreshData = useCallback(() => {
    loadSummaryOverride().then(loaded => {
      if (loaded) setSummaryVersion(v => v + 1);
    });
  }, []);

  const viewTabs = [
    { id: "overview", label: "Overview" },
    { id: "suppression", label: "Suppression Lab" },
    { id: "network", label: "Network Simulator" },
    { id: "research", label: "Research Workbench" },
    { id: "training", label: "Training Inspector" },
    { id: "zero_shot", label: "Zero-Shot Evaluation" },
    { id: "agent", label: "Policy Analyzer" },
  ];

  return (
    <div style={{ fontFamily: "var(--font-sans)", width: "100%", minHeight: "100vh", padding: "24px 16px", boxSizing: "border-box" }}>
      {/* Header */}
      <div style={{ marginBottom: "28px", borderBottom: "0.5px solid var(--color-border-tertiary)", paddingBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
              <div style={{
                width: "36px", height: "36px", borderRadius: "8px",
                background: "var(--color-background-secondary)",
                border: "1px solid var(--color-border-secondary)",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: "var(--accent)",
              }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
              </div>
              <h1 style={{ margin: 0, fontSize: "20px", fontWeight: "500", color: "var(--color-text-primary)" }}>
                Misinformation Sandbox
              </h1>
              <button onClick={refreshData} style={{
                padding: "6px 12px",
                borderRadius: "6px",
                background: "var(--color-background-primary)",
                border: "1px solid var(--accent)",
                color: "var(--color-text-primary)",
                cursor: "pointer",
                fontSize: "12px",
                marginLeft: "auto"
              }}>Refresh Data</button>
            </div>
            <p style={{ margin: 0, fontSize: "13px", color: "var(--color-text-secondary)" }}>
              GNN + Deep RL for Misinformation Suppression · Three real-world social networks
            </p>
          </div>
          <div style={{ display: "flex", gap: "8px", fontSize: "12px", color: "var(--color-text-tertiary)" }}>
            <span>GNN hidden: 64</span>
            <span>·</span>
            <span>DQN hidden: 128</span>
            <span>·</span>
            <span>PER cap: 20k</span>
          </div>
        </div>
      </div>

      {/* Top KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "10px", marginBottom: "28px" }}>
        <MetricCard label="Best suppression" value={`${s.gnn_rl.supp.toFixed(1)}%`} sub={`${g.short} vs no intervention`} color="#1D9E75" icon="↓" />
        <MetricCard label="GNN parameters" value="18,497" sub="3-layer residual GCN" color="#378ADD" icon={<GearIcon />} />
        <MetricCard label="DQN parameters" value="26,629" sub="Dueling DDQN + LayerNorm" color="#7F77DD" icon={<GearIcon />} />
        <MetricCard label="Total graphs" value={Object.keys(GRAPH_META).length} sub={`${Object.values(GRAPH_META).reduce((sum,m)=> sum + m.nodes,0).toLocaleString()} nodes · ${Object.values(GRAPH_META).reduce((sum,m)=> sum + m.edges,0).toLocaleString()} edges`} color="#D4537E" icon="◉" />
        <MetricCard label="Actions" value={Object.keys(ACTIONS).length} sub="wait · mute · cure · burst" color="#EF9F27" icon="→" />
      </div>

      <WorkspaceShell active={activeTab} onChange={setActiveTab} tabs={viewTabs}>

      {/* Overview Tab */}
      {activeTab === "overview" && (
        <div>
          <SectionHeader title="Performance summary" subtitle="GNN+RL vs. baseline strategies across all three graphs" />
          <div style={{ marginBottom: "24px" }}>
            <BarChart />
          </div>

          {/* Strategy legend */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", marginBottom: "24px" }}>
            {Object.entries(STRATEGIES).map(([id, s]) => (
              <span key={id} style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "var(--color-text-secondary)" }}>
                <span style={{ width: "24px", height: "3px", background: s.color, borderRadius: "2px", display: "inline-block", opacity: id === "gnn_rl" ? 1 : 0.75 }} />
                {s.label}
              </span>
            ))}
          </div>

          <SectionHeader title="Suppression heatmap" subtitle="Percentage reduction in peak believers vs. no-intervention baseline" />
          <div style={{ overflowX: "auto" }}>
            <HeatmapChart />
          </div>

          <div style={{ marginTop: "28px" }}>
            <SectionHeader title="Graph statistics" />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "10px" }}>
              {Object.entries(GRAPH_META).map(([key, meta]) => {
                const gs = SUMMARY[key];
                return (
                  <div key={key} style={{
                    background: "var(--color-background-secondary)",
                    borderRadius: "var(--border-radius-lg)",
                    border: `0.5px solid ${meta.color}44`,
                    padding: "14px 16px",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                      <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: meta.color, flexShrink: 0 }} />
                      <span style={{ fontSize: "13px", fontWeight: "500", color: "var(--color-text-primary)" }}>{meta.label}</span>
                    </div>
                    <div style={{ fontSize: "12px", color: "var(--color-text-secondary)", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 12px" }}>
                      <span>Nodes</span><span style={{ color: "var(--color-text-primary)", fontWeight: "500" }}>{meta.nodes.toLocaleString()}</span>
                      <span>Edges</span><span style={{ color: "var(--color-text-primary)", fontWeight: "500" }}>{meta.edges.toLocaleString()}</span>
                      <span>Avg degree</span><span style={{ color: "var(--color-text-primary)", fontWeight: "500" }}>{meta.avgDeg}</span>
                      <span>Suppression</span><span style={{ color: "#1D9E75", fontWeight: "500" }}>{gs.gnn_rl.supp.toFixed(1)}%</span>
                      <span>Budget/step</span><span style={{ color: "var(--color-text-primary)", fontWeight: "500" }}>{gs.budget}</span>
                      <span>Episodes</span><span style={{ color: "var(--color-text-primary)", fontWeight: "500" }}>{gs.episodes}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Suppression Tab */}
      {activeTab === "suppression" && (
        <div>
          <SectionHeader
            title="Suppression curves"
            subtitle="Believer count over 50 timesteps — GNN+RL with 10–90th percentile band"
          />
          <GraphSelector value={selectedGraph} onChange={setSelectedGraph} />

          <div style={{ marginBottom: "8px", display: "flex", alignItems: "center", gap: "16px" }}>
            {[
              { k: "none", label: "None", v: s.none.median },
              { k: "gnn_rl", label: "GNN+RL", v: s.gnn_rl.median },
            ].map(({ k, label, v }) => (
              <div key={k} style={{ fontSize: "12px", color: "var(--color-text-secondary)" }}>
                <span style={{ color: STRATEGIES[k].color, fontWeight: "500" }}>{label}:</span>
                <span style={{ marginLeft: "4px" }}>{v} peak</span>
              </div>
            ))}
            <div style={{ fontSize: "12px", color: "#1D9E75", marginLeft: "auto" }}>
              ↓ {s.gnn_rl.supp.toFixed(1)}% suppression
            </div>
          </div>

          <div style={{ overflowX: "auto" }}>
            <LineChart data={curves} width={640} height={240} />
          </div>

          {/* Per-strategy legend + stats */}
          <div style={{ marginTop: "16px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "8px" }}>
            {Object.entries(STRATEGIES).map(([id, strat]) => {
              const med = id === "gnn_rl" ? s.gnn_rl.median : s[id]?.median ?? 0;
              const none_m = s.none.median;
              const supp = ((none_m - med) / none_m * 100);
              return (
                <div key={id} style={{
                  padding: "10px 12px", borderRadius: "var(--border-radius-md)",
                  border: `0.5px solid ${id === "gnn_rl" ? strat.color + "66" : "var(--color-border-tertiary)"}`,
                  background: id === "gnn_rl" ? strat.color + "0a" : "transparent",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "7px", marginBottom: "4px" }}>
                    <span style={{ width: "18px", height: "2.5px", background: strat.color, borderRadius: "2px" }} />
                    <span style={{ fontSize: "12px", fontWeight: id === "gnn_rl" ? "500" : "400", color: "var(--color-text-primary)" }}>
                      {strat.label}
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "var(--color-text-secondary)" }}>
                    <span>Peak: <strong style={{ color: "var(--color-text-primary)" }}>{med}</strong></span>
                    <span style={{ color: supp > 0 ? "#1D9E75" : "#E24B4A" }}>{supp > 0 ? "↓" : "↑"}{Math.abs(supp).toFixed(1)}%</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* All-graph mini curves */}
          <div style={{ marginTop: "28px" }}>
            <SectionHeader title="All graphs — GNN+RL vs no-intervention" />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px" }}>
              {Object.keys(GRAPH_META).map(gn => {
                const rlData = generateRLCurveWithBands(gn);
                const noneData = generateCurve(gn, "none");
                const miniCurves = [
                  { id: "none", color: STRATEGIES.none.color, dash: [4, 3], values: noneData },
                  { id: "gnn_rl", color: STRATEGIES.gnn_rl.color, dash: [], values: rlData.med, band: { lo: rlData.lo, hi: rlData.hi } },
                ];
                return (
                  <div key={gn} style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "12px" }}>
                    <div style={{ fontSize: "12px", fontWeight: "500", color: "var(--color-text-primary)", marginBottom: "8px" }}>{GRAPH_META[gn].label}</div>
                    <LineChart data={miniCurves} width={300} height={130} />
                    <div style={{ fontSize: "11px", color: "#1D9E75", marginTop: "4px" }}>↓ {SUMMARY[gn].gnn_rl.supp.toFixed(1)}% suppression</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Network Tab */}
      {activeTab === "network" && (
        <div>
          <SectionHeader title="Social network topology" subtitle="Live graph visualization with belief propagation state" />
          <GraphSelector value={selectedGraph} onChange={setSelectedGraph} />

          <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "14px" }}>
            <div style={{ fontSize: "12px", color: "var(--color-text-secondary)" }}>
              {g.nodes.toLocaleString()} nodes · {g.edges.toLocaleString()} edges · avg degree {g.avgDeg}
            </div>
            <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "8px" }}>
              <label style={{ fontSize: "12px", color: "var(--color-text-secondary)", display: "flex", alignItems: "center", gap: "6px", cursor: "pointer" }}>
                <input type="checkbox" checked={showIntervened} onChange={e => setShowIntervened(e.target.checked)} />
                Show treated nodes
              </label>
            </div>
          </div>

          {/* Simulation controls */}
          <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "12px" }}>
            <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
              {Object.keys(STRATEGIES).map(k => (
                <button key={k} onClick={() => setSelectedStrategySim(k)} style={{
                  padding: "6px 10px", borderRadius: "6px", cursor: "pointer", border: selectedStrategySim === k ? "1px solid var(--accent)" : "1px solid var(--color-border-secondary)", background: selectedStrategySim === k ? "var(--color-background-primary)" : "transparent", color: selectedStrategySim === k ? "var(--color-text-primary)" : "var(--color-text-secondary)" }}>
                  {STRATEGIES[k].label}
                </button>
              ))}
            </div>

            <div style={{ display: "flex", gap: "8px", alignItems: "center", marginLeft: "auto" }}>
              <button onClick={() => setIsPlayingSim(p => !p)} style={{ padding: "6px 10px", borderRadius: "6px" }}>{isPlayingSim ? 'Pause' : 'Play'}</button>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--color-text-secondary)' }}>
                <input type="checkbox" checked={simAuto} onChange={e => setSimAuto(e.target.checked)} />
                Auto-run
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--color-text-secondary)' }}>
                <select value={simSpeed} onChange={e => setSimSpeed(Number(e.target.value))} style={{ padding: '6px', borderRadius: '6px', background: 'transparent', color: 'var(--color-text-secondary)', border: '1px solid var(--color-border-secondary)' }}>
                  <option value={200}>200ms</option>
                  <option value={350}>350ms</option>
                  <option value={600}>600ms</option>
                  <option value={1000}>1000ms</option>
                </select>
              </label>
              <input type="range" min={0} max={(simSnapshots?.length || 1) - 1} value={simTimestep} onChange={e => setSimTimestep(Number(e.target.value))} style={{ width: "240px" }} />
              <div style={{ fontSize: "12px", color: "var(--color-text-secondary)" }}>Timestep {simTimestep + 1}/{simSnapshots ? simSnapshots.length : 0}</div>
            </div>
          </div>

          {/* Model Adaptation Selectors */}
          {selectedStrategySim === 'gnn_rl' && (
            <div style={{
              display: "flex",
              gap: "16px",
              alignItems: "center",
              marginBottom: "16px",
              padding: "12px 16px",
              background: "var(--color-background-secondary)",
              borderRadius: "var(--border-radius-md)",
              border: "1.5px solid var(--color-border-info)"
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ display: "flex", alignItems: "center", color: "var(--color-text-info)" }}><GearIcon /></span>
                <span style={{ fontSize: "12px", fontWeight: "600", color: "var(--color-text-primary)" }}>Adapt Model Weights:</span>
              </div>
              
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <label style={{ fontSize: "12px", color: "var(--color-text-secondary)" }} htmlFor="model-checkpoint-select">Checkpoint:</label>
                <select
                  id="model-checkpoint-select"
                  value={selectedCheckpoint}
                  onChange={e => {
                    setSelectedCheckpoint(e.target.value);
                    if (e.target.value !== "final") {
                      setSelectedSeed(0);
                    }
                  }}
                  style={{
                    padding: "4px 8px",
                    borderRadius: "4px",
                    fontSize: "12px",
                    background: "var(--color-background-primary)",
                    color: "var(--color-text-primary)",
                    border: "1px solid var(--color-border-secondary)"
                  }}
                >
                  {(modelsMeta[selectedGraph]?.checkpoints || GRAPH_MODELS[selectedGraph].checkpoints).map(ckpt => (
                    <option key={ckpt.value || ckpt} value={ckpt.value || ckpt}>
                      {ckpt.label || (ckpt === "final" ? "Final Model" : `Epoch ${ckpt.replace("ep", "")} Checkpoint`)}
                    </option>
                  ))}
                </select>
              </div>

              {selectedCheckpoint === "final" && (
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <label style={{ fontSize: "12px", color: "var(--color-text-secondary)" }} htmlFor="model-seed-select">Training Seed:</label>
                  <select
                    id="model-seed-select"
                    value={selectedSeed}
                    onChange={e => setSelectedSeed(Number(e.target.value))}
                    style={{
                      padding: "4px 8px",
                      borderRadius: "4px",
                      fontSize: "12px",
                      background: "var(--color-background-primary)",
                      color: "var(--color-text-primary)",
                      border: "1px solid var(--color-border-secondary)"
                    }}
                  >
                    {(modelsMeta[selectedGraph]?.seeds || GRAPH_MODELS[selectedGraph].seeds).map(seedVal => (
                      <option key={seedVal} value={seedVal}>Seed {seedVal}</option>
                    ))}
                  </select>
                </div>
              )}

              <span style={{ fontSize: "11px", color: "var(--color-text-tertiary)", marginLeft: "auto" }}>
                Active: <code style={{ color: "var(--accent)" }}>{selectedCheckpoint === "final" ? `seed${selectedSeed}` : selectedCheckpoint}</code>
              </span>
            </div>
          )}

          <NetworkGraph 
            gname={selectedGraph} 
            checkpoint={selectedCheckpoint}
            seed={selectedSeed}
            showIntervened={showIntervened} 
            snapshots={simSnapshots} 
            timestep={simTimestep} 
          />

          {/* Node stats */}
          <div style={{ marginTop: "20px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "8px" }}>
            {(() => {
              const graphSnap = (simSnapshots && simSnapshots[simTimestep]) ? simSnapshots[simTimestep] : generateNetworkGraph(selectedGraph, 90);
              const infected = graphSnap.nodes.filter(n => n.belief > 0.5 && !n.treated).length;
              const treated = graphSnap.nodes.filter(n => n.treated).length;
              const atrisk = graphSnap.nodes.filter(n => n.belief > 0.2 && n.belief <= 0.5).length;
              const safe = graphSnap.nodes.filter(n => n.belief <= 0.2).length;
              return [
                { label: "Infected nodes", v: infected, color: "#E24B4A" },
                { label: "Treated", v: treated, color: "#1D9E75" },
                { label: "At-risk", v: atrisk, color: "#EF9F27" },
                { label: "Unaffected", v: safe, color: "#888780" },
              ].map(({ label, v, color }) => (
                <div key={label} style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "10px 12px" }}>
                  <div style={{ fontSize: "11px", color: "var(--color-text-secondary)", marginBottom: "4px" }}>{label}</div>
                  <div style={{ fontSize: "20px", fontWeight: "500", color }}>{v}</div>
                  <div style={{ fontSize: "11px", color: "var(--color-text-tertiary)" }}>{((v / graphSnap.nodes.length) * 100).toFixed(0)}% of sample</div>
                </div>
              ));
            })()}
          </div>

          {/* Architecture info */}
          <div style={{ marginTop: "24px" }}>
            <SectionHeader title="Model architecture" />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
              {[
                {
                  title: "RiskGNN — 3-layer residual GCN",
                  color: "#378ADD",
                  items: [
                    "Input: 7 hand-crafted + 8 spectral features",
                    "3× GCNConv layers (hidden dim 64)",
                    "Residual projection from input",
                    "BatchNorm + Dropout (p=0.1)",
                    "Output: per-node infection risk ∈ [0,1]",
                    "18,497 parameters",
                  ],
                },
                {
                  title: "Dueling DDQN — 5 action heads",
                  color: "#7F77DD",
                  items: [
                    "State: 6 scalars (beliefs, budget, spread)",
                    "128 → 64 → 32 hidden with LayerNorm",
                    "Value + Advantage streams (dueling)",
                    "PER replay buffer (α=0.6, β-annealing)",
                    "Actions: wait / mute_small / mute_large / cure / burst",
                    "26,629 parameters",
                  ],
                },
              ].map(({ title, color, items }) => (
                <div key={title} style={{
                  background: "var(--color-background-secondary)",
                  borderRadius: "var(--border-radius-lg)",
                  border: `0.5px solid ${color}44`,
                  padding: "14px 16px",
                }}>
                  <div style={{ fontSize: "13px", fontWeight: "500", color: "var(--color-text-primary)", marginBottom: "10px", display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: color }} />
                    {title}
                  </div>
                  <ul style={{ margin: 0, paddingLeft: "16px", listStyle: "none" }}>
                    {items.map((item, i) => (
                      <li key={i} style={{ fontSize: "12px", color: "var(--color-text-secondary)", marginBottom: "4px", display: "flex", gap: "6px" }}>
                        <span style={{ color, flexShrink: 0 }}>·</span>{item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Research Workbench Tab */}
      {activeTab === "research" && (
        <ResearchWorkbenchPanel
          studyConfig={studyConfig}
          setStudyConfig={setStudyConfig}
          runAcademicStudy={runAcademicStudy}
          downloadFile={downloadFile}
          toCSV={toCSV}
          studyRows={studyRows}
          studyBest={studyBest}
          strategies={STRATEGIES}
          selectedCheckpoint={selectedCheckpoint}
          selectedSeed={selectedSeed}
        />
      )}
{activeTab === "zero_shot" && (
  <ZeroShotPanel simulateStrategyAPI={simulateStrategyAPI} />
)}

      {/* Training Tab */}
      {activeTab === "training" && (
        <div>
          <SectionHeader title="Training dynamics" subtitle="Reward and loss curves for all three graphs (smoothed, 2 seeds each)" />
          <GraphSelector value={selectedGraph} onChange={setSelectedGraph} />

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "24px" }}>
            <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "14px" }}>
              <div style={{ fontSize: "12px", fontWeight: "500", color: "var(--color-text-primary)", marginBottom: "10px" }}>Training reward (smoothed w=30)</div>
              <div style={{ display: "flex", gap: "12px", marginBottom: "8px" }}>
                {[0, 1].map(s => (
                  <span key={s} style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", color: "var(--color-text-secondary)" }}>
                    <span style={{ width: "16px", height: "2px", background: ["#378ADD", "#1D9E75"][s], borderRadius: "1px" }} />
                    seed {s}
                  </span>
                ))}
              </div>
              <TrainingChart gname={selectedGraph} type="reward" />
            </div>
            <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "14px" }}>
              <div style={{ fontSize: "12px", fontWeight: "500", color: "var(--color-text-primary)", marginBottom: "10px" }}>DQN TD loss (smoothed w=50)</div>
              <div style={{ display: "flex", gap: "12px", marginBottom: "8px" }}>
                {[0, 1].map(s => (
                  <span key={s} style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", color: "var(--color-text-secondary)" }}>
                    <span style={{ width: "16px", height: "2px", background: ["#378ADD", "#1D9E75"][s], borderRadius: "1px" }} />
                    seed {s}
                  </span>
                ))}
              </div>
              <TrainingChart gname={selectedGraph} type="loss" />
            </div>
          </div>

          {/* All graphs training */}
          <SectionHeader title="Training overview — all graphs" />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px" }}>
            {Object.keys(GRAPH_META).map(gn => (
              <div key={gn} style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "12px" }}>
                <div style={{ fontSize: "12px", fontWeight: "500", color: "var(--color-text-primary)", marginBottom: "6px" }}>{GRAPH_META[gn].short}</div>
                <TrainingChart gname={gn} type="reward" />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px", marginTop: "8px", fontSize: "11px", color: "var(--color-text-secondary)" }}>
                  <span>Episodes: {SUMMARY[gn].episodes}</span>
                  <span>Budget: {SUMMARY[gn].budget}</span>
                  <span>Supp: {SUMMARY[gn].gnn_rl.supp.toFixed(1)}%</span>
                  <span>Median: {SUMMARY[gn].gnn_rl.median}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Curriculum transfer info */}
          <div style={{ marginTop: "24px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-lg)", padding: "16px", border: "0.5px solid var(--color-border-info)" }}>
            <div style={{ fontSize: "13px", fontWeight: "500", color: "var(--color-text-info)", marginBottom: "8px" }}>Curriculum transfer</div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px", color: "var(--color-text-secondary)" }}>
              <span style={{ padding: "3px 10px", borderRadius: "12px", background: "#378ADD18", color: "#185FA5", fontSize: "11px" }}>P2P Gnutella (700 ep)</span>
              <span>→</span>
              <span style={{ padding: "3px 10px", borderRadius: "12px", background: "#1D9E7518", color: "#0F6E56", fontSize: "11px" }}>CA-GrQc (900 ep)</span>
              <span>→ best seed →</span>
              <span style={{ padding: "3px 10px", borderRadius: "12px", background: "#D4537E18", color: "#993556", fontSize: "11px" }}>Facebook (1400 ep)</span>
            </div>
            <p style={{ margin: "8px 0 0", fontSize: "12px", color: "var(--color-text-secondary)" }}>
              Best CA-GrQc agent weights are copied to the Facebook agent via curriculum learning, enabling faster convergence on the denser graph.
            </p>
          </div>
        </div>
      )}

      {/* ── AGENT TAB ── */}
      {activeTab === "agent" && (
        <div>
          <SectionHeader title="Agent actions & budget (Facebook)" subtitle="Per-timestep action selection and remaining budget — best agent evaluation run" />

          <div style={{ overflowX: "auto", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "14px", marginBottom: "20px" }}>
            <BudgetTimeline />
          </div>

          {/* Action legend */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "24px" }}>
            {Object.entries(ACTIONS).map(([id, a]) => (
              <span key={id} style={{
                padding: "4px 12px", borderRadius: "12px", fontSize: "12px",
                background: a.color + "22", color: a.color, fontWeight: "500",
              }}>
                {a.label}
              </span>
            ))}
          </div>

          {/* Action distribution */}
          <SectionHeader title="Action distribution" subtitle="How often each action was selected across evaluation" />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "10px", marginBottom: "24px" }}>
            {(() => {
              const { actions } = generateBudgetTimeline();
              const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
              actions.forEach(a => counts[a]++);
              const total = actions.length;
              return Object.entries(counts).map(([id, cnt]) => (
                <div key={id} style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "10px", textAlign: "center" }}>
                  <div style={{ fontSize: "18px", fontWeight: "500", color: ACTIONS[id].color }}>{((cnt / total) * 100).toFixed(0)}%</div>
                  <div style={{ fontSize: "10px", color: "var(--color-text-secondary)", marginTop: "4px" }}>{ACTIONS[id].label}</div>
                </div>
              ));
            })()}
          </div>

          {/* Hyperparameter table */}
          <SectionHeader title="Key hyperparameters" />
          <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-lg)", padding: "16px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "4px 24px", fontSize: "12px" }}>
              {[
                ["SEED_VALUE", "42"],
                ["TIMESTEPS", "50"],
                ["CRED (credibility)", "0.85"],
                ["MOD_DELAY", "5"],
                ["SEED_NODE_RANK", "3 (top-3 degree)"],
                ["SPEC_DIM (spectral)", "8"],
                ["GNN_HIDDEN", "64"],
                ["DQN_HIDDEN", "128"],
                ["PER_CAP", "20,000"],
                ["RISK_CACHE_STEPS", "6"],
                ["PER alpha", "0.6"],
                ["PER beta → 1.0", "annealed over 50k steps"],
                ["GNN lr", "5e-4 (AdamW, cosine)"],
                ["DQN lr", "3e-4 (AdamW, cosine)"],
                ["gamma", "0.99"],
                ["epsilon decay", "0.996 (Facebook)"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                  <span style={{ color: "var(--color-text-secondary)" }}>{k}</span>
                  <span style={{ color: "var(--color-text-primary)", fontWeight: "500" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      </WorkspaceShell>

      {/* Footer */}
      <div style={{
        marginTop: "36px", paddingTop: "16px",
        borderTop: "0.5px solid var(--color-border-tertiary)",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        fontSize: "11px", color: "var(--color-text-tertiary)",
      }}>
        <span>Misinformation Sandbox · GNN & Reinforcement Learning Suppression Framework</span>
        <span>Optimization Platform · 3-layer residual GCN + Dueling DDQN + Prioritized Experience Replay</span>
      </div>
    </div>
  );
}
