import React, { useState } from "react";

// Shield Icon SVG
const ShieldIcon = ({ size = 24, color = "#2ecc71" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);

// Seeded random helper for reproducible unseen graphs
function seededRand(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 4294967296; };
}

function generateUnseenGraph(type, numNodes, param) {
  const rng = seededRand(Date.now() + 999);
  const nodesArr = [];
  const edgesArr = [];
  const n = numNodes;

  // Simple ring layout for aesthetic rendering
  for (let i = 0; i < n; i++) {
    const angle = (i / n) * 2 * Math.PI + rng() * 0.15;
    const r = 0.22 + rng() * 0.26;
    nodesArr.push({
      id: i,
      x: 0.5 + r * Math.cos(angle),
      y: 0.5 + r * Math.sin(angle),
      belief: rng() < 0.5 ? 0.8 + rng() * 0.2 : rng() * 0.3,
      degree: 0,
    });
  }

  const edgeSet = new Set();
  if (type === "barabasi_albert") {
    const m = Math.max(1, Math.min(n - 1, param));
    for (let i = 1; i < n; i++) {
      const numLinks = Math.min(i, m);
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
  } else if (type === "erdos_renyi") {
    const p = Math.max(0.01, Math.min(0.2, param / 100));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (rng() < p) {
          edgesArr.push([i, j]);
          nodesArr[i].degree++;
          nodesArr[j].degree++;
        }
      }
    }
  } else {
    // Small world lattice
    const k = 4;
    const beta = Math.max(0.05, Math.min(0.8, param / 10));
    for (let i = 0; i < n; i++) {
      for (let j = 1; j <= k / 2; j++) {
        const nbr = (i + j) % n;
        edgesArr.push([i, nbr]);
      }
    }
    for (let idx = 0; idx < edgesArr.length; idx++) {
      if (rng() < beta) {
        const u = edgesArr[idx][0];
        let v = Math.floor(rng() * n);
        while (v === u || edgeSet.has(`${Math.min(u, v)}-${Math.max(u, v)}`)) {
          v = Math.floor(rng() * n);
        }
        edgesArr[idx][1] = v;
      }
    }
    edgesArr.forEach(([u, v]) => {
      nodesArr[u].degree++;
      nodesArr[v].degree++;
    });
  }

  // Mark seed node (highest degree)
  const seedIdx = nodesArr.reduce((best, nd, i) => nd.degree > nodesArr[best].degree ? i : best, 0);
  nodesArr[seedIdx].belief = 1.0;
  nodesArr[seedIdx].isSeed = true;

  return { nodes: nodesArr, edges: edgesArr, seedIdx };
}

export default function ZeroShotPanel({ simulateStrategyAPI }) {
  const [unseenType, setUnseenType] = useState("barabasi_albert");
  const [unseenNodes, setUnseenNodes] = useState(80);
  const [unseenParam, setUnseenParam] = useState(3);
  const [evaluating, setEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState(null);
  const [activeStep, setActiveStep] = useState(49);
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-step effect for unseen graph snapshots
  React.useEffect(() => {
    if (!isPlaying || !evalResult) return;
    const id = setInterval(() => {
      setActiveStep(s => (s + 1) % 50);
    }, 400);
    return () => clearInterval(id);
  }, [isPlaying, evalResult]);
  const handleEvaluateUnseen = async () => {
    setEvaluating(true);
    setEvalResult(null);
    try {
      const base = generateUnseenGraph(unseenType, unseenNodes, unseenParam);
      const noneSnaps = await simulateStrategyAPI("unseen_custom", "none", base.nodes, base.edges, 50, 20.0);
      const gnnSnaps = await simulateStrategyAPI("unseen_custom", "gnn_rl", base.nodes, base.edges, 50, 20.0);

      if (noneSnaps && gnnSnaps) {
        const noneCurve = noneSnaps.map(snap => snap.filter(n => n.belief > 0.5).length);
        const gnnCurve = gnnSnaps.map(snap => snap.filter(n => n.belief > 0.5).length);
        
        // Exclude initial snapshot (index 0) from peak calculation to reflect intervention effect
        const nonePeak = Math.max(...noneCurve.slice(1));
        const gnnPeak = Math.max(...gnnCurve.slice(1));
        const rating = nonePeak > 0 ? ((nonePeak - gnnPeak) / nonePeak * 100) : 0.0;

        const positionedSnaps = gnnSnaps.map(snap => ({
          nodes: snap.map((n, idx) => ({
            ...n,
            x: base.nodes[idx].x,
            y: base.nodes[idx].y
          })),
          edges: base.edges
        }));

        setEvalResult({
          noneCurve,
          gnnCurve,
          nonePeak,
          gnnPeak,
          rating,
          snapshots: positionedSnaps
        });
        setActiveStep(49);
      }
    } catch (err) {
      alert("Evaluation failed. Make sure the FastAPI backend is running on port 8000!");
      console.error(err);
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div>
      <div style={{
        background: "var(--color-background-secondary)",
        borderRadius: "var(--border-radius-lg)",
        padding: "20px",
        border: "1px solid var(--color-border-secondary)",
      }}>
        <h3 style={{ margin: "0 0 4px", fontSize: "14px", fontWeight: 600, color: "var(--color-text-primary)" }}>
          Zero-Shot Inference on Unseen Graph Topologies
        </h3>
        <p style={{ margin: "0 0 16px", fontSize: "12px", color: "var(--color-text-secondary)" }}>
          Generate an arbitrary network structure not seen during training, and evaluate the GNN+RL model's containment rating in real time.
        </p>
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: "12px",
          marginBottom: "20px",
        }}>
          <label style={{ fontSize: "11px", color: "var(--color-text-secondary)", display: "flex", flexDirection: "column", gap: "4px" }}>
            Graph Model
            <select
              value={unseenType}
              onChange={e => setUnseenType(e.target.value)}
              style={{
                padding: "8px 10px",
                borderRadius: "6px",
                background: "var(--color-background-secondary)",
                color: "var(--color-text-primary)",
                border: "1px solid var(--color-border-secondary)",
                fontSize: "12px",
                cursor: "pointer",
                outline: "none",
              }}
            >
              <option value="barabasi_albert" style={{ background: "var(--color-background-secondary)", color: "var(--color-text-primary)" }}>Barabási-Albert (Scale-Free)</option>
              <option value="erdos_renyi" style={{ background: "var(--color-background-secondary)", color: "var(--color-text-primary)" }}>Erdős-Rényi (Random)</option>
              <option value="watts_strogatz" style={{ background: "var(--color-background-secondary)", color: "var(--color-text-primary)" }}>Watts-Strogatz (Small-World)</option>
            </select>
          </label>
          <label style={{ fontSize: "11px", color: "var(--color-text-secondary)", display: "flex", flexDirection: "column", gap: "4px" }}>
            Node Count ({unseenNodes})
            <input
              type="range"
              min={40}
              max={120}
              value={unseenNodes}
              onChange={e => setUnseenNodes(Number(e.target.value))}
              style={{ marginTop: "4px" }}
            />
          </label>
          <label style={{ fontSize: "11px", color: "var(--color-text-secondary)", display: "flex", flexDirection: "column", gap: "4px" }}>
            Density Factor / Param ({unseenParam})
            <input
              type="range"
              min={1}
              max={8}
              value={unseenParam}
              onChange={e => setUnseenParam(Number(e.target.value))}
              style={{ marginTop: "4px" }}
            />
          </label>
        </div>
        <button
          onClick={handleEvaluateUnseen}
          disabled={evaluating}
          style={{
            alignSelf: "flex-end",
            padding: "7px 14px",
            borderRadius: "6px",
            cursor: "pointer",
            background: "var(--accent)",
            color: "white",
            fontWeight: "500",
            border: "none",
            opacity: evaluating ? 0.7 : 1,
          }}
        >
          {evaluating ? "Evaluating..." : "Generate & Run Real-Time evaluation"}
        </button>
      </div>

      {evalResult && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1.2fr",
          gap: "20px",
          background: "var(--color-background-primary)",
          borderRadius: "var(--border-radius-lg)",
          padding: "16px",
          border: "0.5px solid var(--color-border-tertiary)",
        }}>
          {/* Visualizer Panel */}
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "12px", fontWeight: "600", color: "var(--color-text-primary)" }}>
                Topology propagation (Timestep {activeStep + 1}/50)
              </span>
              <div style={{ display: "flex", gap: "6px" }}>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  style={{ padding: "4px 8px", borderRadius: "4px", fontSize: "10px", cursor: "pointer" }}
                >
                  {isPlaying ? "Pause" : "Play propagation"}
                </button>
                <button
                  onClick={() => {
                    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(evalResult, null, 2));
                    const downloadAnchor = document.createElement('a');
                    downloadAnchor.setAttribute("href", dataStr);
                    downloadAnchor.setAttribute("download", `zero_shot_simulation_${unseenType}.json`);
                    document.body.appendChild(downloadAnchor);
                    downloadAnchor.click();
                    downloadAnchor.remove();
                  }}
                  style={{ padding: "4px 8px", borderRadius: "4px", fontSize: "10px", cursor: "pointer", border: "1px solid var(--color-accent)", color: "var(--color-text-primary)" }}
                >
                  Download JSON
                </button>
              </div>
            </div>
            {/* Render small custom network snapshot */}
            <div style={{
              height: "190px",
              border: "1px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-md)",
              background: "var(--color-background-secondary)",
              position: "relative",
              overflow: "hidden",
            }}>
              <svg viewBox="0 0 400 300" style={{ width: "100%", height: "100%" }}>
                {/* Edges */}
                {evalResult.snapshots[activeStep].edges.map(([u, v], i) => {
                  const nu = evalResult.snapshots[activeStep].nodes[u];
                  const nv = evalResult.snapshots[activeStep].nodes[v];
                  if (!nu || !nv) return null;
                  return (
                    <line
                      key={i}
                      x1={nu.x * 400} y1={nu.y * 300}
                      x2={nv.x * 400} y2={nv.y * 300}
                      stroke="var(--color-border-tertiary)"
                      strokeWidth="0.4"
                    />
                  );
                })}
                {/* Nodes */}
                {evalResult.snapshots[activeStep].nodes.map((n, i) => {
                  let fill = "#B4B2A9"; // Unaffected
                  if (n.isSeed) fill = "#D4537E";
                  else if (n.treated) fill = "#1D9E75";
                  else if (n.belief > 0.5) fill = "#E24B4A";
                  else if (n.belief > 0.2) fill = "#EF9F27";
                  return (
                    <circle
                      key={i}
                      cx={n.x * 400} cy={n.y * 300}
                      r={n.isSeed ? 4.5 : 2.5}
                      fill={fill}
                    />
                  );
                })}
              </svg>
            </div>
            <input
              type="range"
              min={0}
              max={49}
              value={activeStep}
              onChange={e => setActiveStep(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </div>
          {/* Results Curve & Suppression Rating */}
          <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            <div>
              <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                background: "rgba(46, 204, 113, 0.08)",
                border: "0.5px solid rgba(46, 204, 113, 0.35)",
                borderRadius: "var(--border-radius-md)",
                padding: "10px 14px",
                marginBottom: "12px",
              }}>
                <div>
                  <div style={{ fontSize: "10px", color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                    Zero-Shot Containment Rating
                  </div>
                  <div style={{ fontSize: "20px", fontWeight: "700", color: "#2ecc71" }}>
                    {evalResult.rating.toFixed(1)}% suppression
                  </div>
                </div>
                <ShieldIcon size={24} color="#2ecc71" />
              </div>
              <div style={{ fontSize: "11px", color: "var(--color-text-secondary)", marginBottom: "8px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
                <span>Peak (No intervention): <strong style={{ color: "#e74c3c" }}>{evalResult.nonePeak}</strong></span>
                <span>Peak (GNN+RL Model): <strong style={{ color: "#2ecc71" }}>{evalResult.gnnPeak}</strong></span>
              </div>
            </div>
            {/* Lightweight Line Chart */}
            <div style={{ height: "110px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "6px" }}>
              <svg viewBox="0 0 300 100" style={{ width: "100%", height: "100%" }}>
                {[0, 0.5, 1].map((f, idx) => (
                  <line key={idx} x1={20} y1={10 + 80 * f} x2={290} y2={10 + 80 * f} stroke="var(--color-border-tertiary)" strokeWidth="0.3" />
                ))}
                <path
                  d={evalResult.noneCurve.map((v, i) => `${i === 0 ? "M" : "L"}${20 + (i / 49) * 270},${90 - (v / Math.max(evalResult.nonePeak, 1)) * 80}`).join(" ")}
                  fill="none"
                  stroke="#e74c3c"
                  strokeWidth="1"
                  strokeDasharray="2,2"
                />
                <path
                  d={evalResult.gnnCurve.map((v, i) => `${i === 0 ? "M" : "L"}${20 + (i / 49) * 270},${90 - (v / Math.max(evalResult.nonePeak, 1)) * 80}`).join(" ")}
                  fill="none"
                  stroke="#2ecc71"
                  strokeWidth="1.6"
                />
                <text x={20} y={98} fontSize="7" fill="var(--color-text-tertiary)">T=0</text>
                <text x={290} y={98} textAnchor="end" fontSize="7" fill="var(--color-text-tertiary)">T=50</text>
              </svg>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
