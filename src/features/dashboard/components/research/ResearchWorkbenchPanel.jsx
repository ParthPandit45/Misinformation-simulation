import React from "react";

export default function ResearchWorkbenchPanel({
  studyConfig,
  setStudyConfig,
  runAcademicStudy,
  downloadFile,
  toCSV,
  studyRows,
  studyBest,
  strategies,
}) {
  return (
    <div>
      <div style={{ marginBottom: "16px" }}>
        <h2 style={{ margin: 0, fontSize: "16px", fontWeight: 600, color: "var(--color-text-primary)" }}>Academic Experiment Workbench</h2>
        <p style={{ margin: "4px 0 0", fontSize: "13px", color: "var(--color-text-secondary)" }}>
          Run repeatable benchmarks across all graphs and export publication-ready summary tables.
        </p>
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
        gap: "10px",
        marginBottom: "16px",
        background: "var(--color-background-primary)",
        borderRadius: "var(--border-radius-lg)",
        padding: "14px",
        border: "0.5px solid var(--color-border-tertiary)",
      }}>
        <label style={{ fontSize: "12px", color: "var(--color-text-secondary)", display: "grid", gap: "6px" }}>
          Timesteps
          <input
            type="number"
            min={20}
            max={200}
            value={studyConfig.timesteps}
            onChange={(e) => setStudyConfig((c) => ({ ...c, timesteps: Math.max(20, Math.min(200, Number(e.target.value) || 60)) }))}
            style={{ padding: "7px", borderRadius: "6px", background: "transparent", color: "var(--color-text-primary)", border: "1px solid var(--color-border-secondary)" }}
          />
        </label>

        <label style={{ fontSize: "12px", color: "var(--color-text-secondary)", display: "grid", gap: "6px" }}>
          Treatment budget/step
          <input
            type="number"
            min={1}
            max={10}
            value={studyConfig.treatBudget}
            onChange={(e) => setStudyConfig((c) => ({ ...c, treatBudget: Math.max(1, Math.min(10, Number(e.target.value) || 3)) }))}
            style={{ padding: "7px", borderRadius: "6px", background: "transparent", color: "var(--color-text-primary)", border: "1px solid var(--color-border-secondary)" }}
          />
        </label>

        <label style={{ fontSize: "12px", color: "var(--color-text-secondary)", display: "grid", gap: "6px" }}>
          Monte Carlo runs
          <input
            type="number"
            min={3}
            max={50}
            value={studyConfig.runs}
            onChange={(e) => setStudyConfig((c) => ({ ...c, runs: Math.max(3, Math.min(50, Number(e.target.value) || 8)) }))}
            style={{ padding: "7px", borderRadius: "6px", background: "transparent", color: "var(--color-text-primary)", border: "1px solid var(--color-border-secondary)" }}
          />
        </label>

        <label style={{ fontSize: "12px", color: "var(--color-text-secondary)", display: "grid", gap: "6px" }}>
          Infection threshold
          <input
            type="number"
            min={0.2}
            max={0.9}
            step={0.05}
            value={studyConfig.infectionThreshold}
            onChange={(e) => setStudyConfig((c) => ({ ...c, infectionThreshold: Math.max(0.2, Math.min(0.9, Number(e.target.value) || 0.5)) }))}
            style={{ padding: "7px", borderRadius: "6px", background: "transparent", color: "var(--color-text-primary)", border: "1px solid var(--color-border-secondary)" }}
          />
        </label>
      </div>

      <div style={{ display: "flex", gap: "8px", marginBottom: "16px" }}>
        <button onClick={runAcademicStudy} style={{ padding: "8px 12px", borderRadius: "6px" }}>Run academic benchmark</button>
        <button
          onClick={() => downloadFile("sandbox_academic_results.csv", toCSV(studyRows), "text/csv;charset=utf-8")}
          disabled={!studyRows.length}
          style={{ padding: "8px 12px", borderRadius: "6px", opacity: studyRows.length ? 1 : 0.5 }}
        >
          Export CSV
        </button>
        <button
          onClick={() => downloadFile("sandbox_academic_results.json", JSON.stringify(studyRows, null, 2), "application/json;charset=utf-8")}
          disabled={!studyRows.length}
          style={{ padding: "8px 12px", borderRadius: "6px", opacity: studyRows.length ? 1 : 0.5 }}
        >
          Export JSON
        </button>
      </div>

      {studyBest && (
        <div style={{
          marginBottom: "14px",
          background: "var(--color-background-primary)",
          borderRadius: "var(--border-radius-md)",
          border: "0.5px solid var(--color-border-tertiary)",
          padding: "12px",
          fontSize: "12px",
          color: "var(--color-text-secondary)",
        }}>
          <span style={{ color: "var(--color-text-primary)", fontWeight: 600 }}>Best observed policy:</span>
          <span style={{ marginLeft: "8px", color: strategies[studyBest.strategy].color }}>{studyBest.strategyLabel}</span>
          <span style={{ marginLeft: "8px" }}>on {studyBest.graph}</span>
          <span style={{ marginLeft: "8px", color: "#1D9E75" }}>Peak suppression: {studyBest.suppressionPeak.toFixed(2)}%</span>
        </div>
      )}

      <div style={{ overflowX: "auto", background: "var(--color-background-primary)", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-tertiary)", marginBottom: "36px" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--color-text-primary)", borderBottom: "1px solid var(--color-border-tertiary)" }}>
              <th style={{ padding: "10px" }}>Graph</th>
              <th style={{ padding: "10px" }}>Strategy</th>
              <th style={{ padding: "10px" }}>Peak infected (mean+-std)</th>
              <th style={{ padding: "10px" }}>Final infected (mean+-std)</th>
              <th style={{ padding: "10px" }}>Treated mean</th>
              <th style={{ padding: "10px" }}>AUC infected (mean+-std)</th>
              <th style={{ padding: "10px" }}>Suppression peak</th>
              <th style={{ padding: "10px" }}>Suppression AUC</th>
            </tr>
          </thead>
          <tbody>
            {studyRows.map((row, idx) => (
              <tr key={`${row.graphKey}-${row.strategy}-${idx}`} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", color: "var(--color-text-secondary)" }}>
                <td style={{ padding: "10px" }}>{row.graph}</td>
                <td style={{ padding: "10px", color: strategies[row.strategy].color }}>{row.strategyLabel}</td>
                <td style={{ padding: "10px" }}>{row.peakMean.toFixed(2)} +/- {row.peakStd.toFixed(2)}</td>
                <td style={{ padding: "10px" }}>{row.finalMean.toFixed(2)} +/- {row.finalStd.toFixed(2)}</td>
                <td style={{ padding: "10px" }}>{row.treatedMean.toFixed(2)}</td>
                <td style={{ padding: "10px" }}>{row.aucMean.toFixed(2)} +/- {row.aucStd.toFixed(2)}</td>
                <td style={{ padding: "10px", color: row.suppressionPeak >= 0 ? "#1D9E75" : "#E24B4A" }}>{row.suppressionPeak.toFixed(2)}%</td>
                <td style={{ padding: "10px", color: row.suppressionAuc >= 0 ? "#1D9E75" : "#E24B4A" }}>{row.suppressionAuc.toFixed(2)}%</td>
              </tr>
            ))}
            {!studyRows.length && (
              <tr>
                <td colSpan={8} style={{ padding: "14px", color: "var(--color-text-tertiary)" }}>
                  No benchmark results yet. Configure parameters and run the academic benchmark.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
