/*
 * Academic evaluation helpers for benchmark-style analysis.
 * These utilities are UI-agnostic and can be reused in tests/notebooks.
 */

export function summarizeSnapshots(snapshots, infectionThreshold = 0.5) {
  const infectedSeries = snapshots.map((s) => s.nodes.filter((n) => n.belief > infectionThreshold && !n.treated).length);
  const treatedSeries = snapshots.map((s) => s.nodes.filter((n) => n.treated).length);
  const peakInfected = Math.max(...infectedSeries);
  const finalInfected = infectedSeries[infectedSeries.length - 1];
  const finalTreated = treatedSeries[treatedSeries.length - 1];
  const aucInfected = infectedSeries.reduce((a, b) => a + b, 0);
  return { peakInfected, finalInfected, finalTreated, aucInfected };
}

export function meanStd(values) {
  if (!values.length) return { mean: 0, std: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / values.length;
  return { mean, std: Math.sqrt(variance) };
}

export function evaluateAcademicBatch({ graphsMeta, strategies, simulate, timesteps, treatBudget, runs, infectionThreshold }) {
  const graphs = Object.keys(graphsMeta);
  const strategyKeys = Object.keys(strategies);
  const rows = [];

  graphs.forEach((gname) => {
    const baselineRuns = Array.from({ length: runs }, (_, runIdx) => summarizeSnapshots(simulate(gname, "none", timesteps, treatBudget, runIdx), infectionThreshold));
    const baselinePeak = meanStd(baselineRuns.map((r) => r.peakInfected)).mean;
    const baselineAuc = meanStd(baselineRuns.map((r) => r.aucInfected)).mean;

    strategyKeys.forEach((strategy) => {
      const runMetrics = Array.from({ length: runs }, (_, runIdx) => summarizeSnapshots(simulate(gname, strategy, timesteps, treatBudget, runIdx), infectionThreshold));

      const peak = meanStd(runMetrics.map((r) => r.peakInfected));
      const finalInf = meanStd(runMetrics.map((r) => r.finalInfected));
      const treated = meanStd(runMetrics.map((r) => r.finalTreated));
      const auc = meanStd(runMetrics.map((r) => r.aucInfected));

      const suppressionPeak = baselinePeak > 0 ? ((baselinePeak - peak.mean) / baselinePeak) * 100 : 0;
      const suppressionAuc = baselineAuc > 0 ? ((baselineAuc - auc.mean) / baselineAuc) * 100 : 0;

      rows.push({
        graph: graphsMeta[gname].short,
        graphKey: gname,
        strategy,
        strategyLabel: strategies[strategy].label,
        peakMean: peak.mean,
        peakStd: peak.std,
        finalMean: finalInf.mean,
        finalStd: finalInf.std,
        treatedMean: treated.mean,
        aucMean: auc.mean,
        aucStd: auc.std,
        suppressionPeak,
        suppressionAuc,
      });
    });
  });

  return rows;
}

export function toCSV(rows) {
  const headers = [
    "graph",
    "strategy",
    "peak_mean",
    "peak_std",
    "final_mean",
    "final_std",
    "treated_mean",
    "auc_mean",
    "auc_std",
    "suppression_peak_pct",
    "suppression_auc_pct",
  ];

  const lines = [headers.join(",")];
  rows.forEach((r) => {
    lines.push([
      r.graph,
      r.strategy,
      r.peakMean.toFixed(3),
      r.peakStd.toFixed(3),
      r.finalMean.toFixed(3),
      r.finalStd.toFixed(3),
      r.treatedMean.toFixed(3),
      r.aucMean.toFixed(3),
      r.aucStd.toFixed(3),
      r.suppressionPeak.toFixed(3),
      r.suppressionAuc.toFixed(3),
    ].join(","));
  });

  return lines.join("\n");
}
