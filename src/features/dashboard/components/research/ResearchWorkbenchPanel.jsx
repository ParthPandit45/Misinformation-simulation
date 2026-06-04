import React, { useRef, useMemo, useCallback, useState, useEffect } from "react";

// ========== BENCHMARK ANALYZER ENGINE ==========
const BenchmarkAnalyzer = {
  rankByPerformance: (results) => {
    return [...results].sort((a, b) => b.suppressionPeak - a.suppressionPeak);
  },
  calculateRelativeImprovement: (baseline, target) => {
    if (baseline === 0) return target > 0 ? 100 : 0;
    return ((target - baseline) / Math.abs(baseline)) * 100;
  },
  identifyBestPerStrategy: (results, strategies) => {
    const byStrategy = {};
    results.forEach(row => {
      if (!byStrategy[row.strategy]) byStrategy[row.strategy] = [];
      byStrategy[row.strategy].push(row);
    });
    const best = {};
    Object.entries(byStrategy).forEach(([strategy, rows]) => {
      best[strategy] = rows.reduce((a, b) => b.suppressionPeak > a.suppressionPeak ? b : a);
    });
    return best;
  },
  calculateDominance: (result1, result2) => {
    const suppDiff = result1.suppressionPeak - result2.suppressionPeak;
    const aucDiff = result1.suppressionAuc - result2.suppressionAuc;
    const treatDiff = result2.treatedMean - result1.treatedMean; // Lower is better
    return suppDiff + (aucDiff * 0.7) + (treatDiff * 0.3);
  },
  groupByGraph: (results) => {
    const grouped = {};
    results.forEach(row => {
      if (!grouped[row.graph]) grouped[row.graph] = [];
      grouped[row.graph].push(row);
    });
    return grouped;
  },
  computeStrategyStats: (results, strategy) => {
    const filtered = results.filter(r => r.strategy === strategy);
    if (!filtered.length) return null;
    return {
      count: filtered.length,
      avgSuppression: filtered.reduce((s, r) => s + r.suppressionPeak, 0) / filtered.length,
      maxSuppression: Math.max(...filtered.map(r => r.suppressionPeak)),
      minSuppression: Math.min(...filtered.map(r => r.suppressionPeak)),
      avgCost: filtered.reduce((s, r) => s + r.treatedMean, 0) / filtered.length,
      avgAuc: filtered.reduce((s, r) => s + r.suppressionAuc, 0) / filtered.length,
    };
  },
};

// ========== STATISTICS ENGINE ==========
const StatisticsEngine = {
  mean: (arr) => arr.length === 0 ? 0 : arr.reduce((a, b) => a + b) / arr.length,
  std: (arr) => {
    const m = StatisticsEngine.mean(arr);
    const variance = arr.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / (arr.length || 1);
    return Math.sqrt(variance);
  },
  percentile: (arr, p) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    const weight = idx % 1;
    return sorted[lower] * (1 - weight) + (sorted[upper] || sorted[lower]) * weight;
  },
  median: (arr) => StatisticsEngine.percentile(arr, 50),
  quartiles: (arr) => ({
    q1: StatisticsEngine.percentile(arr, 25),
    q2: StatisticsEngine.percentile(arr, 50),
    q3: StatisticsEngine.percentile(arr, 75),
    iqr: StatisticsEngine.percentile(arr, 75) - StatisticsEngine.percentile(arr, 25),
  }),
  skewness: (arr) => {
    const m = StatisticsEngine.mean(arr);
    const s = StatisticsEngine.std(arr);
    if (s === 0) return 0;
    return (arr.reduce((sum, x) => sum + Math.pow((x - m) / s, 3), 0) / arr.length);
  },
  kurtosis: (arr) => {
    const m = StatisticsEngine.mean(arr);
    const s = StatisticsEngine.std(arr);
    if (s === 0) return 0;
    return (arr.reduce((sum, x) => sum + Math.pow((x - m) / s, 4), 0) / arr.length) - 3;
  },
  confidenceInterval: (arr, confidence = 0.95) => {
    const m = StatisticsEngine.mean(arr);
    const s = StatisticsEngine.std(arr);
    const z = 1.96; // 95% CI
    const margin = z * (s / Math.sqrt(arr.length));
    return { lower: m - margin, upper: m + margin, margin };
  },
  covariance: (arr1, arr2) => {
    const m1 = StatisticsEngine.mean(arr1);
    const m2 = StatisticsEngine.mean(arr2);
    const n = Math.min(arr1.length, arr2.length);
    return arr1.slice(0, n).reduce((sum, x, i) => sum + (x - m1) * (arr2[i] - m2), 0) / n;
  },
  correlation: (arr1, arr2) => {
    const cov = StatisticsEngine.covariance(arr1, arr2);
    const s1 = StatisticsEngine.std(arr1);
    const s2 = StatisticsEngine.std(arr2);
    if (s1 === 0 || s2 === 0) return 0;
    return cov / (s1 * s2);
  },
};

// ========== DATA COMPARATOR ==========
const DataComparator = {
  compareStrategyPair: (results, strategy1, strategy2) => {
    const res1 = results.filter(r => r.strategy === strategy1);
    const res2 = results.filter(r => r.strategy === strategy2);
    return {
      strategy1: strategy1,
      strategy2: strategy2,
      suppDiff: StatisticsEngine.mean(res1.map(r => r.suppressionPeak)) - 
                StatisticsEngine.mean(res2.map(r => r.suppressionPeak)),
      costRatio: StatisticsEngine.mean(res1.map(r => r.treatedMean)) / 
                 (StatisticsEngine.mean(res2.map(r => r.treatedMean)) || 1),
      aucDiff: StatisticsEngine.mean(res1.map(r => r.suppressionAuc)) - 
               StatisticsEngine.mean(res2.map(r => r.suppressionAuc)),
      count1: res1.length,
      count2: res2.length,
    };
  },
  compareGraphs: (results, graph1, graph2) => {
    const g1 = results.filter(r => r.graph === graph1);
    const g2 = results.filter(r => r.graph === graph2);
    return {
      graph1, graph2,
      avgSupp1: StatisticsEngine.mean(g1.map(r => r.suppressionPeak)),
      avgSupp2: StatisticsEngine.mean(g2.map(r => r.suppressionPeak)),
      variance1: StatisticsEngine.std(g1.map(r => r.suppressionPeak)),
      variance2: StatisticsEngine.std(g2.map(r => r.suppressionPeak)),
    };
  },
  findSignificantDifference: (results, strategy1, strategy2, threshold = 0.05) => {
    const res1 = results.filter(r => r.strategy === strategy1).map(r => r.suppressionPeak);
    const res2 = results.filter(r => r.strategy === strategy2).map(r => r.suppressionPeak);
    if (res1.length < 2 || res2.length < 2) return null;
    
    const m1 = StatisticsEngine.mean(res1);
    const m2 = StatisticsEngine.mean(res2);
    const s1 = StatisticsEngine.std(res1);
    const s2 = StatisticsEngine.std(res2);
    const pooledStd = Math.sqrt(((res1.length - 1) * s1 * s1 + (res2.length - 1) * s2 * s2) / (res1.length + res2.length - 2));
    const t = (m1 - m2) / (pooledStd * Math.sqrt(1 / res1.length + 1 / res2.length));
    return { t, isSignificant: Math.abs(t) > 1.96 };
  },
};

// ========== PERFORMANCE PROFILER ==========
const PerformanceProfiler = {
  profileResult: (result) => {
    return {
      peakEfficiency: result.suppressionPeak / (result.treatedMean || 1),
      aucEfficiency: result.suppressionAuc / (result.treatedMean || 1),
      treatCost: result.treatedMean,
      peakRatio: result.suppressionPeak,
      aucRatio: result.suppressionAuc,
      finalRatio: (result.finalMean || 0) / (result.peakMean || 1),
    };
  },
  compareEfficiency: (results) => {
    return results.map(r => ({
      ...r,
      efficiency: PerformanceProfiler.profileResult(r),
    })).sort((a, b) => b.efficiency.peakEfficiency - a.efficiency.peakEfficiency);
  },
  identifyOutliers: (results, field = 'suppressionPeak', threshold = 2) => {
    const values = results.map(r => r[field]);
    const m = StatisticsEngine.mean(values);
    const s = StatisticsEngine.std(values);
    return results.filter(r => Math.abs((r[field] - m) / (s || 1)) > threshold);
  },
};

// ========== RESULTS CACHE ==========
class ResultsCache {
  constructor(maxSize = 100, ttl = 1800000) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.ttl = ttl;
    this.hits = 0;
    this.misses = 0;
  }
  
  generateKey(config, graphKey, strategy) {
    return `${JSON.stringify(config)}-${graphKey}-${strategy}`;
  }
  
  set(config, graphKey, strategy, result) {
    const key = this.generateKey(config, graphKey, strategy);
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, { result, timestamp: Date.now() });
  }
  
  get(config, graphKey, strategy) {
    const key = this.generateKey(config, graphKey, strategy);
    if (this.cache.has(key)) {
      const entry = this.cache.get(key);
      if (Date.now() - entry.timestamp < this.ttl) {
        this.hits++;
        return entry.result;
      }
      this.cache.delete(key);
    }
    this.misses++;
    return null;
  }
  
  getMetrics() {
    return {
      size: this.cache.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: (this.hits / (this.hits + this.misses)) || 0,
    };
  }
  
  clear() {
    this.cache.clear();
  }
}

// ========== EXPERIMENT VALIDATOR ==========
const ExperimentValidator = {
  validateConfig: (config) => {
    const errors = [];
    if (config.timesteps < 20 || config.timesteps > 200) errors.push("Timesteps must be 20-200");
    if (config.treatBudget < 1 || config.treatBudget > 10) errors.push("Treatment budget must be 1-10");
    if (config.runs < 3 || config.runs > 50) errors.push("Runs must be 3-50");
    if (config.infectionThreshold < 0.2 || config.infectionThreshold > 0.9) errors.push("Infection threshold 0.2-0.9");
    return { isValid: errors.length === 0, errors };
  },
  validateResults: (rows) => {
    const issues = [];
    rows.forEach((row, idx) => {
      if (isNaN(row.suppressionPeak)) issues.push(`Row ${idx}: Invalid suppression peak`);
      if (row.treatedMean < 0) issues.push(`Row ${idx}: Negative treated mean`);
      if (row.peakMean <= 0) issues.push(`Row ${idx}: Invalid peak mean`);
    });
    return { isValid: issues.length === 0, issues };
  },
  sanitizeResult: (result) => {
    return {
      ...result,
      suppressionPeak: Math.max(0, Math.min(100, result.suppressionPeak || 0)),
      suppressionAuc: Math.max(0, Math.min(100, result.suppressionAuc || 0)),
      treatedMean: Math.max(0, result.treatedMean || 0),
      peakMean: Math.max(0, result.peakMean || 0),
      finalMean: Math.max(0, result.finalMean || 0),
    };
  },
};

// ========== METRICS COMPUTER ==========
const MetricsComputer = {
  computeGraphMetrics: (results, graphKey) => {
    const graphResults = results.filter(r => r.graph === graphKey);
    if (!graphResults.length) return null;
    
    const suppressions = graphResults.map(r => r.suppressionPeak);
    const aucs = graphResults.map(r => r.suppressionAuc);
    const costs = graphResults.map(r => r.treatedMean);
    
    return {
      graph: graphKey,
      avgSuppression: StatisticsEngine.mean(suppressions),
      stdSuppression: StatisticsEngine.std(suppressions),
      avgAuc: StatisticsEngine.mean(aucs),
      avgCost: StatisticsEngine.mean(costs),
      strategies: graphResults.length,
      suppRange: [Math.min(...suppressions), Math.max(...suppressions)],
    };
  },
  computeStrategyMetrics: (results, strategy) => {
    const stratResults = results.filter(r => r.strategy === strategy);
    if (!stratResults.length) return null;
    
    const suppressions = stratResults.map(r => r.suppressionPeak);
    const costs = stratResults.map(r => r.treatedMean);
    
    return {
      strategy,
      avgSuppression: StatisticsEngine.mean(suppressions),
      consistency: 100 - StatisticsEngine.std(suppressions),
      avgCost: StatisticsEngine.mean(costs),
      graphs: stratResults.length,
      suppRange: [Math.min(...suppressions), Math.max(...suppressions)],
      costEfficiency: StatisticsEngine.mean(suppressions) / (StatisticsEngine.mean(costs) || 1),
    };
  },
  computeCorrelations: (results) => {
    const suppressions = results.map(r => r.suppressionPeak);
    const costs = results.map(r => r.treatedMean);
    const peaks = results.map(r => r.peakMean);
    
    return {
      suppVsCost: StatisticsEngine.correlation(suppressions, costs),
      suppVsPeak: StatisticsEngine.correlation(suppressions, peaks),
      costVsPeak: StatisticsEngine.correlation(costs, peaks),
    };
  },
};

// ========== REPORT GENERATOR ==========
const ReportGenerator = {
  generateSummary: (results) => {
    if (!results.length) return null;
    
    const suppressions = results.map(r => r.suppressionPeak);
    const costs = results.map(r => r.treatedMean);
    
    return {
      totalTests: results.length,
      avgSuppression: StatisticsEngine.mean(suppressions).toFixed(2),
      bestSuppression: Math.max(...suppressions).toFixed(2),
      worstSuppression: Math.min(...suppressions).toFixed(2),
      avgCost: StatisticsEngine.mean(costs).toFixed(2),
      stats: StatisticsEngine.quartiles(suppressions),
      timestamp: new Date().toISOString(),
    };
  },
  generateDetailedReport: (results, config) => {
    const graphs = new Set(results.map(r => r.graph));
    const strategies = new Set(results.map(r => r.strategy));
    
    return {
      metadata: {
        timestamp: new Date().toISOString(),
        config,
        totalResults: results.length,
        uniqueGraphs: graphs.size,
        uniqueStrategies: strategies.size,
      },
      graphMetrics: Array.from(graphs).map(g => MetricsComputer.computeGraphMetrics(results, g)),
      strategyMetrics: Array.from(strategies).map(s => MetricsComputer.computeStrategyMetrics(results, s)),
      summary: ReportGenerator.generateSummary(results),
      correlations: MetricsComputer.computeCorrelations(results),
      topPerformers: BenchmarkAnalyzer.rankByPerformance(results).slice(0, 5),
    };
  },
};

// ========== CHART DATA FORMATTER ==========
const ChartDataFormatter = {
  formatForBarChart: (results, field = 'suppressionPeak') => {
    return results.map(r => ({
      name: `${r.graph.substring(0, 8)}/${r.strategyLabel.substring(0, 10)}`,
      value: r[field],
    }));
  },
  formatStrategyComparison: (results) => {
    const strategies = new Set(results.map(r => r.strategy));
    const graphs = new Set(results.map(r => r.graph));
    
    return {
      labels: Array.from(graphs),
      datasets: Array.from(strategies).map(s => ({
        label: s,
        data: Array.from(graphs).map(g => {
          const result = results.find(r => r.strategy === s && r.graph === g);
          return result ? result.suppressionPeak : 0;
        }),
      })),
    };
  },
  formatDistribution: (results, field = 'suppressionPeak') => {
    const values = results.map(r => r[field]);
    const bins = 10;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / bins || 1;
    
    const histogram = Array(bins).fill(0);
    values.forEach(v => {
      const binIdx = Math.min(bins - 1, Math.floor((v - min) / binWidth));
      histogram[binIdx]++;
    });
    
    return histogram.map((count, i) => ({
      range: `${(min + i * binWidth).toFixed(1)}-${(min + (i + 1) * binWidth).toFixed(1)}`,
      count,
    }));
  },
};

// ========== SESSION TRACKER ==========
class SessionTracker {
  constructor() {
    this.sessions = [];
    this.currentSession = null;
  }
  
  startSession(config) {
    this.currentSession = {
      id: `session_${Date.now()}`,
      config,
      startTime: Date.now(),
      results: [],
      status: 'active',
    };
    this.sessions.push(this.currentSession);
    return this.currentSession.id;
  }
  
  addResult(result) {
    if (this.currentSession) {
      this.currentSession.results.push(result);
    }
  }
  
  endSession() {
    if (this.currentSession) {
      this.currentSession.status = 'completed';
      this.currentSession.endTime = Date.now();
      this.currentSession.duration = this.currentSession.endTime - this.currentSession.startTime;
    }
  }
  
  getSessions() {
    return this.sessions;
  }
  
  getSessionStats() {
    return this.sessions.map(s => ({
      id: s.id,
      duration: s.duration,
      resultCount: s.results.length,
      status: s.status,
    }));
  }
}

// ========== CORRELATION ANALYZER ==========
const CorrelationAnalyzer = {
  analyzeMetricCorrelations: (results) => {
    const metrics = {
      suppression: results.map(r => r.suppressionPeak),
      cost: results.map(r => r.treatedMean),
      peak: results.map(r => r.peakMean),
      final: results.map(r => r.finalMean),
      auc: results.map(r => r.suppressionAuc),
    };
    
    const correlations = {};
    const metricNames = Object.keys(metrics);
    for (let i = 0; i < metricNames.length; i++) {
      for (let j = i + 1; j < metricNames.length; j++) {
        const key = `${metricNames[i]}_vs_${metricNames[j]}`;
        correlations[key] = StatisticsEngine.correlation(metrics[metricNames[i]], metrics[metricNames[j]]);
      }
    }
    
    return correlations;
  },
  findStrongCorrelations: (results, threshold = 0.7) => {
    const correlations = CorrelationAnalyzer.analyzeMetricCorrelations(results);
    return Object.entries(correlations).filter(([_, corr]) => Math.abs(corr) > threshold);
  },
};

// ========== EXPORT UTILITIES ==========
const ExportUtilities = {
  toMarkdownTable: (results) => {
    let md = "| Graph | Strategy | Peak Supp | Avg Cost | AUC Supp |\n";
    md += "|-------|----------|-----------|----------|----------|\n";
    results.forEach(r => {
      md += `| ${r.graph} | ${r.strategyLabel} | ${r.suppressionPeak.toFixed(2)}% | ${r.treatedMean.toFixed(2)} | ${r.suppressionAuc.toFixed(2)}% |\n`;
    });
    return md;
  },
  toLatexTable: (results) => {
    let latex = "\\begin{tabular}{lllll}\n";
    latex += "Graph & Strategy & Peak Supp & Avg Cost & AUC Supp \\\\\n";
    latex += "\\hline\n";
    results.forEach(r => {
      latex += `${r.graph} & ${r.strategyLabel} & ${r.suppressionPeak.toFixed(2)}\\% & ${r.treatedMean.toFixed(2)} & ${r.suppressionAuc.toFixed(2)}\\% \\\\\n`;
    });
    latex += "\\end{tabular}\n";
    return latex;
  },
};

// ========== CUSTOM REACT HOOKS ==========
const useBenchmarkCache = (maxSize = 100) => {
  const cacheRef = useRef(new ResultsCache(maxSize));
  return cacheRef.current;
};

const useBenchmarkResults = (results) => {
  return useMemo(() => ({
    summary: ReportGenerator.generateSummary(results),
    ranked: BenchmarkAnalyzer.rankByPerformance(results),
    byGraph: BenchmarkAnalyzer.groupByGraph(results),
    correlations: MetricsComputer.computeCorrelations(results),
  }), [results]);
};

const useStrategyComparison = (results, strategy1, strategy2) => {
  return useMemo(() => {
    if (!results.length) return null;
    return DataComparator.compareStrategyPair(results, strategy1, strategy2);
  }, [results, strategy1, strategy2]);
};

// ========== DEBUG UTILITIES ==========
const DebugUtils = {
  logBenchmark: (message, data, level = 'info') => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${level.toUpperCase()}] ${message}`, data);
  },
  summarizeBenchmark: (results) => {
    if (!results.length) return "No results";
    return {
      totalTests: results.length,
      avgSuppression: StatisticsEngine.mean(results.map(r => r.suppressionPeak)).toFixed(2),
      bestResult: BenchmarkAnalyzer.rankByPerformance(results)[0],
    };
  },
};

// ========== ANALYSIS CONFIGURATION ==========
const AnalysisConfig = {
  presets: {
    quick: { timesteps: 30, treatBudget: 2, runs: 3, infectionThreshold: 0.5 },
    standard: { timesteps: 60, treatBudget: 3, runs: 8, infectionThreshold: 0.5 },
    thorough: { timesteps: 100, treatBudget: 4, runs: 20, infectionThreshold: 0.5 },
    custom: (ts, tb, r, it) => ({ timesteps: ts, treatBudget: tb, runs: r, infectionThreshold: it }),
  },
  validatePreset: (name) => AnalysisConfig.presets[name] || AnalysisConfig.presets.standard,
};

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
