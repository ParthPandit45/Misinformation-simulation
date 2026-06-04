import { useState, useEffect, useRef, useCallback, useMemo, memo } from "react";
import {
  LineChart as ReLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
  BarChart as ReBarChart,
  Bar,
  Cell
} from "recharts";
import WorkspaceShell from "./layout/WorkspaceShell.jsx";
import ResearchWorkbenchPanel from "./research/ResearchWorkbenchPanel.jsx";
import ZeroShotPanel from "./research/ZeroShotPanel.jsx";
import { evaluateAcademicBatch, toCSV } from "../../../utils/academicEvaluation.js";
import { predictGNN, simulateStrategyAPI } from "../../../services/api.js";
import { GRAPH_META, STRATEGIES, ACTIONS, SUMMARY, loadSummaryOverride } from "../data/dashboardData.js";

// ─────────────────────────────────────────────────────────────────────────────
// ADVANCED ANALYTICS & DATA PROCESSING UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Advanced statistics calculator for comprehensive data analysis
 * Computes multiple statistical measures for belief propagation metrics
 */
const StatisticsEngine = {
  // Compute percentile values from array
  percentile: (arr, p) => {
    if (!arr.length) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    const weight = idx % 1;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  },

  // Calculate quartiles for distribution analysis
  quartiles: (arr) => ({
    q0: Math.min(...arr),
    q1: StatisticsEngine.percentile(arr, 25),
    q2: StatisticsEngine.percentile(arr, 50),
    q3: StatisticsEngine.percentile(arr, 75),
    q4: Math.max(...arr),
  }),

  // Compute skewness of distribution
  skewness: (arr) => {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((acc, v) => acc + (v - mean) ** 2, 0) / arr.length;
    const std = Math.sqrt(variance);
    if (std === 0) return 0;
    const m3 = arr.reduce((acc, v) => acc + ((v - mean) ** 3), 0) / arr.length;
    return m3 / (std ** 3);
  },

  // Calculate kurtosis for tail analysis
  kurtosis: (arr) => {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((acc, v) => acc + (v - mean) ** 2, 0) / arr.length;
    const std = Math.sqrt(variance);
    if (std === 0) return 0;
    const m4 = arr.reduce((acc, v) => acc + ((v - mean) ** 4), 0) / arr.length;
    return m4 / (variance ** 2) - 3;
  },

  // Compute coefficient of variation for normalized spread
  coefficientOfVariation: (arr) => {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    if (mean === 0) return 0;
    const std = Math.sqrt(arr.reduce((acc, v) => acc + (v - mean) ** 2, 0) / arr.length);
    return (std / mean) * 100;
  },

  // Moving average computation
  movingAverage: (arr, windowSize) => {
    if (windowSize <= 0 || windowSize > arr.length) return arr;
    const result = [];
    for (let i = 0; i < arr.length; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2));
      const end = Math.min(arr.length, i + Math.ceil(windowSize / 2));
      const window = arr.slice(start, end);
      result.push(window.reduce((a, b) => a + b, 0) / window.length);
    }
    return result;
  },

  // Exponential moving average for smoothing
  exponentialMovingAverage: (arr, alpha = 0.3) => {
    if (!arr.length) return [];
    const result = [arr[0]];
    for (let i = 1; i < arr.length; i++) {
      result.push(alpha * arr[i] + (1 - alpha) * result[i - 1]);
    }
    return result;
  },

  // Autocorrelation for temporal dependency
  autocorrelation: (arr, lag = 1) => {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const c0 = arr.reduce((acc, v) => acc + (v - mean) ** 2, 0) / arr.length;
    if (c0 === 0) return 0;
    const c_lag = arr.slice(lag).reduce((acc, v, i) => acc + (v - mean) * (arr[i] - mean), 0) / arr.length;
    return c_lag / c0;
  },
};

/**
 * Performance analysis utilities for strategy comparison
 */
const PerformanceAnalyzer = {
  // Calculate suppression effectiveness
  calculateSuppression: (baselineMetric, strategicMetric) => {
    if (baselineMetric === 0) return 0;
    return ((baselineMetric - strategicMetric) / baselineMetric) * 100;
  },

  // Rank strategies by performance
  rankStrategies: (metrics) => {
    return Object.entries(metrics)
      .sort(([, a], [, b]) => b - a)
      .map(([strategy, score], rank) => ({ rank: rank + 1, strategy, score }));
  },

  // Calculate relative improvement over baseline
  relativeImprovement: (baseline, alternatives) => {
    return Object.fromEntries(
      Object.entries(alternatives).map(([key, value]) => [
        key,
        baseline > 0 ? ((baseline - value) / baseline) * 100 : 0,
      ])
    );
  },

  // Win-loss-tie comparison
  compareStrategies: (strategy1Metrics, strategy2Metrics) => {
    let wins = 0, losses = 0, ties = 0;
    Object.keys(strategy1Metrics).forEach((key) => {
      const s1 = strategy1Metrics[key];
      const s2 = strategy2Metrics[key];
      if (s1 < s2) wins++;
      else if (s1 > s2) losses++;
      else ties++;
    });
    return { wins, losses, ties, winRate: (wins / (wins + losses + ties)) * 100 };
  },

  // Calculate dominance score (Pareto efficiency)
  calculateDominanceScore: (metrics, weights) => {
    return Object.entries(metrics).reduce((score, [key, value]) => {
      return score + (value * (weights[key] || 1));
    }, 0);
  },
};

/**
 * Network graph analysis utilities
 */
const GraphAnalyzer = {
  // Calculate node centrality measures
  calculateCentrality: (nodes, edges) => {
    const adj = nodes.map(() => []);
    edges.forEach(([a, b]) => {
      adj[a].push(b);
      adj[b].push(a);
    });

    const centrality = nodes.map((n, idx) => {
      const degree = adj[idx].length;
      const avgBelief = adj[idx].length > 0
        ? adj[idx].reduce((sum, nei) => sum + nodes[nei].belief, 0) / adj[idx].length
        : 0;
      return { idx, degree, avgBelief, closenessApprox: 1 / (degree + 1) };
    });

    return centrality;
  },

  // Identify influential clusters
  identifyInfluentialClusters: (nodes, edges, beliefThreshold = 0.5) => {
    const adj = nodes.map(() => []);
    edges.forEach(([a, b]) => {
      adj[a].push(b);
      adj[b].push(a);
    });

    const highBelief = nodes.filter(n => n.belief > beliefThreshold);
    const clusters = [];
    const visited = new Set();

    highBelief.forEach(node => {
      if (!visited.has(node.id)) {
        const cluster = [];
        const queue = [node.id];

        while (queue.length > 0) {
          const current = queue.shift();
          if (visited.has(current)) continue;
          visited.add(current);
          cluster.push(current);

          adj[current].forEach(neighbor => {
            if (!visited.has(neighbor) && nodes[neighbor].belief > beliefThreshold) {
              queue.push(neighbor);
            }
          });
        }

        if (cluster.length > 0) {
          clusters.push({
            size: cluster.length,
            avgBelief: cluster.reduce((sum, idx) => sum + nodes[idx].belief, 0) / cluster.length,
            nodes: cluster,
          });
        }
      }
    });

    return clusters.sort((a, b) => b.size - a.size);
  },

  // Calculate graph connectivity metrics
  calculateConnectivityMetrics: (nodes, edges) => {
    const n = nodes.length;
    const m = edges.length;
    const density = n > 1 ? (2 * m) / (n * (n - 1)) : 0;

    const adj = nodes.map(() => []);
    edges.forEach(([a, b]) => {
      adj[a].push(b);
      adj[b].push(a);
    });

    const degrees = adj.map(neighbors => neighbors.length);
    const avgDegree = degrees.reduce((a, b) => a + b, 0) / n;
    const maxDegree = Math.max(...degrees);
    const minDegree = Math.min(...degrees);

    return { density, avgDegree, maxDegree, minDegree, nodes: n, edges: m };
  },

  // Compute infection spread velocity
  calculateSpreadVelocity: (snapshots) => {
    if (snapshots.length < 2) return [];
    const velocities = [];

    for (let t = 1; t < snapshots.length; t++) {
      const infected = (snap) =>
        snap.nodes.filter(n => n.belief > 0.5 && !n.treated).length;
      const delta = infected(snapshots[t]) - infected(snapshots[t - 1]);
      velocities.push({ timestep: t, velocity: delta });
    }

    return velocities;
  },
};

/**
 * Budget and resource optimization utilities
 */
const BudgetOptimizer = {
  // Analyze budget efficiency
  analyzeBudgetEfficiency: (budget, suppressionAchieved) => {
    return suppressionAchieved / budget;
  },

  // Calculate ROI for different strategies
  calculateROI: (budgetSpent, suppressionGain) => {
    return budgetSpent > 0 ? (suppressionGain / budgetSpent) * 100 : 0;
  },

  // Predict optimal budget allocation
  optimizeBudgetAllocation: (strategyEfficiency) => {
    const totalEfficiency = Object.values(strategyEfficiency).reduce((a, b) => a + b, 0);
    return Object.fromEntries(
      Object.entries(strategyEfficiency).map(([strategy, efficiency]) => [
        strategy,
        (efficiency / totalEfficiency) * 100,
      ])
    );
  },

  // Time-series budget forecasting
  forecastBudgetNeed: (historicalUsage, targetSuppression) => {
    if (!historicalUsage.length) return 0;
    const avgUsagePerSuppression =
      historicalUsage.reduce((sum, u) => sum + u, 0) / historicalUsage.length;
    return targetSuppression * avgUsagePerSuppression;
  },
};

/**
 * Data filtering and search utilities
 */
const DataFilter = {
  // Filter snapshots by belief range
  filterByBelief: (snapshots, minBelief, maxBelief) => {
    return snapshots.map(snap => ({
      ...snap,
      nodes: snap.nodes.filter(n => n.belief >= minBelief && n.belief <= maxBelief),
    }));
  },

  // Find timesteps matching criteria
  findMatchingTimesteps: (snapshots, criteria) => {
    return snapshots
      .map((snap, idx) => {
        const infected = snap.nodes.filter(n => n.belief > 0.5 && !n.treated).length;
        const treated = snap.nodes.filter(n => n.treated).length;
        if (
          (!criteria.minInfected || infected >= criteria.minInfected) &&
          (!criteria.maxInfected || infected <= criteria.maxInfected) &&
          (!criteria.minTreated || treated >= criteria.minTreated) &&
          (!criteria.maxTreated || treated <= criteria.maxTreated)
        ) {
          return idx;
        }
        return -1;
      })
      .filter(idx => idx !== -1);
  },

  // Search strategies by performance threshold
  searchByPerformance: (results, minSuppression, maxSuppression) => {
    return results.filter(
      r =>
        r.suppressionPeak >= minSuppression &&
        r.suppressionPeak <= maxSuppression
    );
  },
};

/**
 * Time-series analysis tools
 */
const TimeSeriesAnalyzer = {
  // Detect peaks in time series
  detectPeaks: (values, prominence = 1) => {
    const peaks = [];
    for (let i = 1; i < values.length - 1; i++) {
      if (values[i] > values[i - 1] && values[i] > values[i + 1]) {
        const p = (values[i] - values[i - 1]) + (values[i] - values[i + 1]);
        if (p >= prominence) peaks.push({ index: i, value: values[i], prominence: p });
      }
    }
    return peaks;
  },

  // Detect valleys (troughs)
  detectValleys: (values, prominence = 1) => {
    const valleys = [];
    for (let i = 1; i < values.length - 1; i++) {
      if (values[i] < values[i - 1] && values[i] < values[i + 1]) {
        const p = (values[i - 1] - values[i]) + (values[i + 1] - values[i]);
        if (p >= prominence) valleys.push({ index: i, value: values[i], prominence: p });
      }
    }
    return valleys;
  },

  // Calculate inflection points
  calculateInflectionPoints: (values) => {
    const inflections = [];
    for (let i = 1; i < values.length - 1; i++) {
      const d1 = values[i] - values[i - 1];
      const d2 = values[i + 1] - values[i];
      if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) {
        inflections.push({ index: i, value: values[i] });
      }
    }
    return inflections;
  },

  // Linear trend estimation
  estimateTrend: (values) => {
    const n = values.length;
    const sumX = (n * (n - 1)) / 2;
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = values.reduce((sum, v, i) => sum + v * i, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return { slope, intercept, rSquared: null };
  },

  // Volatility analysis
  calculateVolatility: (values) => {
    const diffs = [];
    for (let i = 1; i < values.length; i++) {
      diffs.push(Math.abs(values[i] - values[i - 1]));
    }
    return {
      mean: diffs.reduce((a, b) => a + b, 0) / diffs.length,
      max: Math.max(...diffs),
      variance: StatisticsEngine.coefficientOfVariation(diffs),
    };
  },
};

/**
 * Performance benchmarking utilities
 */
const Benchmarker = {
  // Measure execution time
  measureTime: async (fn) => {
    const start = performance.now();
    const result = await fn();
    const end = performance.now();
    return { result, duration: end - start };
  },

  // Compare performance across strategies
  benchmarkStrategies: (strategies, testFn) => {
    return Object.fromEntries(
      strategies.map(strategy => [
        strategy,
        {
          ...Benchmarker.measureTime(() => testFn(strategy)),
          strategy,
        },
      ])
    );
  },
};

/**
 * Data export and serialization utilities
 */
const DataExporter = {
  // Convert metrics to JSON with formatting
  toFormattedJSON: (data, pretty = true) => {
    return JSON.stringify(data, null, pretty ? 2 : 0);
  },

  // Create downloadable CSV from matrix
  createMatrixCSV: (matrix, headers) => {
    const lines = [headers.join(",")];
    matrix.forEach(row => {
      lines.push(row.map(v => (typeof v === "string" ? `"${v}"` : v)).join(","));
    });
    return lines.join("\n");
  },

  // Serialize network graph to standard format
  serializeGraph: (nodes, edges) => {
    return {
      nodes: nodes.map(n => ({
        id: n.id,
        x: n.x,
        y: n.y,
        belief: n.belief,
        treated: n.treated,
        isSeed: n.isSeed,
      })),
      edges: edges,
      metadata: {
        nodeCount: nodes.length,
        edgeCount: edges.length,
        timestamp: new Date().toISOString(),
      },
    };
  },
};

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

// ─────────────────────────────────────────────────────────────────────────────
// CACHING & SESSION MANAGEMENT
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Lightweight cache system for computed results
 * Reduces redundant calculations for frequently accessed data
 */
class ComputationCache {
  constructor(maxSize = 100) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.hits = 0;
    this.misses = 0;
  }

  key(...args) {
    return JSON.stringify(args);
  }

  set(key, value, ttl = null) {
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, { value, timestamp: Date.now(), ttl });
  }

  get(key) {
    const entry = this.cache.get(key);
    if (!entry) {
      this.misses++;
      return null;
    }
    if (entry.ttl && Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      this.misses++;
      return null;
    }
    this.hits++;
    return entry.value;
  }

  has(key) {
    const entry = this.cache.get(key);
    return entry && (!entry.ttl || Date.now() - entry.timestamp < entry.ttl);
  }

  clear() {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  stats() {
    const total = this.hits + this.misses;
    return {
      size: this.cache.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? (this.hits / total) * 100 : 0,
    };
  }
}

/**
 * Session state manager for user preferences and history
 */
class SessionManager {
  constructor(storageKey = "dashboard-session") {
    this.storageKey = storageKey;
    this.load();
  }

  load() {
    try {
      const stored = localStorage.getItem(this.storageKey);
      this.state = stored ? JSON.parse(stored) : {};
    } catch {
      this.state = {};
    }
  }

  save() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.state));
    } catch {
      console.warn("Failed to save session state");
    }
  }

  get(key, defaultValue = null) {
    return this.state[key] ?? defaultValue;
  }

  set(key, value) {
    this.state[key] = value;
    this.save();
  }

  update(updates) {
    this.state = { ...this.state, ...updates };
    this.save();
  }

  clear() {
    this.state = {};
    localStorage.removeItem(this.storageKey);
  }

  getHistory(key) {
    return this.get(`history_${key}`, []);
  }

  addToHistory(key, value, maxLength = 20) {
    const history = this.getHistory(key);
    const newHistory = [value, ...history.filter(v => v !== value)].slice(0, maxLength);
    this.set(`history_${key}`, newHistory);
  }
}

/**
 * Error handling and logging system
 */
const ErrorHandler = {
  log: (message, context = {}, level = "info") => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${level.toUpperCase()}] ${message}`, context);
  },

  error: (message, error, context = {}) => {
    ErrorHandler.log(message, { ...context, error: error?.message }, "error");
  },

  warn: (message, context = {}) => {
    ErrorHandler.log(message, context, "warn");
  },

  debug: (message, context = {}) => {
    if (process.env.DEBUG) {
      ErrorHandler.log(message, context, "debug");
    }
  },

  trackError: (errorName, details) => {
    ErrorHandler.log(`Error: ${errorName}`, details, "error");
  },
};

/**
 * Validation utilities for data integrity
 */
const Validators = {
  isValidGraph: (nodes, edges) => {
    return (
      Array.isArray(nodes) &&
      Array.isArray(edges) &&
      nodes.length > 0 &&
      nodes.every(n => typeof n.id === "number" && typeof n.belief === "number") &&
      edges.every(e => Array.isArray(e) && e.length === 2)
    );
  },

  isValidSnapshot: (snap) => {
    return snap && Validators.isValidGraph(snap.nodes, snap.edges);
  },

  validateMetrics: (metrics) => {
    return Object.entries(metrics).every(
      ([key, value]) =>
        typeof value === "number" &&
        !isNaN(value) &&
        isFinite(value)
    );
  },

  sanitizeNumber: (value, min = -Infinity, max = Infinity, defaultValue = 0) => {
    const num = Number(value);
    if (!isFinite(num)) return defaultValue;
    return Math.max(min, Math.min(max, num));
  },

  validateBeliefRange: (belief) => {
    return Validators.sanitizeNumber(belief, 0, 1);
  },
};

/**
 * Metrics computation engine
 */
const MetricsEngine = {
  // Calculate node statistics from snapshot
  calculateNodeStats: (nodes) => {
    const infected = nodes.filter(n => n.belief > 0.5 && !n.treated).length;
    const treated = nodes.filter(n => n.treated).length;
    const avgBelief = nodes.reduce((sum, n) => sum + n.belief, 0) / nodes.length;
    const maxBelief = Math.max(...nodes.map(n => n.belief));
    const minBelief = Math.min(...nodes.map(n => n.belief));

    return { infected, treated, avgBelief, maxBelief, minBelief };
  },

  // Calculate aggregate metrics from snapshots
  calculateAggregateMetrics: (snapshots) => {
    if (!snapshots.length) return null;

    const infectedCurve = snapshots.map(
      s => s.nodes.filter(n => n.belief > 0.5 && !n.treated).length
    );
    const peakInfected = Math.max(...infectedCurve);
    const finalInfected = infectedCurve[infectedCurve.length - 1];
    const avgInfected = infectedCurve.reduce((a, b) => a + b, 0) / infectedCurve.length;

    const treatedCurve = snapshots.map(s => s.nodes.filter(n => n.treated).length);
    const totalTreated = treatedCurve[treatedCurve.length - 1];

    return {
      peakInfected,
      finalInfected,
      avgInfected,
      totalTreated,
      aucInfected: infectedCurve.reduce((a, b) => a + b, 0),
    };
  },

  // Calculate suppression metrics relative to baseline
  calculateSuppressionMetrics: (baselineMetrics, strategicMetrics) => {
    return {
      peakSuppression: PerformanceAnalyzer.calculateSuppression(
        baselineMetrics.peakInfected,
        strategicMetrics.peakInfected
      ),
      aucSuppression: PerformanceAnalyzer.calculateSuppression(
        baselineMetrics.aucInfected,
        strategicMetrics.aucInfected
      ),
      finalSuppression: PerformanceAnalyzer.calculateSuppression(
        baselineMetrics.finalInfected,
        strategicMetrics.finalInfected
      ),
    };
  },

  // Cost-benefit analysis
  calculateCostBenefit: (budgetUsed, suppression) => {
    return {
      roi: BudgetOptimizer.calculateROI(budgetUsed, suppression),
      efficiency: BudgetOptimizer.analyzeBudgetEfficiency(budgetUsed, suppression),
      costPerPercent: budgetUsed / Math.max(suppression, 0.01),
    };
  },
};

// Generate suppression curves (seeded deterministic)
function seededRand(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 4294967296; };
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPUTATIONAL HELPERS & OPTIMIZATION TECHNIQUES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Curve interpolation and smoothing utilities
 */
const CurveUtils = {
  // Cubic Hermite interpolation for smoother curves
  cubicHermite: (p0, p1, m0, m1, t) => {
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
  },

  // Catmull-Rom spline interpolation
  catmullRom: (p0, p1, p2, p3, t) => {
    const t2 = t * t;
    const t3 = t2 * t;
    const v0 = (p2 - p0) * 0.5;
    const v1 = (p3 - p1) * 0.5;
    return (2 * p1 - 2 * p2 + v0 + v1) * t3 +
           (-3 * p1 + 3 * p2 - 2 * v0 - v1) * t2 +
           v0 * t + p1;
  },

  // Bezier curve interpolation
  bezier: (controlPoints, t) => {
    const n = controlPoints.length - 1;
    let result = 0;
    for (let i = 0; i <= n; i++) {
      const binomial = CurveUtils.binomialCoefficient(n, i);
      const basis = binomial * Math.pow(1 - t, n - i) * Math.pow(t, i);
      result += basis * controlPoints[i];
    }
    return result;
  },

  binomialCoefficient: (n, k) => {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    let result = 1;
    for (let i = 0; i < k; i++) {
      result = result * (n - i) / (i + 1);
    }
    return result;
  },

  // Adaptive smoothing based on signal characteristics
  adaptiveSmoothing: (values, sensitivity = 0.5) => {
    const variance = StatisticsEngine.coefficientOfVariation(values);
    const alpha = Math.min(0.5, Math.max(0.1, sensitivity * variance / 100));
    return StatisticsEngine.exponentialMovingAverage(values, alpha);
  },

  // Noise reduction using median filter
  medianFilter: (values, windowSize = 3) => {
    const result = [];
    const halfWindow = Math.floor(windowSize / 2);
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - halfWindow);
      const end = Math.min(values.length, i + halfWindow + 1);
      const window = values.slice(start, end).sort((a, b) => a - b);
      result.push(window[Math.floor(window.length / 2)]);
    }
    return result;
  },
};

/**
 * Sampling and distribution utilities
 */
const SamplingUtils = {
  // Weighted random sampling
  weightedSample: (items, weights, rng) => {
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    let random = rng() * totalWeight;
    for (let i = 0; i < items.length; i++) {
      random -= weights[i];
      if (random < 0) return items[i];
    }
    return items[items.length - 1];
  },

  // Stratified sampling
  stratifiedSample: (items, strata, count, rng) => {
    const groups = {};
    items.forEach((item, idx) => {
      const key = strata(item);
      if (!groups[key]) groups[key] = [];
      groups[key].push(item);
    });

    const sampled = [];
    const samplesPerGroup = Math.ceil(count / Object.keys(groups).length);
    Object.values(groups).forEach(group => {
      for (let i = 0; i < Math.min(samplesPerGroup, group.length); i++) {
        const idx = Math.floor(rng() * group.length);
        sampled.push(group[idx]);
      }
    });
    return sampled.slice(0, count);
  },

  // Rejection sampling
  rejectionSample: (targetDist, proposalDist, M, rng) => {
    let attempts = 0;
    const maxAttempts = 1000;
    while (attempts < maxAttempts) {
      const sample = proposalDist(rng);
      const u = rng();
      if (u < targetDist(sample) / (M * proposalDist(sample))) {
        return sample;
      }
      attempts++;
    }
    return null;
  },
};

/**
 * Network generation utilities
 */
const NetworkGenerators = {
  // Watts-Strogatz small-world network
  generateSmallWorld: (n, k, beta, rng) => {
    const nodes = [];
    const edges = [];
    const edgeSet = new Set();

    // Ring lattice
    for (let i = 0; i < n; i++) {
      nodes.push({ id: i, x: 0.5, y: 0.5, degree: 0 });
    }

    for (let i = 0; i < n; i++) {
      for (let j = 1; j <= k / 2; j++) {
        const target = (i + j) % n;
        const key = `${Math.min(i, target)}-${Math.max(i, target)}`;
        if (!edgeSet.has(key)) {
          edgeSet.add(key);
          edges.push([i, target]);
          nodes[i].degree++;
          nodes[target].degree++;
        }
      }
    }

    // Rewiring
    for (let i = 0; i < edges.length; i++) {
      if (rng() < beta) {
        const [u, v] = edges[i];
        let newV = Math.floor(rng() * n);
        let attempts = 0;
        while ((newV === u || newV === v || edgeSet.has(`${Math.min(u, newV)}-${Math.max(u, newV)}`)) && attempts < 10) {
          newV = Math.floor(rng() * n);
          attempts++;
        }
        if (newV !== u && newV !== v) {
          edges[i][1] = newV;
        }
      }
    }

    return { nodes, edges };
  },

  // Random regular graph
  generateRandomRegular: (n, k, rng) => {
    if ((n * k) % 2 !== 0) throw new Error("n*k must be even");
    
    const nodes = [];
    const stubs = [];

    for (let i = 0; i < n; i++) {
      nodes.push({ id: i, x: 0.5, y: 0.5, degree: 0 });
      for (let j = 0; j < k; j++) {
        stubs.push(i);
      }
    }

    const edges = [];
    const edgeSet = new Set();
    while (stubs.length > 0) {
      const i1 = Math.floor(rng() * stubs.length);
      const i2 = Math.floor(rng() * stubs.length);
      if (i1 !== i2) {
        const u = stubs[i1];
        const v = stubs[i2];
        const key = `${Math.min(u, v)}-${Math.max(u, v)}`;
        if (!edgeSet.has(key)) {
          edgeSet.add(key);
          edges.push([u, v]);
          nodes[u].degree++;
          nodes[v].degree++;
          stubs.splice(Math.max(i1, i2), 1);
          stubs.splice(Math.min(i1, i2), 1);
        }
      }
    }

    return { nodes, edges };
  },

  // Geometric random graph
  generateGeometricRandom: (n, radius, rng) => {
    const nodes = [];
    const edges = [];
    const edgeSet = new Set();

    for (let i = 0; i < n; i++) {
      nodes.push({
        id: i,
        x: rng(),
        y: rng(),
        degree: 0,
      });
    }

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < radius) {
          edges.push([i, j]);
          nodes[i].degree++;
          nodes[j].degree++;
        }
      }
    }

    return { nodes, edges };
  },
};

/**
 * Clustering and community detection
 */
const ClusteringUtils = {
  // Simple modularity-based clustering
  detectCommunities: (nodes, edges, numClusters = 3) => {
    const n = nodes.length;
    const adj = nodes.map(() => []);
    edges.forEach(([a, b]) => {
      adj[a].push(b);
      adj[b].push(a);
    });

    // K-means on belief values
    let centroids = [];
    for (let i = 0; i < numClusters; i++) {
      centroids.push(i / numClusters);
    }

    let clusters = Array(n).fill(0);
    for (let iter = 0; iter < 10; iter++) {
      // Assign nodes to nearest centroid
      clusters = nodes.map(node =>
        centroids.reduce((closest, _, idx) =>
          Math.abs(node.belief - centroids[idx]) < Math.abs(node.belief - centroids[closest]) ? idx : closest, 0
        )
      );

      // Update centroids
      const newCentroids = [];
      for (let c = 0; c < numClusters; c++) {
        const clusterNodes = nodes.filter((_, idx) => clusters[idx] === c);
        if (clusterNodes.length > 0) {
          newCentroids.push(
            clusterNodes.reduce((sum, n) => sum + n.belief, 0) / clusterNodes.length
          );
        } else {
          newCentroids.push(centroids[c]);
        }
      }
      centroids = newCentroids;
    }

    return clusters;
  },
};

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

// ─────────────────────────────────────────────────────────────────────────────
// ADVANCED STATE MANAGEMENT & MEMOIZATION HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Custom hook for expensive computations with memoization
 */
function useMemoizedComputation(computeFn, dependencies, options = {}) {
  const cacheRef = useRef(new ComputationCache(options.cacheSize || 50));
  const resultRef = useRef(null);
  const depsRef = useRef(null);

  const cacheKey = cacheRef.current.key(...dependencies);
  
  if (depsRef.current !== cacheKey) {
    const cached = cacheRef.current.get(cacheKey);
    if (cached !== null) {
      resultRef.current = cached;
    } else {
      resultRef.current = computeFn();
      cacheRef.current.set(cacheKey, resultRef.current);
    }
    depsRef.current = cacheKey;
  }

  return resultRef.current;
}

/**
 * Custom hook for debounced state updates
 */
function useDebouncedState(initialValue, delay = 300) {
  const [value, setValue] = useState(initialValue);
  const [debouncedValue, setDebouncedValue] = useState(initialValue);
  const timeoutRef = useRef(null);

  useEffect(() => {
    timeoutRef.current = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(timeoutRef.current);
  }, [value, delay]);

  return [debouncedValue, setValue, value];
}

/**
 * Custom hook for throttled callbacks
 */
function useThrottledCallback(callback, delay = 300) {
  const lastRunRef = useRef(Date.now());

  return useCallback((...args) => {
    const now = Date.now();
    if (now - lastRunRef.current >= delay) {
      lastRunRef.current = now;
      callback(...args);
    }
  }, [callback, delay]);
}

/**
 * Custom hook for previous value tracking
 */
function usePrevious(value) {
  const ref = useRef();
  useEffect(() => {
    ref.current = value;
  }, [value]);
  return ref.current;
}

/**
 * Custom hook for async operations with cleanup
 */
function useAsync(asyncFunction, immediate = true) {
  const [state, setState] = useState({
    status: "idle",
    data: null,
    error: null,
  });

  const execute = useCallback(async () => {
    setState({ status: "pending", data: null, error: null });
    try {
      const response = await asyncFunction();
      setState({ status: "success", data: response, error: null });
      return response;
    } catch (error) {
      setState({ status: "error", data: null, error });
      throw error;
    }
  }, [asyncFunction]);

  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate]);

  return { ...state, execute };
}

/**
 * Export configuration and analysis utilities
 */
const AnalysisConfig = {
  // Preset analysis configurations
  presets: {
    comprehensive: {
      includeStats: true,
      includeDistribution: true,
      includeTrends: true,
      includeComparison: true,
      includeClustering: true,
    },
    minimal: {
      includeStats: true,
      includeDistribution: false,
      includeTrends: false,
      includeComparison: false,
      includeClustering: false,
    },
    focused: {
      includeStats: true,
      includeDistribution: true,
      includeTrends: true,
      includeComparison: true,
      includeClustering: false,
    },
  },

  // Generate analysis report
  generateReport: (snapshots, config = {}) => {
    const finalConfig = { ...AnalysisConfig.presets.comprehensive, ...config };
    const report = {
      timestamp: new Date().toISOString(),
      snapshotCount: snapshots.length,
    };

    if (finalConfig.includeStats) {
      const lastSnapshot = snapshots[snapshots.length - 1];
      report.nodeStats = MetricsEngine.calculateNodeStats(lastSnapshot.nodes);
      report.aggregateMetrics = MetricsEngine.calculateAggregateMetrics(snapshots);
    }

    if (finalConfig.includeDistribution) {
      const beliefs = snapshots.flatMap(s => s.nodes.map(n => n.belief));
      report.distribution = {
        ...StatisticsEngine.quartiles(beliefs),
        skewness: StatisticsEngine.skewness(beliefs),
        kurtosis: StatisticsEngine.kurtosis(beliefs),
      };
    }

    if (finalConfig.includeTrends) {
      const infectedCurve = snapshots.map(
        s => s.nodes.filter(n => n.belief > 0.5 && !n.treated).length
      );
      report.trends = {
        slope: TimeSeriesAnalyzer.estimateTrend(infectedCurve).slope,
        peaks: TimeSeriesAnalyzer.detectPeaks(infectedCurve),
        valleys: TimeSeriesAnalyzer.detectValleys(infectedCurve),
        volatility: TimeSeriesAnalyzer.calculateVolatility(infectedCurve),
      };
    }

    return report;
  },
};

/**
 * Performance monitoring utilities
 */
const PerformanceMonitor = {
  metrics: {},

  startMeasure: (label) => {
    PerformanceMonitor.metrics[label] = performance.now();
  },

  endMeasure: (label) => {
    if (!PerformanceMonitor.metrics[label]) return null;
    const duration = performance.now() - PerformanceMonitor.metrics[label];
    delete PerformanceMonitor.metrics[label];
    return duration;
  },

  measure: async (label, fn) => {
    PerformanceMonitor.startMeasure(label);
    try {
      const result = await fn();
      const duration = PerformanceMonitor.endMeasure(label);
      ErrorHandler.debug(`${label} completed in ${duration.toFixed(2)}ms`);
      return result;
    } catch (error) {
      PerformanceMonitor.endMeasure(label);
      throw error;
    }
  },

  getMetrics: () => {
    return Object.entries(PerformanceMonitor.metrics).map(([label, startTime]) => ({
      label,
      duration: performance.now() - startTime,
    }));
  },
};

/**
 * Notification and alert system
 */
const NotificationSystem = {
  notifications: [],

  add: (message, type = "info", duration = 3000) => {
    const id = Date.now();
    const notification = { id, message, type, timestamp: Date.now() };
    NotificationSystem.notifications.push(notification);

    if (duration > 0) {
      setTimeout(() => NotificationSystem.remove(id), duration);
    }

    return id;
  },

  remove: (id) => {
    NotificationSystem.notifications = NotificationSystem.notifications.filter(n => n.id !== id);
  },

  success: (message, duration) => NotificationSystem.add(message, "success", duration),
  error: (message, duration) => NotificationSystem.add(message, "error", duration),
  warning: (message, duration) => NotificationSystem.add(message, "warning", duration),
  info: (message, duration) => NotificationSystem.add(message, "info", duration),
};

// ─────────────────────────────────────────────────────────────────────────────
// CHART COMPONENTS (Optimized with Recharts)
// ─────────────────────────────────────────────────────────────────────────────
function LineChart({ data, height = 240 }) {
  const chartData = useMemo(() => {
    if (!data || !data.length) return [];
    const timesteps = data[0].values.length;
    return Array.from({ length: timesteps }, (_, i) => {
      const entry = { timestep: i };
      data.forEach(s => {
        entry[s.id] = s.values[i];
        if (s.band) {
          entry[`${s.id}_lo`] = s.band.lo[i];
          entry[`${s.id}_hi`] = s.band.hi[i];
        }
      });
      return entry;
    });
  }, [data]);

  return (
    <div style={{ width: "100%", height, marginTop: "10px" }} className="chart-container">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="timestep" fontSize={10} tick={{ fill: "var(--color-text-tertiary)" }} />
          <YAxis fontSize={10} tick={{ fill: "var(--color-text-tertiary)" }} />
          <Tooltip />
          {data.map(s => (
            <Line
              key={s.id}
              type="monotone"
              dataKey={s.id}
              stroke={s.color}
              strokeWidth={s.id === "gnn_rl" ? 3 : 1.5}
              dot={false}
              strokeDasharray={s.dash ? s.dash.join(" ") : "0"}
              name={s.label}
              isAnimationActive={false}
            />
          ))}
          {data.map(s => s.band && (
            <Area
              key={s.id + "-band"}
              type="monotone"
              dataKey={`${s.id}_hi`}
              dataKey2={`${s.id}_lo`}
              stroke="none"
              fill={s.color}
              fillOpacity={0.1}
              isAnimationActive={false}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function BarChart({ height = 200 }) {
  const data = useMemo(() => {
    return Object.entries(GRAPH_META).map(([key, meta]) => {
      const g = SUMMARY[key];
      const entry = { name: meta.short };
      Object.keys(STRATEGIES).forEach(strat => {
        entry[strat] = strat === "gnn_rl" ? g.gnn_rl.median : g[strat]?.median ?? 0;
      });
      return entry;
    });
  }, []);

  return (
    <div style={{ width: "100%", height }} className="chart-container">
      <ResponsiveContainer width="100%" height="100%">
        <ReBarChart data={data} margin={{ top: 20, right: 30, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" fontSize={10} tick={{ fill: "var(--color-text-tertiary)" }} />
          <YAxis fontSize={10} tick={{ fill: "var(--color-text-tertiary)" }} />
          <Tooltip cursor={{ fill: "transparent" }} />
          {Object.entries(STRATEGIES).map(([id, strat]) => (
            <Bar key={id} dataKey={id} fill={strat.color} name={strat.label} radius={[2, 2, 0, 0]} />
          ))}
        </ReBarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CustomBarChart({ data, width = 560, height = 180 }) {
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
    const base = 2.5 + nd.degree * 0.4;
    return Math.min(base, 8);
  };

  // Improved coordinate mapping for "zoomed-out" effect with 15% safety padding
  const mapX = useCallback((val) => 15 + val * (width - 30), [width]);
  const mapY = useCallback((val) => 15 + val * (height - 30), [height]);

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
const MemoizedNetworkGraph = memo(NetworkGraph);

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

          <MemoizedNetworkGraph 
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
