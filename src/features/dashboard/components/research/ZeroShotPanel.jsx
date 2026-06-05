import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";

// Shield Icon SVG
const ShieldIcon = ({ size = 24, color = "#2ecc71" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);

// ─────────────────────────────────────────────────────────────────────────────
// ZERO-SHOT EVALUATION ANALYTICS & UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Graph topology analysis engine
 */
const GraphAnalyzer = {
  // Calculate degree distribution
  calculateDegreeDistribution: (nodes, edges) => {
    const degrees = nodes.map(() => 0);
    edges.forEach(([a, b]) => {
      degrees[a]++;
      degrees[b]++;
    });
    return {
      degrees,
      mean: degrees.reduce((a, b) => a + b, 0) / nodes.length,
      max: Math.max(...degrees),
      min: Math.min(...degrees),
      median: degrees.sort((a, b) => a - b)[Math.floor(degrees.length / 2)],
    };
  },

  // Compute graph density
  graphDensity: (nodeCount, edgeCount) => {
    return nodeCount > 1 ? (2 * edgeCount) / (nodeCount * (nodeCount - 1)) : 0;
  },

  // Calculate clustering coefficient
  clusteringCoefficient: (nodes, edges) => {
    const adj = nodes.map(() => new Set());
    edges.forEach(([a, b]) => {
      adj[a].add(b);
      adj[b].add(a);
    });

    let totalTriangles = 0;
    let totalTriplets = 0;

    nodes.forEach((n, i) => {
      const neighbors = Array.from(adj[i]);
      const degree = neighbors.length;
      if (degree < 2) return;

      const possibleTriangles = (degree * (degree - 1)) / 2;
      let actualTriangles = 0;

      for (let j = 0; j < neighbors.length; j++) {
        for (let k = j + 1; k < neighbors.length; k++) {
          if (adj[neighbors[j]].has(neighbors[k])) {
            actualTriangles++;
          }
        }
      }

      totalTriangles += actualTriangles;
      totalTriplets += possibleTriangles;
    });

    return totalTriplets > 0 ? totalTriangles / totalTriplets : 0;
  },

  // Calculate assortativity coefficient
  assortativity: (nodes, edges) => {
    const degrees = nodes.map(() => 0);
    edges.forEach(([a, b]) => {
      degrees[a]++;
      degrees[b]++;
    });

    let sumProduct = 0, sumDeg1 = 0, sumDeg2 = 0, sumSq1 = 0, sumSq2 = 0;

    edges.forEach(([a, b]) => {
      const da = degrees[a];
      const db = degrees[b];
      sumProduct += da * db;
      sumDeg1 += da;
      sumDeg2 += db;
      sumSq1 += da * da;
      sumSq2 += db * db;
    });

    const m = edges.length;
    const num = sumProduct / m - (sumDeg1 / m) * (sumDeg2 / m);
    const den = Math.sqrt((sumSq1 / m - Math.pow(sumDeg1 / m, 2)) * (sumSq2 / m - Math.pow(sumDeg2 / m, 2)));

    return den !== 0 ? num / den : 0;
  },

  // Identify graph diameter (approximate via BFS from highest degree node)
  approximateDiameter: (nodes, edges) => {
    const adj = nodes.map(() => []);
    edges.forEach(([a, b]) => {
      adj[a].push(b);
      adj[b].push(a);
    });

    let maxDiameter = 0;
    const startNodes = [
      nodes.reduce((max, n, i) => (adj[i].length > adj[max].length ? i : max), 0),
      nodes.reduce((min, n, i) => (adj[i].length < adj[min].length ? i : min), 0),
    ];

    startNodes.forEach(start => {
      const dist = Array(nodes.length).fill(-1);
      const queue = [start];
      dist[start] = 0;
      let head = 0;

      while (head < queue.length) {
        const u = queue[head++];
        adj[u].forEach(v => {
          if (dist[v] === -1) {
            dist[v] = dist[u] + 1;
            queue.push(v);
          }
        });
      }

      const diameter = Math.max(...dist);
      maxDiameter = Math.max(maxDiameter, diameter);
    });

    return maxDiameter === -1 ? 0 : maxDiameter;
  },

  // Compute spectral radius (largest eigenvalue approximation)
  spectralRadius: (nodes, edges) => {
    const adj = nodes.map(() => Array(nodes.length).fill(0));
    edges.forEach(([a, b]) => {
      adj[a][b]++;
      adj[b][a]++;
    });

    let v = nodes.map(() => 1);
    for (let iter = 0; iter < 20; iter++) {
      let newV = adj.map((row, i) => row.reduce((sum, val, j) => sum + val * v[j], 0));
      const norm = Math.sqrt(newV.reduce((sum, x) => sum + x * x, 0));
      v = newV.map(x => x / norm);
    }

    return Math.sqrt(adj.reduce((sum, row, i) => sum + row.reduce((s, val, j) => s + val * v[i] * v[j], 0), 0));
  },

  // Network efficiency metrics
  networkMetrics: (nodes, edges) => {
    return {
      nodeCount: nodes.length,
      edgeCount: edges.length,
      density: GraphAnalyzer.graphDensity(nodes.length, edges.length),
      degreeDistribution: GraphAnalyzer.calculateDegreeDistribution(nodes, edges),
      clusteringCoeff: GraphAnalyzer.clusteringCoefficient(nodes, edges),
      assortativity: GraphAnalyzer.assortativity(nodes, edges),
      diameter: GraphAnalyzer.approximateDiameter(nodes, edges),
      spectralRadius: GraphAnalyzer.spectralRadius(nodes, edges),
    };
  },
};

/**
 * Zero-shot simulation evaluation metrics
 */
const EvaluationMetrics = {
  // Calculate suppression efficiency
  suppressionEfficiency: (baseline, strategic) => {
    return baseline > 0 ? ((baseline - strategic) / baseline) * 100 : 0;
  },

  // AUC-based suppression metric
  aucSuppression: (baselineCurve, strategicCurve) => {
    const baselineAuc = baselineCurve.reduce((a, b) => a + b, 0);
    const strategicAuc = strategicCurve.reduce((a, b) => a + b, 0);
    return EvaluationMetrics.suppressionEfficiency(baselineAuc, strategicAuc);
  },

  // Time-to-peak reduction
  timeToReduction: (baselineCurve, strategicCurve, threshold = 0.5) => {
    const baselineTime = baselineCurve.findIndex(v => v >= baselineCurve[0] * threshold);
    const strategicTime = strategicCurve.findIndex(v => v >= strategicCurve[0] * threshold);

    if (baselineTime === -1 || strategicTime === -1) return 0;
    return baselineTime > 0 ? ((baselineTime - strategicTime) / baselineTime) * 100 : 0;
  },

  // Robustness score (inverse of final infected count)
  robustnessScore: (finalInfected, totalNodes) => {
    return totalNodes > 0 ? (1 - finalInfected / totalNodes) * 100 : 0;
  },

  // Confidence interval for suppression metric
  confidenceInterval: (suppression, sampleSize, confidence = 0.95) => {
    const zScore = confidence === 0.95 ? 1.96 : 2.576;
    const stdError = Math.sqrt((suppression * (100 - suppression)) / sampleSize);
    return {
      lower: suppression - zScore * stdError,
      upper: suppression + zScore * stdError,
      margin: zScore * stdError,
    };
  },

  // Compute relative improvement vs baseline
  relativeImprovement: (baseline, strategic) => {
    if (baseline === 0) return 0;
    return ((baseline - strategic) / Math.abs(baseline)) * 100;
  },

  // Variance in suppression across timesteps
  suppressionVariance: (baselineCurve, strategicCurve) => {
    const suppressions = baselineCurve.map((b, i) =>
      EvaluationMetrics.suppressionEfficiency(b, strategicCurve[i])
    );
    const mean = suppressions.reduce((a, b) => a + b, 0) / suppressions.length;
    const variance = suppressions.reduce((acc, v) => acc + (v - mean) ** 2, 0) / suppressions.length;
    return Math.sqrt(variance);
  },
};

/**
 * Simulation result analysis
 */
const SimulationAnalyzer = {
  // Extract timeline statistics
  extractTimestepStats: (snapshots) => {
    return snapshots.map((snap, idx) => {
      const infected = snap.filter(n => n.belief > 0.5).length;
      const treated = snap.filter(n => n.treated).length;
      const avgBelief = snap.reduce((sum, n) => sum + n.belief, 0) / snap.length;
      const maxBelief = Math.max(...snap.map(n => n.belief));

      return { timestep: idx, infected, treated, avgBelief, maxBelief };
    });
  },

  // Detect inflection points in curve
  findInflectionPoints: (curve) => {
    const inflections = [];
    for (let i = 1; i < curve.length - 1; i++) {
      const d1 = curve[i] - curve[i - 1];
      const d2 = curve[i + 1] - curve[i];
      if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) {
        inflections.push({ index: i, value: curve[i] });
      }
    }
    return inflections;
  },

  // Calculate suppression onset (timestep when intervention becomes visible)
  suppressionOnset: (baselineCurve, strategicCurve, threshold = 5) => {
    for (let t = 1; t < Math.min(baselineCurve.length, strategicCurve.length); t++) {
      const diff = baselineCurve[t] - strategicCurve[t];
      if (diff >= threshold) return t;
    }
    return -1;
  },

  // Analyze suppression plateau (steady state suppression)
  suppressionPlateau: (baselineCurve, strategicCurve, windowSize = 5) => {
    const suppressions = baselineCurve.map((b, i) =>
      EvaluationMetrics.suppressionEfficiency(b, strategicCurve[i])
    );

    const plateaus = [];
    for (let i = windowSize; i < suppressions.length; i++) {
      const window = suppressions.slice(i - windowSize, i);
      const variance = window.reduce((acc, v, idx) => {
        const mean = window.reduce((a, b) => a + b, 0) / window.length;
        return acc + (v - mean) ** 2;
      }, 0) / windowSize;

      if (variance < 1) {
        const mean = window.reduce((a, b) => a + b, 0) / windowSize;
        plateaus.push({ start: i - windowSize, end: i, value: mean, variance });
      }
    }

    return plateaus.length > 0 ? plateaus[plateaus.length - 1] : null;
  },

  // Compute curve similarity (Euclidean distance)
  curveSimilarity: (curve1, curve2) => {
    const minLen = Math.min(curve1.length, curve2.length);
    let sumSq = 0;
    for (let i = 0; i < minLen; i++) {
      sumSq += (curve1[i] - curve2[i]) ** 2;
    }
    const distance = Math.sqrt(sumSq / minLen);
    const maxVal = Math.max(...curve1, ...curve2);
    return maxVal > 0 ? (1 - distance / maxVal) * 100 : 100;
  },
};

/**
 * Performance benchmarking for zero-shot evaluation
 */
const BenchmarkEngine = {
  // Compare suppression across different graph types
  compareTopologies: (results) => {
    return Object.fromEntries(
      Object.entries(results).map(([topologyType, metrics]) => [
        topologyType,
        {
          avgSuppression: metrics.reduce((sum, m) => sum + m.suppression, 0) / metrics.length,
          maxSuppression: Math.max(...metrics.map(m => m.suppression)),
          minSuppression: Math.min(...metrics.map(m => m.suppression)),
          variance: metrics.reduce((sum, m, idx, arr) => {
            const mean = arr.reduce((a, b) => a + b.suppression, 0) / arr.length;
            return sum + (m.suppression - mean) ** 2;
          }, 0) / metrics.length,
        },
      ])
    );
  },

  // Rank topologies by generalization potential
  rankByGeneralization: (topologyMetrics) => {
    return Object.entries(topologyMetrics)
      .map(([name, metrics]) => ({
        name,
        score: metrics.avgSuppression - metrics.variance * 0.5,
        metrics,
      }))
      .sort((a, b) => b.score - a.score);
  },

  // Test robustness across parameter ranges
  robustnessTest: (baselineMetric, perturbedMetrics, tolerance = 10) => {
    const passes = perturbedMetrics.filter(m => Math.abs(m - baselineMetric) <= tolerance).length;
    return (passes / perturbedMetrics.length) * 100;
  },
};

/**
 * Data export and serialization utilities
 */
const ExportUtils = {
  // Create comprehensive evaluation report
  generateEvaluationReport: (evalResult, graphMetrics, timestamp = new Date()) => {
    return {
      timestamp: timestamp.toISOString(),
      evaluation: {
        nonePeak: evalResult.nonePeak,
        gnnPeak: evalResult.gnnPeak,
        suppressionRating: evalResult.rating,
        aucSuppression: EvaluationMetrics.aucSuppression(evalResult.noneCurve, evalResult.gnnCurve),
      },
      graph: graphMetrics,
      inflectionPoints: {
        baseline: SimulationAnalyzer.findInflectionPoints(evalResult.noneCurve),
        strategic: SimulationAnalyzer.findInflectionPoints(evalResult.gnnCurve),
      },
      suppressionOnset: SimulationAnalyzer.suppressionOnset(evalResult.noneCurve, evalResult.gnnCurve),
      plateau: SimulationAnalyzer.suppressionPlateau(evalResult.noneCurve, evalResult.gnnCurve),
    };
  },

  // Export as CSV
  toCsv: (data, filename) => {
    const headers = Object.keys(data[0] || {});
    const rows = [headers.join(",")];
    data.forEach(row => {
      rows.push(headers.map(h => row[h] ?? "").join(","));
    });
    return rows.join("\n");
  },

  // Create downloadable file
  downloadFile: (filename, content, mimeType = "application/json") => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  },
};

/**
 * Caching system for zero-shot evaluations
 */
class EvaluationCache {
  constructor(maxSize = 50) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  generateKey(type, numNodes, param) {
    return `${type}_${numNodes}_${param}`;
  }

  set(type, numNodes, param, result) {
    const key = this.generateKey(type, numNodes, param);
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, { result, timestamp: Date.now() });
  }

  get(type, numNodes, param) {
    const key = this.generateKey(type, numNodes, param);
    return this.cache.get(key)?.result || null;
  }

  has(type, numNodes, param) {
    const key = this.generateKey(type, numNodes, param);
    return this.cache.has(key);
  }

  clear() {
    this.cache.clear();
  }

  size() {
    return this.cache.size;
  }
}

/**
 * Error handling and validation for zero-shot evaluation
 */
const ValidationUtils = {
  // Validate graph topology
  isValidTopology: (nodes, edges) => {
    return (
      Array.isArray(nodes) &&
      Array.isArray(edges) &&
      nodes.length > 0 &&
      nodes.every(n => typeof n.belief === "number" && n.belief >= 0 && n.belief <= 1) &&
      edges.every(e => Array.isArray(e) && e.length === 2)
    );
  },

  // Validate simulation results
  isValidSimulation: (noneCurve, gnnCurve) => {
    return (
      Array.isArray(noneCurve) &&
      Array.isArray(gnnCurve) &&
      noneCurve.length === gnnCurve.length &&
      noneCurve.every(v => typeof v === "number" && v >= 0) &&
      gnnCurve.every(v => typeof v === "number" && v >= 0)
    );
  },

  // Sanitize parameters
  sanitizeParams: (type, nodes, param) => {
    return {
      type: ["barabasi_albert", "erdos_renyi", "watts_strogatz"].includes(type) ? type : "barabasi_albert",
      nodes: Math.max(20, Math.min(200, Math.floor(nodes))),
      param: Math.max(1, Math.min(10, Math.floor(param))),
    };
  },
};

/**
 * Custom React hooks for zero-shot panel
 */
function useEvaluationHistory() {
  const [history, setHistory] = useState([]);

  const addToHistory = useCallback((entry) => {
    setHistory(prev => [entry, ...prev].slice(0, 20));
  }, []);

  return { history, addToHistory };
}

function useGraphMetrics(nodes, edges) {
  return useMemo(() => {
    if (!nodes || !edges) return null;
    return GraphAnalyzer.networkMetrics(nodes, edges);
  }, [nodes, edges]);
}

// Seeded random helper for reproducible unseen graphs
function seededRand(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 4294967296; };
}

// ─────────────────────────────────────────────────────────────────────────────
// VISUALIZATION & RENDERING UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Advanced visualization helpers for network rendering
 */
const VisualizationUtils = {
  // Calculate node colors based on belief state
  getNodeColor: (belief, treated, isSeed) => {
    if (isSeed) return "#D4537E";
    if (treated) return "#1D9E75";
    if (belief > 0.7) return "#E24B4A";
    if (belief > 0.5) return "#EF9F27";
    if (belief > 0.2) return "#97C459";
    return "#B4B2A9";
  },

  // Calculate node size based on degree and centrality
  calculateNodeRadius: (node, degree, centrality = 1) => {
    const baseRadius = 2.5 + degree * 0.3;
    const scaledRadius = baseRadius * (0.8 + centrality * 0.4);
    return Math.min(scaledRadius, 8);
  },

  // Generate opacity for edges based on connection strength
  calculateEdgeOpacity: (source, target, nodes) => {
    const avgBelief = (nodes[source].belief + nodes[target].belief) / 2;
    return Math.max(0.2, Math.min(0.8, avgBelief));
  },

  // Calculate label position to avoid overlap
  calculateLabelPosition: (nodeX, nodeY, canvasWidth, canvasHeight) => {
    const tx = nodeX > canvasWidth * 0.7 ? nodeX - 120 : nodeX + 12;
    const ty = nodeY > canvasHeight * 0.8 ? nodeY - 60 : nodeY + 8;
    return { x: tx, y: ty };
  },

  // Apply force-directed layout adjustment
  adjustNodePosition: (x, y, repulsion = 0.1, attraction = 0.05) => {
    return {
      x: x + (Math.random() - 0.5) * repulsion,
      y: y + (Math.random() - 0.5) * repulsion,
    };
  },

  // Generate SVG path for smooth curve between nodes
  generateCurvePath: (x1, y1, x2, y2) => {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const controlX = x1 + dx * 0.3;
    const controlY = y1 + dy * 0.5;
    return `Q ${controlX} ${controlY} ${x2} ${y2}`;
  },
};

/**
 * Animation and transition utilities
 */
const AnimationUtils = {
  // Easing functions
  easeInOutQuad: (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
  easeInQuad: (t) => t * t,
  easeOutQuad: (t) => 1 - (1 - t) * (1 - t),
  easeInCubic: (t) => t * t * t,
  easeOutCubic: (t) => 1 - Math.pow(1 - t, 3),

  // Interpolate between two values
  interpolate: (start, end, t, easingFn = AnimationUtils.easeInOutQuad) => {
    return start + (end - start) * easingFn(t);
  },

  // Generate animation frame sequence
  generateFrames: (startValue, endValue, frameCount, easingFn) => {
    const frames = [];
    for (let i = 0; i <= frameCount; i++) {
      const t = i / frameCount;
      frames.push(AnimationUtils.interpolate(startValue, endValue, t, easingFn));
    }
    return frames;
  },
};

/**
 * Statistical utilities for evaluation results
 */
const StatisticsUtils = {
  // Calculate standard deviation
  standardDeviation: (values) => {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  },

  // Calculate percentile
  percentile: (values, p) => {
    const sorted = [...values].sort((a, b) => a - b);
    const idx = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    const weight = idx % 1;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  },

  // Compute range
  range: (values) => ({
    min: Math.min(...values),
    max: Math.max(...values),
    span: Math.max(...values) - Math.min(...values),
  }),

  // Calculate z-score normalization
  normalize: (value, mean, stdDev) => {
    return stdDev !== 0 ? (value - mean) / stdDev : 0;
  },

  // Detect outliers using IQR method
  detectOutliers: (values, multiplier = 1.5) => {
    const q1 = StatisticsUtils.percentile(values, 25);
    const q3 = StatisticsUtils.percentile(values, 75);
    const iqr = q3 - q1;
    return values.filter(v => v < q1 - multiplier * iqr || v > q3 + multiplier * iqr);
  },
};

/**
 * Performance profiling utilities
 */
const ProfilingUtils = {
  // Measure function execution time
  measureTime: async (fn, label = "operation") => {
    const start = performance.now();
    try {
      const result = await fn();
      const duration = performance.now() - start;
      return { result, duration, success: true, label };
    } catch (error) {
      const duration = performance.now() - start;
      return { error, duration, success: false, label };
    }
  },

  // Profile memory usage (estimate)
  estimateMemory: (obj) => {
    const objectList = [];
    const stack = [obj];
    let bytes = 0;

    while (stack.length) {
      const value = stack.pop();
      if (typeof value === "boolean") bytes += 4;
      else if (typeof value === "string") bytes += value.length * 2;
      else if (typeof value === "number") bytes += 8;
      else if (typeof value === "object" && value !== null) {
        if (objectList.indexOf(value) === -1) {
          objectList.push(value);
          Object.keys(value).forEach(prop => stack.push(value[prop]));
        }
      }
    }

    return bytes;
  },

  // Track function call count
  createCallCounter: (fn) => {
    let count = 0;
    return {
      fn: (...args) => {
        count++;
        return fn(...args);
      },
      count: () => count,
      reset: () => { count = 0; },
    };
  },
};

/**
 * Utility for managing evaluation state and results
 */
class EvaluationStateManager {
  constructor() {
    this.state = {
      loading: false,
      error: null,
      results: null,
      timestamp: null,
      metadata: {},
    };
  }

  setState(updates) {
    this.state = { ...this.state, ...updates };
  }

  setLoading(isLoading) {
    this.setState({ loading: isLoading });
  }

  setError(error) {
    this.setState({ error });
  }

  setResults(results, metadata = {}) {
    this.setState({
      results,
      timestamp: new Date(),
      metadata,
      error: null,
    });
  }

  getState() {
    return { ...this.state };
  }

  reset() {
    this.state = {
      loading: false,
      error: null,
      results: null,
      timestamp: null,
      metadata: {},
    };
  }
}

/**
 * Advanced comparison utilities for evaluation results
 */
const ComparisonUtils = {
  // Compare two evaluation results
  compareResults: (result1, result2) => {
    return {
      peakDifference: Math.abs(result1.gnnPeak - result2.gnnPeak),
      ratingDifference: Math.abs(result1.rating - result2.rating),
      aucDifference: Math.abs(
        EvaluationMetrics.aucSuppression(result1.noneCurve, result1.gnnCurve) -
        EvaluationMetrics.aucSuppression(result2.noneCurve, result2.gnnCurve)
      ),
      similarity: SimulationAnalyzer.curveSimilarity(result1.gnnCurve, result2.gnnCurve),
    };
  },

  // Aggregate multiple evaluation results
  aggregateResults: (results) => {
    if (!results.length) return null;

    const suppressions = results.map(r => r.rating);
    const peaks = results.map(r => r.gnnPeak);
    const aucSuppressions = results.map(r =>
      EvaluationMetrics.aucSuppression(r.noneCurve, r.gnnCurve)
    );

    return {
      avgSuppression: suppressions.reduce((a, b) => a + b, 0) / suppressions.length,
      stdSuppression: StatisticsUtils.standardDeviation(suppressions),
      avgPeak: peaks.reduce((a, b) => a + b, 0) / peaks.length,
      stdPeak: StatisticsUtils.standardDeviation(peaks),
      avgAucSuppression: aucSuppressions.reduce((a, b) => a + b, 0) / aucSuppressions.length,
      minSuppression: Math.min(...suppressions),
      maxSuppression: Math.max(...suppressions),
      resultCount: results.length,
    };
  },

  // Find best and worst performing results
  rankResults: (results) => {
    return results
      .map((r, idx) => ({ ...r, index: idx }))
      .sort((a, b) => b.rating - a.rating);
  },
};

/**
 * Logging and debugging utilities
 */
const DebugUtils = {
  // Console logging with context
  log: (message, context = {}, level = "info") => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${level.toUpperCase()}] ${message}`, context);
  },

  // Create evaluation summary for logging
  summarizeEvaluation: (evalResult, graphMetrics) => {
    return {
      suppression: evalResult.rating.toFixed(2),
      peakReduction: (evalResult.nonePeak - evalResult.gnnPeak).toFixed(1),
      graphSize: graphMetrics.nodeCount,
      graphDensity: graphMetrics.density.toFixed(3),
      timestamp: new Date().toLocaleTimeString(),
    };
  },
};

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

  // Initialize evaluation utilities
  const evaluationCacheRef = useRef(new EvaluationCache(50));
  const stateManagerRef = useRef(new EvaluationStateManager());
  const { history, addToHistory } = useEvaluationHistory();

  // Compute graph metrics for current topology
  const graphMetricsRef = useRef(null);

  // Auto-step effect for unseen graph snapshots
  React.useEffect(() => {
    if (!isPlaying || !evalResult) return;
    const id = setInterval(() => {
      setActiveStep(s => (s + 1) % 50);
    }, 400);
    return () => clearInterval(id);
  }, [isPlaying, evalResult]);

  // Handle zero-shot evaluation
  const handleEvaluateUnseen = useCallback(async () => {
    // Validate and sanitize parameters
    const params = ValidationUtils.sanitizeParams(unseenType, unseenNodes, unseenParam);
    
    // Check cache
    if (evaluationCacheRef.current.has(params.type, params.nodes, params.param)) {
      DebugUtils.log("Cache hit for zero-shot evaluation", { params });
      const cached = evaluationCacheRef.current.get(params.type, params.nodes, params.param);
      setEvalResult(cached);
      const summary = DebugUtils.summarizeEvaluation(cached, graphMetricsRef.current);
      addToHistory({ ...summary, source: "cache" });
      return;
    }

    setEvaluating(true);
    stateManagerRef.current.setLoading(true);

    try {
      // Generate network graph
      const base = generateUnseenGraph(params.type, params.nodes, params.param);

      // Validate topology
      if (!ValidationUtils.isValidTopology(base.nodes, base.edges)) {
        throw new Error("Invalid graph topology generated");
      }

      // Calculate graph metrics for analysis
      graphMetricsRef.current = GraphAnalyzer.networkMetrics(base.nodes, base.edges);
      DebugUtils.log("Graph metrics calculated", graphMetricsRef.current);

      // Run simulations
      const noneSnaps = await simulateStrategyAPI("unseen_custom", "none", base.nodes, base.edges, 50, 20.0);
      const gnnSnaps = await simulateStrategyAPI("unseen_custom", "gnn_rl", base.nodes, base.edges, 50, 20.0);

      if (noneSnaps && gnnSnaps) {
        // Validate simulation results
        const noneCurve = noneSnaps.map(snap => snap.filter(n => n.belief > 0.5).length);
        const gnnCurve = gnnSnaps.map(snap => snap.filter(n => n.belief > 0.5).length);

        if (!ValidationUtils.isValidSimulation(noneCurve, gnnCurve)) {
          throw new Error("Invalid simulation results");
        }

        // Calculate evaluation metrics
        const nonePeak = Math.max(...noneCurve.slice(1));
        const gnnPeak = Math.max(...gnnCurve.slice(1));
        const rating = nonePeak > 0 ? ((nonePeak - gnnPeak) / nonePeak * 100) : 0.0;

        // Generate comprehensive evaluation report
        const evalReport = ExportUtils.generateEvaluationReport(
          { nonePeak, gnnPeak, rating, noneCurve, gnnCurve },
          graphMetricsRef.current
        );

        const positionedSnaps = gnnSnaps.map(snap => ({
          nodes: snap.map((n, idx) => ({
            ...n,
            x: base.nodes[idx].x,
            y: base.nodes[idx].y
          })),
          edges: base.edges
        }));

        const result = {
          noneCurve,
          gnnCurve,
          nonePeak,
          gnnPeak,
          rating,
          snapshots: positionedSnaps,
          report: evalReport,
        };

        // Cache the result
        evaluationCacheRef.current.set(params.type, params.nodes, params.param, result);

        // Update state
        setEvalResult(result);
        stateManagerRef.current.setResults(result, { topology: params, graphMetrics: graphMetricsRef.current });

        // Add to history with metrics
        const summary = DebugUtils.summarizeEvaluation(result, graphMetricsRef.current);
        addToHistory({ ...summary, params, graphMetrics: graphMetricsRef.current });

        DebugUtils.log("Evaluation completed successfully", summary);
        setActiveStep(49);
      }
    } catch (err) {
      DebugUtils.log("Evaluation failed", { error: err.message }, "error");
      stateManagerRef.current.setError(err);
      alert("Evaluation failed. Make sure the FastAPI backend is running on port 8000!");
      console.error(err);
    } finally {
      setEvaluating(false);
      stateManagerRef.current.setLoading(false);
    }
  }, [unseenType, unseenNodes, unseenParam, simulateStrategyAPI, addToHistory]);

  // Memoize graph metrics computation
  const currentGraphMetrics = useMemo(() => {
    if (!evalResult) return null;
    return graphMetricsRef.current;
  }, [evalResult]);

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

        {/* Evaluation statistics display */}
        {currentGraphMetrics && (
          <div style={{
            marginBottom: "12px",
            padding: "8px 12px",
            background: "var(--color-background-primary)",
            borderRadius: "6px",
            fontSize: "11px",
            color: "var(--color-text-secondary)",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
            gap: "8px",
          }}>
            <div><strong>Nodes:</strong> {currentGraphMetrics.nodeCount}</div>
            <div><strong>Edges:</strong> {currentGraphMetrics.edgeCount}</div>
            <div><strong>Density:</strong> {currentGraphMetrics.density.toFixed(3)}</div>
            <div><strong>Clustering:</strong> {currentGraphMetrics.clusteringCoeff.toFixed(3)}</div>
            <div><strong>Diameter:</strong> {currentGraphMetrics.diameter}</div>
            <div><strong>Avg Degree:</strong> {(currentGraphMetrics.degreeDistribution.mean).toFixed(1)}</div>
          </div>
        )}

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
            width: "100%",
          }}
        >
          {evaluating ? "Evaluating..." : "Generate & Run Real-Time evaluation"}
        </button>
      </div>

      {/* Evaluation history display */}
      {history.length > 0 && (
        <div style={{
          marginTop: "16px",
          background: "var(--color-background-secondary)",
          borderRadius: "var(--border-radius-md)",
          padding: "12px",
          border: "0.5px solid var(--color-border-tertiary)",
        }}>
          <div style={{ fontSize: "11px", fontWeight: "600", color: "var(--color-text-secondary)", marginBottom: "8px" }}>
            Evaluation History ({history.length})
          </div>
          <div style={{ maxHeight: "120px", overflowY: "auto", fontSize: "10px", color: "var(--color-text-tertiary)" }}>
            {history.slice(0, 5).map((h, i) => (
              <div key={i} style={{ padding: "3px 0", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                {h.timestamp} · Suppression: {h.suppression}% · Nodes: {h.graphSize}
              </div>
            ))}
          </div>
        </div>
      )}

      {evalResult && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1.2fr",
          gap: "20px",
          background: "var(--color-background-primary)",
          borderRadius: "var(--border-radius-lg)",
          padding: "16px",
          border: "0.5px solid var(--color-border-tertiary)",
          marginTop: "16px",
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
                    const report = ExportUtils.generateEvaluationReport(evalResult, currentGraphMetrics);
                    const content = JSON.stringify(report, null, 2);
                    ExportUtils.downloadFile(`zero_shot_evaluation_${unseenType}_${Date.now()}.json`, content);
                  }}
                  style={{ padding: "4px 8px", borderRadius: "4px", fontSize: "10px", cursor: "pointer", border: "1px solid var(--color-accent)", color: "var(--color-text-primary)" }}
                >
                  Download Report
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
                  const mapX = (val) => 15 + val * 370;
                  const mapY = (val) => 15 + val * 270;
                  return (
                    <line
                      key={i}
                      x1={mapX(nu.x)} y1={mapY(nu.y)}
                      x2={mapX(nv.x)} y2={mapY(nv.y)}
                      stroke="var(--color-border-tertiary)"
                      strokeWidth="0.4"
                      opacity={VisualizationUtils.calculateEdgeOpacity(u, v, evalResult.snapshots[activeStep].nodes)}
                    />
                  );
                })}
                {/* Nodes */}
                {evalResult.snapshots[activeStep].nodes.map((n, i) => {
                  const mapX = (val) => 15 + val * 370;
                  const mapY = (val) => 15 + val * 270;
                  const color = VisualizationUtils.getNodeColor(n.belief, n.treated, n.isSeed);
                  return (
                    <circle
                      key={i}
                      cx={mapX(n.x)} cy={mapY(n.y)}
                      r={n.isSeed ? 4.5 : 2.5}
                      fill={color}
                      opacity={0.85}
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
