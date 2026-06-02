export async function predictGNN(gname, checkpoint = "final", seed = 0) {
  const response = await fetch(`${import.meta.env.VITE_API_BASE}/gnn/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ graph: gname, checkpoint, seed })
  });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || 'Failed to fetch GNN predictions');
  }
  const data = await response.json();
  // Expected format: { nodes: [...], edges: [...], seedIdx }
  return data;
}

export async function predictAction(stateVector, gname = "p2p_gnutella", checkpoint = "final", seed = 0) {
  const response = await fetch(`${import.meta.env.VITE_API_BASE}/rl/action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state: stateVector, graph: gname, checkpoint, seed })
  });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || 'Failed to fetch RL action');
  }
  const data = await response.json();
  return data.action;
}

export async function simulateStrategyAPI(gname, strategy, nodes, edges, timesteps = 60, budget = 3.0, checkpoint = "final", seed = 0) {
  const response = await fetch(`${import.meta.env.VITE_API_BASE}/api/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      graph_name: gname,
      strategy: strategy,
      nodes: nodes,
      edges: edges,
      timesteps: timesteps,
      budget: budget,
      checkpoint: checkpoint,
      seed: seed
    })
  });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || 'Failed to simulate strategy on backend');
  }
  const data = await response.json();
  return data.snapshots;
}
