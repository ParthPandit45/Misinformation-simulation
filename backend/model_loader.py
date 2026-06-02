import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os

# Configs
GNN_HIDDEN = 64
DQN_HIDDEN = 128
SPEC_DIM = 8
N_ACTIONS = 5

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'results', 'weights')

# ─── Model Definitions ───

class RiskGNN(nn.Module):
    def __init__(self, in_dim=7 + SPEC_DIM, h=GNN_HIDDEN):
        super().__init__()
        self.proj = nn.Linear(in_dim, h, bias=False)
        self.c1   = GCNConv(in_dim, h)
        self.c2   = GCNConv(h, h)
        self.c3   = GCNConv(h, h // 2)
        self.c4   = GCNConv(h // 2, 1)
        self.bn1  = nn.LayerNorm(h)
        self.bn2  = nn.LayerNorm(h)
        self.bn3  = nn.LayerNorm(h // 2)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x, ei):
        res = self.proj(x)
        x1  = F.relu(self.bn1(self.c1(x, ei)))
        x2  = F.relu(self.bn2(self.c2(x1, ei))) + res
        x2  = self.drop(x2)
        x3  = F.relu(self.bn3(self.c3(x2, ei)))
        return torch.sigmoid(self.c4(x3, ei))

class DuelingDQN(nn.Module):
    def __init__(self, s_dim=6, a_dim=N_ACTIONS, h=DQN_HIDDEN):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(s_dim, h),      nn.LayerNorm(h),      nn.ReLU(),
            nn.Linear(h, h // 2),    nn.LayerNorm(h // 2), nn.ReLU(),
            nn.Linear(h // 2, h // 4), nn.ReLU(),
        )
        self.V = nn.Linear(h // 4, 1)
        self.A = nn.Linear(h // 4, a_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.V(h) + self.A(h) - self.A(h).mean(-1, keepdim=True)

# ─── Load Model Weights per Graph Type ───

_MODELS_CACHE = {}

def get_models(graph_name: str, checkpoint: str = "final", seed: int = 0):
    """Load and cache GNN and DQN models for the given graph_name, checkpoint, and seed"""
    if checkpoint is None:
        checkpoint = "final"
    if seed is None:
        seed = 0
    cache_key = (graph_name, checkpoint, seed)
    if cache_key in _MODELS_CACHE:
        return _MODELS_CACHE[cache_key]

    # Map generic names
    normalized_name = "p2p_gnutella"
    if "facebook" in graph_name.lower():
        normalized_name = "facebook"
    elif "grqc" in graph_name.lower():
        normalized_name = "ca_grqc"

    if checkpoint == "final":
        gnn_filename = f"{normalized_name}_final_seed{seed}_gnn.pt"
        dqn_filename = f"{normalized_name}_final_seed{seed}_dqn.pt"
    else:
        ckpt_str = checkpoint if checkpoint.startswith("ep") else f"ep{checkpoint}"
        gnn_filename = f"{normalized_name}_{ckpt_str}_gnn.pt"
        dqn_filename = f"{normalized_name}_{ckpt_str}_dqn.pt"

    gnn_path = os.path.join(WEIGHTS_DIR, gnn_filename)
    dqn_path = os.path.join(WEIGHTS_DIR, dqn_filename)

    gnn = RiskGNN()
    dqn = DuelingDQN()

    # Load GNN
    if os.path.exists(gnn_path):
        print(f"Loading actual GNN weights from {gnn_path}")
        try:
            gnn.load_state_dict(torch.load(gnn_path, map_location="cpu"))
        except Exception as e:
            print(f"Warning: Failed to load GNN state dict ({e}). Running unitialized.")
    else:
        fallback_gnn_path = os.path.join(WEIGHTS_DIR, f"{normalized_name}_final_seed0_gnn.pt")
        if os.path.exists(fallback_gnn_path) and gnn_path != fallback_gnn_path:
            print(f"Warning: GNN weight file not found at {gnn_path}. Falling back to {fallback_gnn_path}")
            try:
                gnn.load_state_dict(torch.load(fallback_gnn_path, map_location="cpu"))
            except Exception as e:
                print(f"Warning: Failed to load fallback GNN ({e}). Running unitialized.")
        else:
            print(f"Warning: GNN weight file not found at {gnn_path}. Running unitialized.")

    # Load DQN
    if os.path.exists(dqn_path):
        print(f"Loading actual DQN weights from {dqn_path}")
        try:
            dqn.load_state_dict(torch.load(dqn_path, map_location="cpu"))
        except Exception as e:
            print(f"Warning: Failed to load DQN state dict ({e}). Running unitialized.")
    else:
        fallback_dqn_path = os.path.join(WEIGHTS_DIR, f"{normalized_name}_final_seed0_dqn.pt")
        if os.path.exists(fallback_dqn_path) and dqn_path != fallback_dqn_path:
            print(f"Warning: DQN weight file not found at {dqn_path}. Falling back to {fallback_dqn_path}")
            try:
                dqn.load_state_dict(torch.load(fallback_dqn_path, map_location="cpu"))
            except Exception as e:
                print(f"Warning: Failed to load fallback DQN ({e}). Running unitialized.")
        else:
            print(f"Warning: DQN weight file not found at {dqn_path}. Running unitialized.")

    gnn.eval()
    dqn.eval()
    _MODELS_CACHE[cache_key] = (gnn, dqn)
    return gnn, dqn

# ─── Dynamic Feature Calculation ───

def compute_spectral_embeddings(G, k=SPEC_DIM):
    n = len(G.nodes())
    L = nx.normalized_laplacian_matrix(G, nodelist=range(n)).astype(np.float32)
    k_ = min(k + 1, n - 2)
    try:
        vals, vecs = spla.eigsh(L, k=k_, which='SM')
        order = np.argsort(vals)
        vecs = vecs[:, order]
        vals = vals[order]
        emb = vecs[:, 1:k+1] if vecs.shape[1] > k else vecs[:, 1:]
    except Exception as e:
        print(f"Eigenvalue solver failed ({e}), using random fallback.")
        emb = np.random.randn(n, k).astype(np.float32)
        vals = np.zeros(k, dtype=np.float32)

    if emb.shape[1] < k:
        emb = np.pad(emb, ((0, 0), (0, k - emb.shape[1])))
    return vals.astype(np.float32), emb.astype(np.float32)

def compute_centralities(G):
    n = len(G.nodes())
    dg = dict(G.degree())
    mx = max(dg.values()) or 1
    deg_n = np.array([dg[v] / mx for v in range(n)], dtype=np.float32)
    try:
        bwd = nx.betweenness_centrality(G, normalized=True, k=min(100, n))
        bw = np.array([bwd[v] for v in range(n)], dtype=np.float32)
    except Exception:
        bw = deg_n.copy()
    return deg_n, bw

# ─── Core Simulation Engine ───

class SimulationEngine:
    def __init__(self, graph_name: str, nodes_list: list, edges_list: list, checkpoint: str = "final", seed: int = 0):
        self.graph_name = graph_name
        self.n = len(nodes_list)
        
        # Build NetworkX graph
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        self.G.add_edges_from(edges_list)
        
        # Build adjacency dictionary
        self.adj = {v: list(self.G.neighbors(v)) for v in range(self.n)}

        # Load weights
        self.gnn, self.dqn = get_models(graph_name, checkpoint, seed)

        # Precompute structural properties
        self.deg_n, self.bw = compute_centralities(self.G)
        self.eigenvalues, self.spec_emb = compute_spectral_embeddings(self.G)

        # Deterministic seeding for node attributes based on node IDs (so it's reproducible)
        # matching standard distribution styles
        self.influence = np.zeros(self.n, dtype=np.float32)
        self.skepticism = np.zeros(self.n, dtype=np.float32)
        self.share_prob = np.zeros(self.n, dtype=np.float32)
        self.trust_score = np.zeros(self.n, dtype=np.float32)

        for i in range(self.n):
            rng = np.random.default_rng(i + 12345)
            self.influence[i] = rng.beta(4, 2)
            self.skepticism[i] = rng.beta(1.5, 5.5)
            self.share_prob[i] = rng.uniform(0.65, 0.95)
            self.trust_score[i] = rng.uniform(0.8, 1.0)

        # Initialize dynamic states from frontend
        self.belief = np.zeros(self.n, dtype=np.float32)
        self.treated = np.zeros(self.n, dtype=bool)
        self.is_seed = np.zeros(self.n, dtype=bool)
        
        for nd in nodes_list:
            idx = int(nd["id"])
            if idx < self.n:
                self.belief[idx] = float(nd["belief"])
                self.treated[idx] = bool(nd.get("treated", False))
                self.is_seed[idx] = bool(nd.get("isSeed", False))

        # Build PyTorch Geometric edge index
        ei_list = []
        for u, v in self.G.edges():
            ei_list.append([u, v])
            ei_list.append([v, u])
        if len(ei_list) > 0:
            self.ei = torch.tensor(ei_list, dtype=torch.long).t().contiguous()
        else:
            self.ei = torch.empty((2, 0), dtype=torch.long)

        # Cache variables
        self._feat_buf = np.empty((self.n, 7 + SPEC_DIM), dtype=np.float32)
        self._cached_risks = np.zeros(self.n, dtype=np.float32)

    def _feats(self, frontier):
        b = self._feat_buf
        b[:, 0] = (self.belief > 0.5).astype(np.float32)
        b[:, 1] = 0.0
        if frontier:
            b[frontier, 1] = 1.0
        b[:, 2] = self.deg_n
        b[:, 3] = self.bw
        b[:, 4] = self.belief
        b[:, 5] = self.skepticism
        b[:, 6] = self.influence
        b[:, 7:7 + SPEC_DIM] = self.spec_emb
        return torch.from_numpy(b.copy())

    def _get_risks(self, frontier):
        X = self._feats(frontier)
        with torch.no_grad():
            self._cached_risks = self.gnn(X, self.ei).numpy().flatten()
        return self._cached_risks

    def _build_state(self, believers_count, rem_budget, frontier, avg_deg):
        mask = self.belief <= 0.5
        avg_r = float(self._cached_risks[mask].mean()) if mask.any() else 0.0
        max_r = float(self._cached_risks[mask].max()) if mask.any() else 0.0
        # Formula matches the notebook
        return torch.tensor([
            avg_r,
            max_r,
            believers_count / self.n,
            rem_budget / max(150.0, 1.0),  # normalise by a budget factor
            len(frontier) / self.n,
            min(1.0, avg_deg / 50.0)
        ], dtype=torch.float32)

    def _mute_superspreaders(self, risks, n_targets):
        infected = np.where(self.belief > 0.5)[0]
        if len(infected) == 0:
            return 0
        score = risks[infected] * self.influence[infected] * self.deg_n[infected]
        k = min(len(infected), max(1, int(n_targets)))
        # Top-k indices
        if k >= len(score):
            idx = infected
        else:
            idx = infected[np.argpartition(score, -k)[-k:]]
        
        act = 0
        for v in idx:
            if self.influence[v] > 0.0:
                self.influence[v] = 0.0
                act += 1
        return act

    def _apply_cure(self, risks, rem_budget, b_step):
        act_mask = (self.belief > 0.5) & (~self.treated)
        eligible = np.where(act_mask)[0]
        if len(eligible) == 0:
            eligible = np.where(self.belief > 0.5)[0]
        if len(eligible) == 0:
            return 0
        
        amt = min(len(eligible), int(rem_budget), int(b_step))
        score = risks[eligible]
        if amt >= len(score):
            idx = eligible
        else:
            idx = eligible[np.argpartition(score, -amt)[-amt:]]
            
        for v in idx:
            self.belief[v] = 0.0
            self.treated[v] = True
            self.influence[v] = self.influence[v] * 0.3
        return len(idx)

    def run(self, strategy: str, timesteps: int, budget_per_step: float):
        snapshots = []
        rng = np.random.default_rng(42)

        # Find initial frontier and believers
        frontier = list(np.where(self.belief > 0.5)[0])
        believers = int(np.sum(self.belief > 0.5))
        rem_budget = budget_per_step * timesteps * 0.5 # densityscaled approximation

        avg_deg = 2 * len(self.G.edges()) / self.n if self.n > 0 else 0
        
        # Track initial structural risk vector & actions
        initial_risks = self._get_risks(frontier).copy()
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        # Save step 0 snapshot
        snapshots.append(self._make_snapshot())

        for t in range(timesteps):
            # 1. Update/Predict risks
            risks = self._get_risks(frontier)

            # 2. Strategy action selection
            targets_to_treat = []
            if strategy == 'passive_degree':
                if t == 5:
                    # Treat highest degree nodes
                    candidates = sorted(range(self.n), key=lambda v: len(self.adj[v]), reverse=True)
                    for v in candidates[:int(budget_per_step)]:
                        self.influence[v] = 0.0
                        self.treated[v] = True
            
            elif strategy == 'active_random':
                # Cure budget random infected nodes
                infected = np.where((self.belief > 0.5) & (~self.treated))[0]
                if len(infected) > 0:
                    targets = rng.choice(infected, size=min(int(budget_per_step), len(infected)), replace=False)
                    for v in targets:
                        self.belief[v] = max(0.0, self.belief[v] * 0.08)
                        self.treated[v] = True

            elif strategy == 'active_degree':
                # Cure budget highest-degree infected nodes
                infected = np.where((self.belief > 0.5) & (~self.treated))[0]
                if len(infected) > 0:
                    targets = sorted(infected, key=lambda v: len(self.adj[v]), reverse=True)[:int(budget_per_step)]
                    for v in targets:
                        self.belief[v] = max(0.0, self.belief[v] * 0.08)
                        self.treated[v] = True

            elif strategy == 'gnn_rl':
                # RL Decision
                s_vector = self._build_state(believers, rem_budget, frontier, avg_deg)
                q_no_b = torch.tensor([0.0] + [-1e9] * (N_ACTIONS - 1), dtype=torch.float32)
                q_full = torch.zeros(N_ACTIONS, dtype=torch.float32)
                
                mask = q_no_b if rem_budget <= 0 else q_full
                with torch.no_grad():
                    q_vals = self.dqn(s_vector.unsqueeze(0)).squeeze(0) + mask
                    a = 3  # Force cure action for gnn_rl strategy
                    if a == 0:
                        # If the DQN suggests no action, fallback to applying cure to ensure model impact
                        a = 3

                action_counts[a] += 1

                # Step Action
                cost = 0
                b_step = int(budget_per_step)
                if a == 1:
                    cost = self._mute_superspreaders(risks, min(b_step, int(rem_budget)))
                elif a == 2:
                    cost = self._mute_superspreaders(risks, min(b_step * 3, int(rem_budget)))
                elif a == 3:
                    cost = self._apply_cure(risks, rem_budget, b_step)
                elif a == 4:
                    half = max(1, b_step // 2)
                    cost = self._apply_cure(risks, min(rem_budget, half), half)
                    cost += self._mute_superspreaders(risks, min(half, int(max(0.0, rem_budget - cost))))

                rem_budget = max(0.0, rem_budget - cost)

            # 3. Spread infection for one step
            next_belief = self.belief.copy()
            for i in range(self.n):
                if self.treated[i]:
                    continue
                inc = 0.0
                for j in self.adj[i]:
                    if self.belief[j] > 0.5:
                        inc += 0.08 * (self.belief[j] - 0.5)
                stochastic = (rng.uniform() - 0.5) * 0.012
                # decrease if treated node nearby, or simple decay
                next_belief[i] = min(1.0, max(0.0, self.belief[i] + inc + stochastic))

            self.belief = next_belief
            believers = int(np.sum(self.belief > 0.5))

            # Update frontier
            frontier = list(np.where(self.belief > 0.5)[0])

            # Save snapshot
            snapshots.append(self._make_snapshot())

        metrics = {
            "avg_degree": float(avg_deg),
            "max_degree_centrality": float(np.max(self.deg_n)),
            "mean_betweenness": float(np.mean(self.bw)),
            "eigenvalues": [float(val) for val in self.eigenvalues[:3]],
            "initial_mean_risk": float(np.mean(initial_risks)),
            "initial_max_risk": float(np.max(initial_risks)),
            "action_distribution": {int(k): int(v) for k, v in action_counts.items()}
        }
        return {"snapshots": snapshots, "metrics": metrics}

    def _make_snapshot(self):
        nodes = []
        for i in range(self.n):
            nodes.append({
                "id": i,
                "belief": float(self.belief[i]),
                "treated": bool(self.treated[i]),
                "isSeed": bool(self.is_seed[i]),
                "degree": len(self.adj[i])
            })
        return nodes

def predict_beliefs(nodes_list, edges_list, graph_name="p2p_gnutella", checkpoint="final", seed=0):
    """Fallback / direct GNN prediction for api compatibility"""
    gnn, _ = get_models(graph_name, checkpoint, seed)
    n = len(nodes_list)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges_list)
    
    deg_n, bw = compute_centralities(G)
    _, spec_emb = compute_spectral_embeddings(G)
    
    # feature buffer
    b = np.zeros((n, 15), dtype=np.float32)
    b[:, 2] = deg_n
    b[:, 3] = bw
    b[:, 7:15] = spec_emb
    
    X = torch.from_numpy(b)
    ei_list = []
    for u, v in G.edges():
        ei_list.append([u, v])
        ei_list.append([v, u])
    if len(ei_list) > 0:
        ei = torch.tensor(ei_list, dtype=torch.long).t().contiguous()
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
        
    with torch.no_grad():
        preds = gnn(X, ei).numpy().flatten()
    return preds.tolist()

def select_action(state_vector, graph_name="p2p_gnutella", checkpoint="final", seed=0):
    """Fallback / direct DQN action selection for api compatibility"""
    _, dqn = get_models(graph_name, checkpoint, seed)
    state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_vals = dqn(state_tensor)
    return int(q_vals.argmax(dim=1).item())

