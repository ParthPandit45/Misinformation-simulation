from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import os

# Import simulation and inference engines
from .model_loader import predict_beliefs, select_action, SimulationEngine

app = FastAPI(title="Misinformation Sandbox API Server")

# Enable CORS for frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GNNRequest(BaseModel):
    graph: str  # graph identifier, e.g., "p2p_gnutella"
    checkpoint: str = "final"
    seed: int = 0

class RLRequest(BaseModel):
    state: list  # list of floats representing the current state vector
    graph: str = "p2p_gnutella"
    checkpoint: str = "final"
    seed: int = 0

class SimulateRequest(BaseModel):
    graph_name: str
    strategy: str
    nodes: list
    edges: list
    timesteps: int = 50
    budget: float = 3.0
    checkpoint: str = "final"
    seed: int = 0

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/gnn/predict")
def gnn_predict(req: GNNRequest):
    try:
        # Dummy nodes for a quick shape prediction compatibility
        num_nodes = 80
        edges = []
        beliefs = predict_beliefs(list(range(num_nodes)), edges, req.graph, req.checkpoint, req.seed)
        nodes = [
            {
                "id": i,
                "x": (i % 10) / 9,
                "y": (i // 10) / 7,
                "belief": b,
                "degree": 0,
                "isSeed": i == 0,
                "treated": False,
            }
            for i, b in enumerate(beliefs)
        ]
        return {"nodes": nodes, "edges": edges, "seedIdx": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rl/action")
def rl_action(req: RLRequest):
    try:
        action = select_action(req.state, req.graph, req.checkpoint, req.seed)
        return {"action": action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate")
def api_simulate(req: SimulateRequest):
    """
    Simulates belief propagation on a given graph using actual trained models
    or classical baseline strategies. Returns snapshots and computation metrics.
    """
    try:
        engine = SimulationEngine(
            graph_name=req.graph_name,
            nodes_list=req.nodes,
            edges_list=req.edges,
            checkpoint=req.checkpoint,
            seed=req.seed
        )
        result = engine.run(
            strategy=req.strategy,
            timesteps=req.timesteps,
            budget_per_step=req.budget
        )
        # result is {"snapshots": [...], "metrics": {...}}
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

