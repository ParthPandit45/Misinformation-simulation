# Misinformation Sandbox: GNN & Deep RL Containment Dashboard

---

## Overview

The **Misinformation Sandbox** is an interactive academic workbench and visualization dashboard for evaluating and deploying **Graph Neural Networks (GNN)** and **Deep Reinforcement Learning (DRL)** policies to suppress misinformation propagation across complex social networks.

---

## Key Features

- **Interactive Network Simulator** – Visualizes belief propagation dynamics on real‑world topologies (`p2p_gnutella`, `ca_grqc`, `facebook`) with play/pause controls, variable speeds, and live node‑state tracking.
- **Academic Experiment Workbench** – Run repeatable Monte‑Carlo baseline comparisons (Passive/Active Degree, Random, No‑Intervention, GNN+RL) and export results directly to publication‑ready CSV or JSON.
- **Zero‑Shot Evaluation Panel** – Generates arbitrary, unseen synthetic topologies (Barabási‑Albert, Erdős‑Rényi, Watts‑Strogatz) and runs zero‑shot inference with real‑time containment ratings.
- **Model Insight Visuals** – Reward/Loss curves, budget timelines, action distribution heat‑maps, and suppression rating charts.
- **Export‑Ready Outputs** – Download JSON snapshots, CSV tables, and high‑resolution PNGs for papers or presentations.

---

## Repository Structure

```text
mega-dashboard-repo/
├─ .gitignore                 # Ignored files (node_modules, build artefacts, etc.)
├─ README.md                  # This file
├─ package.json                # npm dependencies & scripts
├─ vite.config.js              # Vite configuration
├─ public/
│   └─ _redirects            # Netlify SPA fallback
├─ src/
│   ├─ index.css             # Global CSS variables (dark mode, theming)
│   ├─ main.jsx              # Application entry point
│   ├─ features/
│   │   └─ dashboard/
│   │       ├─ index.jsx      # Feature entrypoint
│   │       ├─ data/dashboardData.js  # Graph/strategy/action constants
│   │       └─ components/
│   │           ├─ layout/WorkspaceShell.jsx   # Sidebar + workspace layout
│   │           └─ research/   # Workbench panels
│   │               ├─ ResearchWorkbenchPanel.jsx
│   │               └─ ZeroShotPanel.jsx
│   ├─ components/
│   │   └─ Dashboard.jsx      # Main orchestrator (legacy location, kept for backward compatibility)
│   └─ utils/
│       ├─ simulation.js      # Curve generation utilities
│       └─ academicEvaluation.js  # Batch metrics, CSV export
├─ backend/
│   ├─ api.py                # FastAPI server entry point
│   └─ model_loader.py       # Model loading, inference, RL policy
└─ scripts/
    └─ export_weights.py     # Helper to export model weights & summary JSON
```

---

## Installation & Development

```bash
# Clone the repository
git clone https://github.com/yourorg/mega-dashboard-repo.git
cd mega-dashboard-repo/mega-repo

# Install dependencies (Node.js >= 20, Python >= 3.10)
npm install          # Front‑end packages
pip install -r backend/requirements.txt   # Python dependencies

# Start the FastAPI backend (default port 8000)
uvicorn backend.api:app --host 0.0.0.0 --port 8000

# In a separate terminal, start the React/Vite dev server
npm run dev
```

The application will be available at `http://localhost:5173` (Vite default) and will communicate with the backend at `http://localhost:8000`.

---

## Building for Production

```bash
npm run build   # Generates optimized assets in ./dist
# Deploy the `dist` folder to any static‑host (Netlify, Vercel, GitHub Pages, etc.)
```

A `netlify.toml` is provided for easy Netlify deployment; it sets the build command and publish directory.

---

## API Reference (FastAPI Backend)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/ping` | Health‑check – returns `{"status": "ok"}` |
| `POST` | `/gnn/predict` | Returns node positions, beliefs, and edges for a given graph/checkpoint/seed. Payload: `{graph, checkpoint, seed}` |
| `POST` | `/rl/action` | Returns the next action (0‑4) for a given state vector. Payload: `{state, graph, checkpoint, seed}` |
| `POST` | `/api/simulate` | Runs a full simulation (strategy, timesteps, budget, checkpoint, seed) and returns snapshots + metrics. |

All responses are JSON‑serialisable and can be consumed directly by the front‑end.

---

## Styling & Theming

The UI uses **CSS variables** defined in `src/index.css` for a dark‑mode friendly palette. Toggle the theme by updating `localStorage.theme` (`"dark"`/`"light"`).

---

## Data & Model Export

- **Export JSON** – The Zero‑Shot panel includes a *Download JSON* button that saves the complete evaluation payload (`{noneCurve, gnnCurve, rating, snapshots, …}`).
- **Export CSV** – The Research Workbench can export benchmark tables via the *Export CSV* button.
- **Model Weights** – Use `scripts/export_weights.py --summary summary_results.json --weights weights/` to generate `public/weights/results.json` which the dashboard will load automatically if present.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## Acknowledgements

- **RiskGNN** implementation is based on the work by **K. Liu et al., 2023**.
- **Dueling DDQN** code follows the architecture from **Van Hasselt et al., 2016**.
- Graph datasets (`p2p_gnutella`, `ca_grqc`, `facebook`) are courtesy of the **SNAP** collection.

---

*Enjoy exploring misinformation containment strategies!*
