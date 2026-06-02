import argparse
import json
import pathlib
import numpy as np

# Placeholder utility to load weight files (.npz) and run inference.
# In a real implementation you would import your trained GNN/RL model
# and apply it to a graph dataset.

def load_weights(weights_dir: pathlib.Path):
    """Load all .npz weight files into a dict for later use.

    This function simply reads the numpy archives; actual model loading
    would depend on the framework (e.g., PyTorch, TensorFlow, JAX).
    """
    weight_data = {}
    for npz_file in weights_dir.glob("*.npz"):
        # np.load returns a dict‑like object with arrays
        try:
            weight_data[npz_file.stem] = np.load(npz_file, allow_pickle=True)
        except Exception as e:
            print(f"Failed to load {npz_file}: {e}")
    return weight_data


def run_inference(weights, graph_path: pathlib.Path):
    """Run a mock inference on the provided graph.

    For demonstration purposes this function returns fabricated summary
    statistics that mimic the structure of `summary_results.json`.
    Replace this with your actual model evaluation logic.
    """
    # In a real scenario you would parse the graph, create tensors, and
    # feed them through the GNN+RL policy.
    # Here we return placeholder numbers.
    return {
        "nodes": 0,
        "edges": 0,
        "budget_per_step": 0,
        "episodes": 0,
        "gnn_rl": {"median": 0.0, "std": 0.0, "suppression_pct": 0.0},
        "baselines": {"none": 0.0, "passive_degree": 0.0, "active_random": 0.0, "active_degree": 0.0},
    }


def generate_summary(weights_dir: pathlib.Path, graph_path: pathlib.Path, out_path: pathlib.Path):
    """Generate a summary JSON for a single graph and write it to `out_path`.

    The output matches the schema used by the front‑end:
    {
        "graph_name": { ...statistics... }
    }
    """
    weights = load_weights(weights_dir)
    summary = {}
    # For a single dedicated dataset we use its filename (without extension) as the key
    graph_name = graph_path.stem
    summary[graph_name] = run_inference(weights, graph_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate summary results from pretrained weights for a custom graph dataset.")
    parser.add_argument("--weights", type=str, default="results/weights", help="Directory containing .npz weight files.")
    parser.add_argument("--graph", type=str, required=True, help="Path to the custom graph file (e.g., edge list).")
    parser.add_argument("--out", type=str, default="public/weights/custom_results.json", help="Output JSON file for the dashboard.")
    args = parser.parse_args()

    generate_summary(pathlib.Path(args.weights), pathlib.Path(args.graph), pathlib.Path(args.out))


if __name__ == "__main__":
    main()
