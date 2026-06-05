"""
export_weights.py — After running the training notebook, use this to bundle results for the dashboard.

Usage:
    python scripts/export_weights.py --summary summary_results.json --weights weights/
"""
import argparse, json, shutil, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="summary_results.json")
    p.add_argument("--weights", default="weights/")
    p.add_argument("--plots",   default="plots/")
    p.add_argument("--out",     default="public/weights/results.json")
    args = p.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)

    out = {}
    for gname, gdata in summary.items():
        out[gname] = {k: gdata[k] for k in ("nodes","edges","budget_per_step","episodes","gnn_rl","baselines") if k in gdata}

    weight_dir = Path(args.weights)
    found = []
    models_meta = {}
    for g in ["p2p_gnutella", "ca_grqc", "facebook"]:
        models_meta[g] = {
            "checkpoints": [],
            "seeds": []
        }
        for seed in [0, 1, 2]:
            gnn = weight_dir / f"{g}_final_seed{seed}_gnn.pt"
            dqn = weight_dir / f"{g}_final_seed{seed}_dqn.pt"
            if gnn.exists() and dqn.exists():
                found.append(f"{g}_final_seed{seed}")
                print(f"  OK {g}_final_seed{seed} (GNN & DQN)")
                if seed not in models_meta[g]["seeds"]:
                    models_meta[g]["seeds"].append(seed)
            else:
                if seed < 2:
                    print(f"  MISSING {g}_final_seed{seed} (GNN or DQN)", file=sys.stderr)
        
        for p in weight_dir.glob(f"{g}_*.pt"):
            name = p.name
            if "final_seed" in name:
                continue
            if "_ep" in name:
                parts = name.split(f"{g}_")
                if len(parts) > 1:
                    ckpt_str = parts[1].split("_")[0]
                    if ckpt_str not in models_meta[g]["checkpoints"]:
                        models_meta[g]["checkpoints"].append(ckpt_str)
        
        models_meta[g]["seeds"].sort()
        models_meta[g]["checkpoints"].sort(key=lambda x: int(x[2:]) if x.startswith("ep") and x[2:].isdigit() else 0)
        models_meta[g]["checkpoints"].insert(0, "final")

    out["_meta"] = {"weight_files": found}
    out["_models_meta"] = models_meta

    out_plots = Path("public/weights/plots")
    out_plots.mkdir(parents=True, exist_ok=True)
    for png in Path(args.plots).glob("*.png"):
        shutil.copy(png, out_plots / png.name)
        print(f"  plot: {png.name}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone: {args.out}")

if __name__ == "__main__":
    main()
