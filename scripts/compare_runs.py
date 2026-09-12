"""
Compare results across multiple experiment runs.

Scans the results/ directory, collects metrics and configs from all runs,
produces a comparison table (CSV) and a bar chart (PNG).

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --results-dir results/
"""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import yaml


def collect_runs(results_dir: str) -> list[dict]:
    """Scans results_dir and collects metrics and config from each run.

    Args:
        results_dir: Path to the results directory.

    Returns:
        List of dicts, one per run, containing run name, config, and metrics.
    """
    runs = []

    for run_name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, run_name)

        if not os.path.isdir(run_dir):
            continue

        metrics_path = os.path.join(run_dir, "metrics.json")
        config_path = os.path.join(run_dir, "config.yaml")

        if not os.path.exists(metrics_path) or not os.path.exists(config_path):
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        runs.append({
            "run_name": run_name,
            "config": config,
            "metrics": metrics,
        })

    return runs


def save_comparison_table(runs: list[dict], results_dir: str) -> None:
    """Saves a CSV comparison table of all runs.

    Args:
        runs: List of run dicts from collect_runs.
        results_dir: Path to the results directory.
    """
    if not runs:
        print("No runs found.")
        return

    output_path = os.path.join(results_dir, "comparison.csv")

    fieldnames = [
        "run_name",
        "model_type",
        "hidden_layer_depth",
        "hidden_layer_width",
        "activation",
        "epochs",
        "learning_rate",
        "optimizer",
        "loss_weight_physics",
        "loss_weight_bc_wall",
        "loss_weight_bc_symmetry",
        "l2_error",
        "max_error",
        "relative_l2_error",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run in runs:
            config = run["config"]
            metrics = run["metrics"]
            writer.writerow({
                "run_name": run["run_name"],
                "model_type": config["model"]["type"],
                "hidden_layer_depth": config["model"]["hidden_layer_depth"],
                "hidden_layer_width": config["model"]["hidden_layer_width"],
                "activation": config["model"]["activation"],
                "epochs": config["training"]["epochs"],
                "learning_rate": config["training"]["learning_rate"],
                "optimizer": config["training"]["optimizer"],
                "loss_weight_physics": config["training"]["loss_weight_physics"],
                "loss_weight_bc_wall": config["training"]["loss_weight_bc_wall"],
                "loss_weight_bc_symmetry": config["training"]["loss_weight_bc_symmetry"],
                "l2_error": metrics.get("l2_error", ""),
                "max_error": metrics.get("max_error", ""),
                "relative_l2_error": metrics.get("relative_l2_error", ""),
            })

    print(f"Comparison table saved to: {output_path}")


def plot_comparison(runs: list[dict], results_dir: str) -> None:
    """Plots a bar chart comparing l2_error across all runs.

    Args:
        runs: List of run dicts from collect_runs.
        results_dir: Path to the results directory.
    """
    if not runs:
        print("No runs found.")
        return

    run_names = [run["run_name"] for run in runs]
    l2_errors = [run["metrics"].get("l2_error", 0) for run in runs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(run_names, l2_errors)
    ax.set_xlabel("Run")
    ax.set_ylabel("L2 Error")
    ax.set_title("L2 Error Comparison Across Runs")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y")

    plt.tight_layout()

    output_path = os.path.join(results_dir, "comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Comparison plot saved to: {output_path}")


def main() -> None:
    """Collects all runs and produces a comparison table and plot."""
    parser = argparse.ArgumentParser(description="Compare results across experiment runs.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to the results directory. Defaults to 'results/'.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    runs = collect_runs(args.results_dir)

    if not runs:
        print("No completed runs found in results directory.")
        return

    print(f"Found {len(runs)} run(s).")

    save_comparison_table(runs, args.results_dir)
    plot_comparison(runs, args.results_dir)


if __name__ == "__main__":
    main()