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
import matplotlib.ticker as ticker
import yaml


def collect_runs(results_dir: str) -> list[dict]:
    """Scans results_dir and collects metrics and config from each run.

    Keeps only the most recent run for each experiment name.

    Args:
        results_dir: Path to the results directory.

    Returns:
        List of dicts, one per experiment, containing the most recent run.
    """
    # collect all runs grouped by experiment name
    runs_by_experiment = {}

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

        # strip timestamp to get experiment name
        # run_name format: exp_001_mlp_baseline_YYYYMMDD_HHMMSS
        experiment_name = "_".join(run_name.split("_")[:-2])

        runs_by_experiment[experiment_name] = {
            "run_name": run_name,
            "experiment_name": experiment_name,
            "config": config,
            "metrics": metrics,
        }

    # return only the most recent run per experiment (sorted keeps last)
    return list(runs_by_experiment.values())


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
        "experiment_name",
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
                "experiment_name": run["experiment_name"],
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
    """Plots a horizontal bar chart comparing l2_error across all runs.

    Uses a log scale on the x-axis so outlier experiments do not
    dominate the plot and make good results unreadable.

    Args:
        runs: List of run dicts from collect_runs.
        results_dir: Path to the results directory.
    """
    if not runs:
        print("No runs found.")
        return

    experiment_names = [run["experiment_name"] for run in runs]
    l2_errors = [run["metrics"].get("l2_error", 0) for run in runs]

    fig_height = max(6, len(runs) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(
        experiment_names,
        l2_errors,
        color="#2b5c8f",
        edgecolor="none",
        height=0.65,
    )
    ax.invert_yaxis()

    # log scale so outliers don't dominate
    ax.set_xscale("log")

    ax.set_xlabel("L2 Error (log scale)", fontsize=11)
    ax.set_ylabel("Experiment", fontsize=11)
    ax.set_title(
        "L2 Error Comparison Across Experiments",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="x", which="major", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.6, alpha=0.35)
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_path = os.path.join(results_dir, "comparison.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
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