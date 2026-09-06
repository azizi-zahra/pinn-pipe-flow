"""
Input/output utilities for pinn-pipe-flow.

Handles creation of run directories and saving of all experiment
artifacts: config, metrics, training history, and model weights.
"""

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime

import torch
import yaml

from pinn_pipe.utils.config import Config


def create_run_dir(experiment_name: str, results_dir: str = "results") -> str:
    """Creates a timestamped directory for a single experiment run.

    Also creates a plots/ subdirectory inside it.

    Args:
        experiment_name: The name field from the experiment config.
        results_dir: Root results directory. Defaults to "results".

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"
    run_dir = os.path.join(results_dir, run_name)

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    return run_dir


def save_config(config: Config, run_dir: str) -> None:
    """Saves the fully merged config to the run directory as config.yaml.

    Args:
        config: The Config object used for this run.
        run_dir: Path to the run directory.
    """
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)


def save_metrics(metrics: dict, run_dir: str) -> None:
    """Saves final evaluation metrics to the run directory as metrics.json.

    Args:
        metrics: Dictionary of metric names to values.
        run_dir: Path to the run directory.
    """
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_history(history: list[dict], run_dir: str) -> None:
    """Saves per-epoch training history to the run directory as history.csv.

    Each entry in history is a dict with keys like epoch, total_loss,
    physics_loss, bc_wall_loss, bc_symmetry_loss, learning_rate.

    Args:
        history: List of dicts, one per epoch.
        run_dir: Path to the run directory.
    """
    if not history:
        return

    history_path = os.path.join(run_dir, "history.csv")
    fieldnames = history[0].keys()

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_model(model: torch.nn.Module, run_dir: str) -> None:
    """Saves the model weights to the run directory as model.pt.

    Args:
        model: The trained PyTorch model.
        run_dir: Path to the run directory.
    """
    model_path = os.path.join(run_dir, "model.pt")
    torch.save(model.state_dict(), model_path)


def load_model(model: torch.nn.Module, run_dir: str) -> torch.nn.Module:
    """Loads saved weights into a model instance.

    The model must have the same architecture as the one that was saved.

    Args:
        model: An unloaded model instance with the correct architecture.
        run_dir: Path to the run directory containing model.pt.

    Returns:
        The model with loaded weights.
    """
    model_path = os.path.join(run_dir, "model.pt")
    model.load_state_dict(torch.load(model_path))
    return model