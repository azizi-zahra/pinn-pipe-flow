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

from pinn_pipe.utils.config import (
    Config,
    ModelConfig,
    PhysicsConfig,
    RunConfig,
    SamplingConfig,
    TrainingConfig,
)


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
    loss_momentum, loss_continuity, loss_bc_wall, loss_bc_wall_v,
    loss_bc_symmetry, loss_bc_inlet, lr.

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


def load_run_config(run_dir: str) -> Config:
    """Loads a saved run configuration from config.yaml without merging.

    Reads results/<run_dir>/config.yaml (or {run_dir}/config.yaml),
    parses it with yaml.safe_load, and constructs the Config dataclass directly.

    Args:
        run_dir: Path to the run directory or run directory name.

    Returns:
        A fully populated Config object.

    Raises:
        FileNotFoundError: If config.yaml is not found.
    """
    if os.path.isfile(run_dir):
        config_path = run_dir
    elif os.path.isfile(os.path.join(run_dir, "config.yaml")):
        config_path = os.path.join(run_dir, "config.yaml")
    elif os.path.isfile(os.path.join("results", run_dir, "config.yaml")):
        config_path = os.path.join("results", run_dir, "config.yaml")
    else:
        raise FileNotFoundError(f"Run config not found in: {run_dir}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    physics_data = data.get("physics", {})
    if "Re_min" in physics_data:
        re_min = physics_data["Re_min"]
    elif "Re_range" in physics_data and isinstance(physics_data["Re_range"], dict):
        re_min = physics_data["Re_range"].get("min", 100.0)
    else:
        re_min = 100.0

    if "Re_max" in physics_data:
        re_max = physics_data["Re_max"]
    elif "Re_range" in physics_data and isinstance(physics_data["Re_range"], dict):
        re_max = physics_data["Re_range"].get("max", 500.0)
    else:
        re_max = 500.0

    physics = PhysicsConfig(
        R=float(physics_data.get("R", 1.0)),
        L=float(physics_data.get("L", 20.0)),
        nu=float(physics_data.get("nu", 0.01)),
        Re_min=float(re_min),
        Re_max=float(re_max),
    )

    model_data = data.get("model", {})
    model = ModelConfig(
        type=str(model_data["type"]),
        hidden_layer_depth=int(model_data["hidden_layer_depth"]),
        hidden_layer_width=int(model_data["hidden_layer_width"]),
        activation=str(model_data["activation"]),
    )

    sampling_data = data.get("sampling", {})
    sampling = SamplingConfig(
        num_interior_points=int(sampling_data["num_interior_points"]),
        num_bc_points=(
            int(sampling_data["num_bc_points"])
            if sampling_data.get("num_bc_points") is not None
            else None
        ),
        num_bc_wall_points=(
            int(sampling_data["num_bc_wall_points"])
            if sampling_data.get("num_bc_wall_points") is not None
            else None
        ),
        num_bc_symmetry_points=(
            int(sampling_data["num_bc_symmetry_points"])
            if sampling_data.get("num_bc_symmetry_points") is not None
            else None
        ),
        num_bc_inlet_points=(
            int(sampling_data["num_bc_inlet_points"])
            if sampling_data.get("num_bc_inlet_points") is not None
            else None
        ),
    )

    training_data = data.get("training", {})
    training = TrainingConfig(
        epochs=int(training_data["epochs"]),
        learning_rate=float(training_data["learning_rate"]),
        optimizer=str(training_data["optimizer"]),
        loss_weight_physics=float(training_data["loss_weight_physics"]),
        loss_weight_continuity=float(training_data.get("loss_weight_continuity", 1.0)),
        loss_weight_bc_wall=float(training_data["loss_weight_bc_wall"]),
        loss_weight_bc_wall_v=float(training_data.get("loss_weight_bc_wall_v", 1.0)),
        loss_weight_bc_symmetry=float(training_data["loss_weight_bc_symmetry"]),
        loss_weight_bc_inlet=float(training_data.get("loss_weight_bc_inlet", 10.0)),
        lr_scheduler=training_data.get("lr_scheduler", None),
        lr_scheduler_params=training_data.get("lr_scheduler_params", None),
        hybrid_switch_epoch=training_data.get("hybrid_switch_epoch", None),
        lbfgs_learning_rate=training_data.get("lbfgs_learning_rate", None),
    )

    run_data = data.get("run", {})
    run = RunConfig(
        seed=int(run_data["seed"]),
        dtype=str(run_data["dtype"]),
    )

    return Config(
        physics=physics,
        model=model,
        sampling=sampling,
        training=training,
        run=run,
    )