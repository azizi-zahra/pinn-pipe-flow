"""
Configuration loading and validation for pinn-pipe-flow.

Merges base.yaml and an experiment-specific yaml into a single
validated Config dataclass that is passed through the entire pipeline.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml


@dataclass
class PhysicsConfig:
    R: float          # pipe radius
    L: float          # pipe length (domain in x direction)
    nu: float         # kinematic viscosity
    Re_min: float     # minimum Re sampled during training
    Re_max: float     # maximum Re sampled during training


@dataclass
class ModelConfig:
    type: str               # model architecture, e.g. "mlp"
    hidden_layer_depth: int
    hidden_layer_width: int
    activation: str         # e.g. "tanh", "silu", "gelu", "sin", "mish"


@dataclass
class SamplingConfig:
    num_interior_points: int


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    optimizer: str          # "adam", "sgd", "lbfgs", "hybrid"
    loss_weight_physics: float
    loss_weight_continuity: float
    loss_weight_bc_wall: float
    loss_weight_bc_wall_v: float
    loss_weight_bc_symmetry: float
    loss_weight_bc_inlet: float
    lr_scheduler: Optional[str] = None
    lr_scheduler_params: Optional[Dict[str, Any]] = None
    hybrid_switch_epoch: Optional[int] = None
    lbfgs_learning_rate: Optional[float] = None


@dataclass
class RunConfig:
    seed: int
    dtype: str


@dataclass
class Config:
    physics:  PhysicsConfig
    model:    ModelConfig
    sampling: SamplingConfig
    training: TrainingConfig
    run:      RunConfig


def load_config(base_path: str, experiment_path: str) -> Config:
    """Loads and merges base and experiment configs into a Config object.

    The experiment config overrides the base config where values conflict.

    Args:
        base_path: Path to base.yaml.
        experiment_path: Path to the experiment yaml file.

    Returns:
        A fully populated Config object.

    Raises:
        FileNotFoundError: If either config file does not exist.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base config not found: {base_path}")

    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

    with open(base_path, "r") as f:
        base_data = yaml.safe_load(f)

    with open(experiment_path, "r") as f:
        experiment_data = yaml.safe_load(f)

    # deep merge: experiment overrides base section by section
    overrides = experiment_data.get("overrides") or {}
    for section, values in overrides.items():
        if section in base_data and isinstance(values, dict):
            base_data[section] = base_data[section] | values
        else:
            base_data[section] = values

    return Config(
        physics=PhysicsConfig(
            R=base_data["physics"]["R"],
            L=base_data["physics"]["L"],
            nu=base_data["physics"]["nu"],
            Re_min=base_data["physics"]["Re_range"]["min"],
            Re_max=base_data["physics"]["Re_range"]["max"],
        ),
        model=ModelConfig(
            type=base_data["model"]["type"],
            hidden_layer_depth=base_data["model"]["hidden_layer_depth"],
            hidden_layer_width=base_data["model"]["hidden_layer_width"],
            activation=base_data["model"]["activation"],
        ),
        sampling=SamplingConfig(
            num_interior_points=base_data["sampling"]["num_interior_points"],
        ),
        training=TrainingConfig(
            epochs=base_data["training"]["epochs"],
            learning_rate=base_data["training"]["learning_rate"],
            optimizer=base_data["training"]["optimizer"],
            loss_weight_physics=base_data["training"]["loss_weight_physics"],
            loss_weight_continuity=base_data["training"]["loss_weight_continuity"],
            loss_weight_bc_wall=base_data["training"]["loss_weight_bc_wall"],
            loss_weight_bc_wall_v=base_data["training"]["loss_weight_bc_wall_v"],
            loss_weight_bc_symmetry=base_data["training"]["loss_weight_bc_symmetry"],
            loss_weight_bc_inlet=base_data["training"]["loss_weight_bc_inlet"],
            lr_scheduler=base_data["training"].get("lr_scheduler", None),
            lr_scheduler_params=base_data["training"].get("lr_scheduler_params", None),
            hybrid_switch_epoch=base_data["training"].get("hybrid_switch_epoch", None),
            lbfgs_learning_rate=base_data["training"].get("lbfgs_learning_rate", None),
        ),
        run=RunConfig(
            seed=base_data["run"]["seed"],
            dtype=base_data["run"]["dtype"],
        ),
    )


def validate_config(config: Config) -> None:
    """Validates the config object.

    Args:
        config: A fully populated Config object.

    Raises:
        ValueError: If any config value is invalid.
    """
    if config.physics.R <= 0:
        raise ValueError(f"R must be positive, got {config.physics.R}")

    if config.physics.L <= 0:
        raise ValueError(f"L must be positive, got {config.physics.L}")

    if config.physics.nu <= 0:
        raise ValueError(f"nu must be positive, got {config.physics.nu}")

    if config.physics.Re_min < 1:
        raise ValueError(f"Re_min must be >= 1, got {config.physics.Re_min}")

    if config.physics.Re_max > 2300:
        raise ValueError(f"Re_max must be <= 2300, got {config.physics.Re_max}")

    if config.physics.Re_min >= config.physics.Re_max:
        raise ValueError(
            f"Re_min must be less than Re_max, "
            f"got min={config.physics.Re_min}, max={config.physics.Re_max}"
        )

    if config.training.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.training.epochs}")

    if config.training.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.training.learning_rate}")

    valid_optimizers = ["adam", "sgd", "lbfgs", "hybrid", "hybrid_adam_lbfgs"]
    if config.training.optimizer not in valid_optimizers:
        raise ValueError(
            f"optimizer must be one of {valid_optimizers}, got {config.training.optimizer}"
        )

    if config.training.lr_scheduler is not None:
        valid_schedulers = [
            "cosine", "cosine_annealing", "step", "multistep",
            "multi_step", "plateau", "reduce_on_plateau", "exponential"
        ]
        if config.training.lr_scheduler.lower() not in valid_schedulers:
            raise ValueError(
                f"lr_scheduler must be one of {valid_schedulers}, "
                f"got {config.training.lr_scheduler}"
            )

    if config.training.hybrid_switch_epoch is not None:
        if not (0 < config.training.hybrid_switch_epoch < config.training.epochs):
            raise ValueError(
                f"hybrid_switch_epoch must be between 1 and epochs ({config.training.epochs}), "
                f"got {config.training.hybrid_switch_epoch}"
            )