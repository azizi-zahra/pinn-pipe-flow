"""
Configuration loading and validation for pinn-pipe-flow.

Merges base.yaml and an experiment-specific yaml into a single
validated Config dataclass that is passed through the entire pipeline.
"""

import os
import yaml
from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    R: float          # pipe radius
    mu: float         # dynamic viscosity
    u_max_min: float  # minimum u_max sampled during training
    u_max_max: float  # maximum u_max sampled during training


@dataclass
class ModelConfig:
    type: str               # model architecture, e.g. "mlp"
    hidden_layer_depth: int
    hidden_layer_width: int
    activation: str         # e.g. "tanh"


@dataclass
class SamplingConfig:
    num_interior_points: int


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    optimizer: str
    loss_weight_physics: float
    loss_weight_bc_wall: float
    loss_weight_bc_symmetry: float


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
            mu=base_data["physics"]["mu"],
            u_max_min=base_data["physics"]["u_max_range"]["min"],
            u_max_max=base_data["physics"]["u_max_range"]["max"],
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
            loss_weight_physics=base_data["training"]["loss_weights"]["physics"],
            loss_weight_bc_wall=base_data["training"]["loss_weights"]["bc_wall"],
            loss_weight_bc_symmetry=base_data["training"]["loss_weights"]["bc_symmetry"],
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

    if config.physics.mu <= 0:
        raise ValueError(f"mu must be positive, got {config.physics.mu}")

    if config.physics.u_max_min >= config.physics.u_max_max:
        raise ValueError(
            f"u_max_range.min must be less than u_max_range.max, "
            f"got min={config.physics.u_max_min}, max={config.physics.u_max_max}"
        )

    if config.training.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.training.epochs}")

    if config.training.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.training.learning_rate}")