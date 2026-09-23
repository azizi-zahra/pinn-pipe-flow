"""
Tests for training/sampler.py.

Verifies that interior, wall, symmetry, and inlet collocation points
are sampled with correct shapes, ranges, and gradient settings.
"""

import pytest
import torch

from pinn_pipe.models import MLP
from pinn_pipe.training import (
    Trainer,
    sample_bc_inlet,
    sample_bc_symmetry,
    sample_bc_wall,
    sample_interior,
)
from pinn_pipe.utils import (
    Config,
    ModelConfig,
    PhysicsConfig,
    RunConfig,
    SamplingConfig,
    TrainingConfig,
    validate_config,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def physics_config():
    """Returns a default PhysicsConfig for testing."""
    return PhysicsConfig(
        R=1.0,
        L=20.0,
        nu=0.01,
        Re_min=100.0,
        Re_max=500.0,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_sample_interior(physics_config):
    """sample_interior returns (r, x, Re) with correct shapes, ranges, and gradient flags."""
    r, x, Re = sample_interior(n=100, config=physics_config)

    assert r.shape == (100, 1)
    assert x.shape == (100, 1)
    assert Re.shape == (100, 1)

    assert (r > 0).all() and (r <= physics_config.R).all(), "r should be in (0, R)"
    assert (x > 0).all() and (x <= physics_config.L).all(), "x should be in (0, L)"
    assert (Re >= physics_config.Re_min).all() and (Re <= physics_config.Re_max).all(), "Re out of bounds"

    assert r.requires_grad is True
    assert x.requires_grad is True


def test_sample_bc_wall(physics_config):
    """sample_bc_wall returns (r_wall, x_bc, Re_bc) with wall values and gradient settings."""
    r_wall, x_bc, Re_bc = sample_bc_wall(n=100, config=physics_config)

    assert r_wall.shape == (100, 1)
    assert x_bc.shape == (100, 1)
    assert Re_bc.shape == (100, 1)

    assert torch.all(r_wall == physics_config.R), "r_wall should all equal R"
    assert (x_bc >= 0).all() and (x_bc <= physics_config.L).all(), "x_bc should be in (0, L)"
    assert (Re_bc >= physics_config.Re_min).all() and (Re_bc <= physics_config.Re_max).all(), "Re_bc out of bounds"

    assert r_wall.requires_grad is True


def test_sample_bc_symmetry(physics_config):
    """sample_bc_symmetry returns (r_sym, x_bc, Re_bc) with symmetry values and gradient settings."""
    r_sym, x_bc, Re_bc = sample_bc_symmetry(n=100, config=physics_config)

    assert r_sym.shape == (100, 1)
    assert x_bc.shape == (100, 1)
    assert Re_bc.shape == (100, 1)

    assert torch.all(r_sym == 0), "r_sym should all equal 0"
    assert (x_bc >= 0).all() and (x_bc <= physics_config.L).all(), "x_bc should be in (0, L)"
    assert (Re_bc >= physics_config.Re_min).all() and (Re_bc <= physics_config.Re_max).all(), "Re_bc out of bounds"

    assert r_sym.requires_grad is True


def test_sample_bc_inlet(physics_config):
    """sample_bc_inlet returns (r_inlet, x_inlet, Re_bc) with inlet values."""
    r_inlet, x_inlet, Re_bc = sample_bc_inlet(n=100, config=physics_config)

    assert r_inlet.shape == (100, 1)
    assert x_inlet.shape == (100, 1)
    assert Re_bc.shape == (100, 1)

    assert torch.all(x_inlet == 0), "x_inlet should all equal 0"
    assert (r_inlet >= 0).all() and (r_inlet <= physics_config.R).all(), "r_inlet should be in (0, R)"
    assert (Re_bc >= physics_config.Re_min).all() and (Re_bc <= physics_config.Re_max).all(), "Re_bc out of bounds"


def test_sampling_config_defaults():
    """SamplingConfig defaults boundary point counts to num_interior_points."""
    config = SamplingConfig(num_interior_points=200)

    assert config.num_interior_points == 200
    assert config.num_bc_points is None
    assert config.num_bc_wall_points == 200
    assert config.num_bc_symmetry_points == 200
    assert config.num_bc_inlet_points == 200


def test_sampling_config_num_bc_points_fallback():
    """SamplingConfig defaults BC counts to num_bc_points when provided."""
    config = SamplingConfig(num_interior_points=200, num_bc_points=400)

    assert config.num_interior_points == 200
    assert config.num_bc_points == 400
    assert config.num_bc_wall_points == 400
    assert config.num_bc_symmetry_points == 400
    assert config.num_bc_inlet_points == 400


def test_sampling_config_granular_overrides():
    """SamplingConfig honors granular per-boundary point counts."""
    config = SamplingConfig(
        num_interior_points=200,
        num_bc_points=400,
        num_bc_wall_points=500,
        num_bc_symmetry_points=300,
        num_bc_inlet_points=600,
    )

    assert config.num_interior_points == 200
    assert config.num_bc_points == 400
    assert config.num_bc_wall_points == 500
    assert config.num_bc_symmetry_points == 300
    assert config.num_bc_inlet_points == 600


def test_validate_config_sampling():
    """validate_config raises ValueError for non-positive sampling point counts."""
    def make_config(sampling_config: SamplingConfig) -> Config:
        return Config(
            physics=PhysicsConfig(R=1.0, L=20.0, nu=0.01, Re_min=100.0, Re_max=500.0),
            model=ModelConfig(type="mlp", hidden_layer_depth=2, hidden_layer_width=16, activation="tanh"),
            sampling=sampling_config,
            training=TrainingConfig(
                epochs=10,
                learning_rate=0.001,
                optimizer="adam",
                loss_weight_physics=1.0,
                loss_weight_continuity=1.0,
                loss_weight_bc_wall=1.0,
                loss_weight_bc_wall_v=1.0,
                loss_weight_bc_symmetry=1.0,
                loss_weight_bc_inlet=10.0,
            ),
            run=RunConfig(seed=42, dtype="float32"),
        )

    with pytest.raises(ValueError, match="num_interior_points must be positive"):
        validate_config(make_config(SamplingConfig(num_interior_points=0)))

    with pytest.raises(ValueError, match="num_bc_points must be positive"):
        cfg = make_config(SamplingConfig(num_interior_points=100, num_bc_points=0))
        validate_config(cfg)

    with pytest.raises(ValueError, match="num_bc_wall_points must be positive"):
        cfg = make_config(SamplingConfig(num_interior_points=100, num_bc_wall_points=-5))
        validate_config(cfg)


def test_sampling_different_counts(physics_config):
    """Each sampler correctly generates specified distinct point counts."""
    n_interior = 500
    n_wall = 300
    n_sym = 250
    n_inlet = 150

    r, x, Re = sample_interior(n=n_interior, config=physics_config)
    r_wall, x_wall, Re_wall = sample_bc_wall(n=n_wall, config=physics_config)
    r_sym, x_sym, Re_sym = sample_bc_symmetry(n=n_sym, config=physics_config)
    r_inlet, x_inlet, Re_inlet = sample_bc_inlet(n=n_inlet, config=physics_config)

    assert r.shape == (n_interior, 1)
    assert r_wall.shape == (n_wall, 1)
    assert r_sym.shape == (n_sym, 1)
    assert r_inlet.shape == (n_inlet, 1)


def test_trainer_with_different_sampling_counts(tmp_path):
    """Trainer successfully trains with different point counts across samplers."""
    config = Config(
        physics=PhysicsConfig(R=1.0, L=20.0, nu=0.01, Re_min=100.0, Re_max=500.0),
        model=ModelConfig(type="mlp", hidden_layer_depth=2, hidden_layer_width=16, activation="tanh"),
        sampling=SamplingConfig(
            num_interior_points=100,
            num_bc_wall_points=80,
            num_bc_symmetry_points=60,
            num_bc_inlet_points=40,
        ),
        training=TrainingConfig(
            epochs=3,
            learning_rate=0.001,
            optimizer="hybrid",
            hybrid_switch_epoch=2,
            lbfgs_learning_rate=0.1,
            loss_weight_physics=1.0,
            loss_weight_continuity=1.0,
            loss_weight_bc_wall=1.0,
            loss_weight_bc_wall_v=1.0,
            loss_weight_bc_symmetry=1.0,
            loss_weight_bc_inlet=10.0,
        ),
        run=RunConfig(seed=42, dtype="float32"),
    )

    model = MLP(config.model)
    trainer = Trainer(model, config, str(tmp_path), torch.device("cpu"))
    trainer.train()

    assert len(trainer.history) == 3
