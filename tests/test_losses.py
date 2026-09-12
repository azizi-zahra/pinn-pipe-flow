"""
Tests for training/losses.py.

Verifies that individual loss functions return non-negative scalars
and that total_loss returns a correctly structured dictionary.
"""

import pytest
import torch

from pinn_pipe.models import MLP
from pinn_pipe.training import (
    bc_symmetry_loss,
    bc_wall_loss,
    physics_loss,
    total_loss,
)
from pinn_pipe.utils import ModelConfig, PhysicsConfig, TrainingConfig


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def physics_config():
    """Returns a default PhysicsConfig for testing."""
    return PhysicsConfig(
        R=1.0,
        mu=1.0,
        u_max_min=0.5,
        u_max_max=2.0,
    )


@pytest.fixture
def training_config():
    """Returns a default TrainingConfig for testing."""
    return TrainingConfig(
        epochs=100,
        learning_rate=1e-3,
        optimizer="adam",
        loss_weight_physics=1.0,
        loss_weight_bc_wall=1.0,
        loss_weight_bc_symmetry=1.0,
    )


@pytest.fixture
def model():
    """Returns a default MLP instance for testing."""
    config = ModelConfig(
        type="mlp",
        hidden_layer_depth=3,
        hidden_layer_width=32,
        activation="tanh",
    )
    return MLP(config)


@pytest.fixture
def r():
    """Returns a batch of interior collocation points with requires_grad=True."""
    r = torch.linspace(0.01, 0.99, 100).reshape(-1, 1)
    r.requires_grad_(True)
    return r


@pytest.fixture
def u_max():
    """Returns a batch of u_max values."""
    return torch.full((100, 1), 1.0)


@pytest.fixture
def r_wall():
    """Returns a batch of wall BC points with requires_grad=True."""
    r = torch.ones(100, 1)  # R = 1.0
    r.requires_grad_(True)
    return r


@pytest.fixture
def r_sym():
    """Returns a batch of symmetry BC points with requires_grad=True."""
    r = torch.zeros(100, 1)
    r.requires_grad_(True)
    return r


# -----------------------------------------------------------------------------
# Individual loss tests
# -----------------------------------------------------------------------------

def test_physics_loss_is_scalar(model, r, u_max, physics_config):
    """physics_loss should return a scalar tensor."""
    loss = physics_loss(model, r, u_max, physics_config)
    assert loss.shape == torch.Size([])


def test_physics_loss_is_non_negative(model, r, u_max, physics_config):
    """physics_loss should be non-negative."""
    loss = physics_loss(model, r, u_max, physics_config)
    assert loss >= 0, "physics_loss should be non-negative"


def test_bc_wall_loss_is_scalar(model, r_wall, u_max):
    """bc_wall_loss should return a scalar tensor."""
    loss = bc_wall_loss(model, r_wall, u_max)
    assert loss.shape == torch.Size([])


def test_bc_wall_loss_is_non_negative(model, r_wall, u_max):
    """bc_wall_loss should be non-negative."""
    loss = bc_wall_loss(model, r_wall, u_max)
    assert loss >= 0, "bc_wall_loss should be non-negative"


def test_bc_symmetry_loss_is_scalar(model, r_sym, u_max):
    """bc_symmetry_loss should return a scalar tensor."""
    loss = bc_symmetry_loss(model, r_sym, u_max)
    assert loss.shape == torch.Size([])


def test_bc_symmetry_loss_is_non_negative(model, r_sym, u_max):
    """bc_symmetry_loss should be non-negative."""
    loss = bc_symmetry_loss(model, r_sym, u_max)
    assert loss >= 0, "bc_symmetry_loss should be non-negative"


# -----------------------------------------------------------------------------
# total_loss tests
# -----------------------------------------------------------------------------

def test_total_loss_returns_correct_keys(model, r, u_max, r_wall, r_sym, physics_config, training_config):
    """total_loss should return a dictionary with exactly the expected keys."""
    losses = total_loss(model, r, u_max, r_wall, r_sym, u_max, physics_config, training_config)
    expected_keys = {"loss_total_tensor", "loss_total", "loss_physics", "loss_bc_wall", "loss_bc_symmetry"}
    assert set(losses.keys()) == expected_keys


def test_total_loss_tensor_is_tensor(model, r, u_max, r_wall, r_sym, physics_config, training_config):
    """loss_total_tensor should be a PyTorch tensor for backpropagation."""
    losses = total_loss(model, r, u_max, r_wall, r_sym, u_max, physics_config, training_config)
    assert isinstance(losses["loss_total_tensor"], torch.Tensor)


def test_total_loss_scalars_are_floats(model, r, u_max, r_wall, r_sym, physics_config, training_config):
    """Logging loss values should be plain Python floats."""
    losses = total_loss(model, r, u_max, r_wall, r_sym, u_max, physics_config, training_config)
    assert isinstance(losses["loss_total"], float)
    assert isinstance(losses["loss_physics"], float)
    assert isinstance(losses["loss_bc_wall"], float)
    assert isinstance(losses["loss_bc_symmetry"], float)


def test_total_loss_is_non_negative(model, r, u_max, r_wall, r_sym, physics_config, training_config):
    """Total loss should be non-negative."""
    losses = total_loss(model, r, u_max, r_wall, r_sym, u_max, physics_config, training_config)
    assert losses["loss_total"] >= 0, "total_loss should be non-negative"