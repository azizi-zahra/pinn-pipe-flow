"""
Tests for training/losses.py.

Verifies that individual loss functions return non-negative scalar tensors
and that total_loss returns a correctly structured dictionary with proper types.
"""

import pytest
import torch

from pinn_pipe.models import MLP
from pinn_pipe.training import (
    bc_inlet_loss,
    bc_symmetry_loss,
    bc_wall_u_loss,
    bc_wall_v_loss,
    physics_loss_continuity,
    physics_loss_momentum,
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
        L=20.0,
        nu=0.01,
        Re_min=100.0,
        Re_max=500.0,
    )


@pytest.fixture
def training_config():
    """Returns a default TrainingConfig for testing."""
    return TrainingConfig(
        epochs=100,
        learning_rate=1e-3,
        optimizer="adam",
        loss_weight_physics=1.0,
        loss_weight_continuity=1.0,
        loss_weight_bc_wall=1.0,
        loss_weight_bc_wall_v=1.0,
        loss_weight_bc_symmetry=1.0,
        loss_weight_bc_inlet=10.0,
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
    """Returns interior radial points with requires_grad=True."""
    r = torch.linspace(0.01, 0.99, 100).reshape(-1, 1)
    r.requires_grad_(True)
    return r


@pytest.fixture
def x():
    """Returns interior axial points with requires_grad=True."""
    x = torch.linspace(0.1, 19.9, 100).reshape(-1, 1)
    x.requires_grad_(True)
    return x


@pytest.fixture
def Re():
    """Returns interior Reynolds numbers."""
    return torch.full((100, 1), 200.0)


@pytest.fixture
def r_wall():
    """Returns wall radial points with requires_grad=True."""
    r = torch.ones(100, 1)
    r.requires_grad_(True)
    return r


@pytest.fixture
def x_bc_wall():
    """Returns wall axial points."""
    return torch.linspace(0.0, 20.0, 100).reshape(-1, 1)


@pytest.fixture
def Re_bc_wall():
    """Returns wall Reynolds numbers."""
    return torch.full((100, 1), 200.0)


@pytest.fixture
def r_sym():
    """Returns centerline radial points with requires_grad=True."""
    r = torch.zeros(100, 1)
    r.requires_grad_(True)
    return r


@pytest.fixture
def x_bc_sym():
    """Returns centerline axial points."""
    return torch.linspace(0.0, 20.0, 100).reshape(-1, 1)


@pytest.fixture
def Re_bc_sym():
    """Returns centerline Reynolds numbers."""
    return torch.full((100, 1), 200.0)


@pytest.fixture
def r_inlet():
    """Returns inlet radial points."""
    return torch.linspace(0.0, 1.0, 100).reshape(-1, 1)


@pytest.fixture
def x_inlet():
    """Returns inlet axial points."""
    return torch.zeros(100, 1)


@pytest.fixture
def Re_bc_inlet():
    """Returns inlet Reynolds numbers."""
    return torch.full((100, 1), 200.0)


# -----------------------------------------------------------------------------
# Individual Loss Tests
# -----------------------------------------------------------------------------

def test_physics_loss_momentum(model, r, x, Re, physics_config):
    """physics_loss_momentum returns a non-negative scalar tensor."""
    loss = physics_loss_momentum(model, r, x, Re, physics_config)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss >= 0


def test_physics_loss_continuity(model, r, x, Re, physics_config):
    """physics_loss_continuity returns a non-negative scalar tensor."""
    loss = physics_loss_continuity(model, r, x, Re, physics_config)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss >= 0


def test_bc_wall_u_loss(model, r_wall, x_bc_wall, Re_bc_wall, physics_config):
    """bc_wall_u_loss returns a non-negative scalar tensor."""
    loss = bc_wall_u_loss(model, r_wall, x_bc_wall, Re_bc_wall, physics_config)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss >= 0


def test_bc_wall_v_loss(model, r_wall, x_bc_wall, Re_bc_wall, physics_config):
    """bc_wall_v_loss returns a non-negative scalar tensor."""
    loss = bc_wall_v_loss(model, r_wall, x_bc_wall, Re_bc_wall, physics_config)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss >= 0


def test_bc_symmetry_loss(model, r_sym, x_bc_sym, Re_bc_sym, physics_config):
    """bc_symmetry_loss returns a non-negative scalar tensor."""
    loss = bc_symmetry_loss(model, r_sym, x_bc_sym, Re_bc_sym, physics_config)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss >= 0


def test_bc_inlet_loss(model, r_inlet, x_inlet, Re_bc_inlet, physics_config):
    """bc_inlet_loss returns a non-negative scalar tensor."""
    loss = bc_inlet_loss(model, r_inlet, x_inlet, Re_bc_inlet, physics_config)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss >= 0


# -----------------------------------------------------------------------------
# Total Loss Tests
# -----------------------------------------------------------------------------

def test_total_loss_structure_and_values(
    model,
    r,
    x,
    Re,
    r_wall,
    x_bc_wall,
    Re_bc_wall,
    r_sym,
    x_bc_sym,
    Re_bc_sym,
    r_inlet,
    x_inlet,
    Re_bc_inlet,
    physics_config,
    training_config,
):
    """total_loss returns a dict with exact expected keys, types, and values >= 0."""
    losses = total_loss(
        model=model,
        r=r,
        x=x,
        Re=Re,
        r_wall=r_wall,
        x_bc_wall=x_bc_wall,
        Re_bc_wall=Re_bc_wall,
        r_sym=r_sym,
        x_bc_sym=x_bc_sym,
        Re_bc_sym=Re_bc_sym,
        r_inlet=r_inlet,
        x_inlet=x_inlet,
        Re_bc_inlet=Re_bc_inlet,
        physics_config=physics_config,
        training_config=training_config,
    )

    expected_keys = {
        "loss_total_tensor",
        "loss_total",
        "loss_momentum",
        "loss_continuity",
        "loss_bc_wall",
        "loss_bc_wall_v",
        "loss_bc_symmetry",
        "loss_bc_inlet",
    }
    assert set(losses.keys()) == expected_keys

    assert isinstance(losses["loss_total_tensor"], torch.Tensor)

    float_keys = [
        "loss_total",
        "loss_momentum",
        "loss_continuity",
        "loss_bc_wall",
        "loss_bc_wall_v",
        "loss_bc_symmetry",
        "loss_bc_inlet",
    ]
    for key in float_keys:
        assert isinstance(losses[key], float), f"Key {key} expected float, got {type(losses[key])}"

    assert losses["loss_total"] >= 0