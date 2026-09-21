"""
Tests for physics/pipe_flow.py.

Verifies momentum and continuity PDE residuals and all boundary condition residuals.
"""

import pytest
import torch

from pinn_pipe.models import MLP
from pinn_pipe.physics import (
    bc_inlet_u,
    bc_symmetry_u,
    bc_wall_u,
    bc_wall_v,
    pde_residual_continuity,
    pde_residual_momentum,
)
from pinn_pipe.utils import ModelConfig, PhysicsConfig


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
    """Returns a batch of interior radial coordinates with requires_grad=True."""
    r = torch.linspace(0.01, 0.99, 100).reshape(-1, 1)
    r.requires_grad_(True)
    return r


@pytest.fixture
def x():
    """Returns a batch of interior axial coordinates with requires_grad=True."""
    x = torch.linspace(0.1, 19.9, 100).reshape(-1, 1)
    x.requires_grad_(True)
    return x


@pytest.fixture
def Re():
    """Returns a batch of Reynolds numbers."""
    return torch.full((100, 1), 200.0)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_pde_residual_momentum_shape(model, r, x, Re, physics_config):
    """pde_residual_momentum should return shape (100, 1)."""
    res = pde_residual_momentum(model, r, x, Re, physics_config)
    assert res.shape == (100, 1), f"Expected (100, 1), got {res.shape}"


def test_pde_residual_continuity_shape(model, r, x, Re, physics_config):
    """pde_residual_continuity should return shape (100, 1)."""
    res = pde_residual_continuity(model, r, x, Re, physics_config)
    assert res.shape == (100, 1), f"Expected (100, 1), got {res.shape}"


def test_bc_wall_u_shape(model, x, Re, physics_config):
    """bc_wall_u should return shape (100, 1) when given r_wall = full((100, 1), 1.0)."""
    r_wall = torch.full((100, 1), physics_config.R)
    res = bc_wall_u(model, r_wall, x, Re, physics_config)
    assert res.shape == (100, 1), f"Expected (100, 1), got {res.shape}"


def test_bc_wall_v_shape(model, x, Re, physics_config):
    """bc_wall_v should return shape (100, 1)."""
    r_wall = torch.full((100, 1), physics_config.R)
    res = bc_wall_v(model, r_wall, x, Re, physics_config)
    assert res.shape == (100, 1), f"Expected (100, 1), got {res.shape}"


def test_bc_symmetry_u_shape(model, x, Re, physics_config):
    """bc_symmetry_u should return shape (100, 1) when given r_sym = zeros(100, 1) requires_grad."""
    r_sym = torch.zeros(100, 1, requires_grad=True)
    res = bc_symmetry_u(model, r_sym, x, Re, physics_config)
    assert res.shape == (100, 1), f"Expected (100, 1), got {res.shape}"


def test_bc_inlet_u_shape(model, r, Re, physics_config):
    """bc_inlet_u should return shape (100, 1) when given x_inlet = zeros(100, 1)."""
    x_inlet = torch.zeros(100, 1)
    res = bc_inlet_u(model, r, x_inlet, Re, physics_config)
    assert res.shape == (100, 1), f"Expected (100, 1), got {res.shape}"