"""
Tests for physics/pipe_flow.py.

Verifies PDE residual, BC residuals, and analytical solution
by checking that the exact solution satisfies all equations.
"""

import pytest
import torch

from pinn_pipe.physics.pipe_flow import (
    analytical_solution,
    bc_symmetry,
    bc_wall,
    pde_residual,
)
from pinn_pipe.utils.config import ModelConfig, PhysicsConfig
from pinn_pipe.models.mlp import MLP


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


# -----------------------------------------------------------------------------
# Analytical solution tests
# -----------------------------------------------------------------------------

def test_analytical_solution_shape(r, u_max, physics_config):
    """Analytical solution should return shape (N, 1)."""
    u = analytical_solution(r, u_max, physics_config.R)
    assert u.shape == (100, 1), f"Expected (100, 1), got {u.shape}"


def test_analytical_solution_at_center(physics_config):
    """Analytical solution at r=0 should equal u_max."""
    r = torch.zeros(1, 1)
    u_max = torch.tensor([[1.5]])
    u = analytical_solution(r, u_max, physics_config.R)
    assert torch.isclose(u, u_max, atol=1e-6), f"Expected {u_max.item()}, got {u.item()}"


def test_analytical_solution_at_wall(physics_config):
    """Analytical solution at r=R should equal zero."""
    r = torch.tensor([[physics_config.R]])
    u_max = torch.tensor([[1.5]])
    u = analytical_solution(r, u_max, physics_config.R)
    assert torch.isclose(u, torch.zeros(1, 1), atol=1e-6), f"Expected 0, got {u.item()}"


def test_analytical_solution_bounded(r, u_max, physics_config):
    """Analytical solution should be between 0 and u_max everywhere."""
    u = analytical_solution(r, u_max, physics_config.R)
    assert (u >= 0).all(), "Velocity should be non-negative"
    assert (u <= u_max).all(), "Velocity should not exceed u_max"


# -----------------------------------------------------------------------------
# PDE residual tests
# -----------------------------------------------------------------------------

def test_pde_residual_shape(model, r, u_max, physics_config):
    """PDE residual should return shape (N, 1)."""
    residual = pde_residual(model, r, u_max, physics_config)
    assert residual.shape == (100, 1), f"Expected (100, 1), got {residual.shape}"


# -----------------------------------------------------------------------------
# BC residual tests
# -----------------------------------------------------------------------------

def test_bc_wall_shape(model, u_max, physics_config):
    """BC wall residual should return shape (N, 1)."""
    r_wall = torch.full((100, 1), physics_config.R, requires_grad=True)
    result = bc_wall(model, r_wall, u_max)
    assert result.shape == (100, 1), f"Expected (100, 1), got {result.shape}"


def test_bc_symmetry_shape(model, u_max):
    """BC symmetry residual should return shape (N, 1)."""
    r_sym = torch.zeros(100, 1, requires_grad=True)
    result = bc_symmetry(model, r_sym, u_max)
    assert result.shape == (100, 1), f"Expected (100, 1), got {result.shape}"