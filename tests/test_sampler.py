"""
Tests for training/sampler.py.

Verifies that interior and boundary condition points are sampled
with correct shapes, ranges, and gradient settings.
"""

import pytest
import torch

from pinn_pipe.training.sampler import sample_bc, sample_interior
from pinn_pipe.utils.config import PhysicsConfig, SamplingConfig


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
def sampling_config():
    """Returns a default SamplingConfig for testing."""
    return SamplingConfig(num_interior_points=100)


# -----------------------------------------------------------------------------
# sample_interior tests
# -----------------------------------------------------------------------------

def test_sample_interior_output_shapes(physics_config, sampling_config):
    """sample_interior should return r and u_max of shape (N, 1)."""
    r, u_max = sample_interior(
        n=sampling_config.num_interior_points,
        R=physics_config.R,
        u_max_min=physics_config.u_max_min,
        u_max_max=physics_config.u_max_max,
    )
    assert r.shape == (sampling_config.num_interior_points, 1)
    assert u_max.shape == (sampling_config.num_interior_points, 1)


def test_sample_interior_r_in_valid_range(physics_config, sampling_config):
    """Interior r values should be in (0, R)."""
    r, _ = sample_interior(
        n=sampling_config.num_interior_points,
        R=physics_config.R,
        u_max_min=physics_config.u_max_min,
        u_max_max=physics_config.u_max_max,
    )
    assert (r >= 0).all(), "r should be non-negative"
    assert (r <= physics_config.R).all(), "r should not exceed R"


def test_sample_interior_u_max_in_valid_range(physics_config, sampling_config):
    """Interior u_max values should be in (u_max_min, u_max_max)."""
    _, u_max = sample_interior(
        n=sampling_config.num_interior_points,
        R=physics_config.R,
        u_max_min=physics_config.u_max_min,
        u_max_max=physics_config.u_max_max,
    )
    assert (u_max >= physics_config.u_max_min).all()
    assert (u_max <= physics_config.u_max_max).all()


def test_sample_interior_r_requires_grad(physics_config, sampling_config):
    """Interior r should have requires_grad=True for autograd."""
    r, _ = sample_interior(
        n=sampling_config.num_interior_points,
        R=physics_config.R,
        u_max_min=physics_config.u_max_min,
        u_max_max=physics_config.u_max_max,
    )
    assert r.requires_grad, "r must have requires_grad=True"


# -----------------------------------------------------------------------------
# sample_bc tests
# -----------------------------------------------------------------------------

def test_sample_bc_output_shapes(physics_config, sampling_config):
    """sample_bc should return r_wall, r_sym, u_max_bc of shape (N, 1)."""
    r_wall, r_sym, u_max_bc = sample_bc(
        n=sampling_config.num_interior_points,
        R=physics_config.R,
        u_max_min=physics_config.u_max_min,
        u_max_max=physics_config.u_max_max,
    )
    assert r_wall.shape == (sampling_config.num_interior_points, 1)
    assert r_sym.shape == (sampling_config.num_interior_points, 1)
    assert u_max_bc.shape == (sampling_config.num_interior_points, 1)


def test_sample_bc_r_wall_equals_R(physics_config, sampling_config):
    """r_wall should be exactly R everywhere."""
    r_wall, _, _ = sample_bc(
        n=sampling_config.num_interior_points,
        R=physics_config.R,
        u_max_min=physics_config.u_max_min,
        u_max_max=physics_config.u_max_max,
    )
    assert torch.all(r_wall == physics_config.R), "r_wall should equal R everywhere"


def test_sample_bc_r_sym_equals_zero(physics_config, sampling_config):
    """r_sym should be exactly zero everywhere."""
    _, r_sym, _ = sample_bc(
        n=sampling_config.num_interior_points,
        R=physics_config.R,
        u_max_min=physics_config.u_max_min,
        u_max_max=physics_config.u_max_max,
    )
    assert torch.all(r_sym == 0), "r_sym should equal zero everywhere"


def test_sample_bc_u_max_in_valid_range(physics_config, sampling_config):
    """BC u_max values should be in (u_max_min, u_max_max)."""
    _, _, u_max_bc = sample_bc(
        n=sampling_config.num_interior_points,
        R=physics_config.R,
        u_max_min=physics_config.u_max_min,
        u_max_max=physics_config.u_max_max,
    )
    assert (u_max_bc >= physics_config.u_max_min).all()
    assert (u_max_bc <= physics_config.u_max_max).all()


def test_sample_bc_requires_grad(physics_config, sampling_config):
    """r_wall and r_sym should have requires_grad=True."""
    r_wall, r_sym, _ = sample_bc(
        n=sampling_config.num_interior_points,
        R=physics_config.R,
        u_max_min=physics_config.u_max_min,
        u_max_max=physics_config.u_max_max,
    )
    assert r_wall.requires_grad, "r_wall must have requires_grad=True"
    assert r_sym.requires_grad, "r_sym must have requires_grad=True"