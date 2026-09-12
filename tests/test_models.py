"""
Tests for models/base.py and models/mlp.py.

Checks model construction, output shapes, repr, and error handling.
"""

import pytest
import torch

from pinn_pipe.models.base import BasePINN
from pinn_pipe.models.mlp import MLP
from pinn_pipe.utils.config import ModelConfig


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def default_model_config():
    """Returns a default ModelConfig for testing."""
    return ModelConfig(
        type="mlp",
        hidden_layer_depth=3,
        hidden_layer_width=32,
        activation="tanh",
    )


@pytest.fixture
def model(default_model_config):
    """Returns a default MLP instance for testing."""
    return MLP(default_model_config)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_base_pinn_cannot_be_instantiated():
    """BasePINN is abstract and should raise TypeError when instantiated."""
    with pytest.raises(TypeError):
        BasePINN()


def test_mlp_output_shape(model):
    """MLP should output shape (N, 1) for input shape (N, 2)."""
    x = torch.rand(100, 2)
    output = model(x)
    assert output.shape == (100, 1), f"Expected (100, 1), got {output.shape}"


def test_mlp_forward_does_not_crash(model):
    """MLP forward pass should run without errors."""
    x = torch.rand(50, 2)
    output = model(x)
    assert output is not None


def test_mlp_repr(model):
    """MLP __repr__ should return a non-empty string."""
    repr_str = repr(model)
    assert isinstance(repr_str, str)
    assert len(repr_str) > 0


def test_mlp_unsupported_activation():
    """MLP should raise ValueError for unsupported activation functions."""
    config = ModelConfig(
        type="mlp",
        hidden_layer_depth=3,
        hidden_layer_width=32,
        activation="sine",
    )
    with pytest.raises(ValueError):
        MLP(config)


def test_mlp_single_input(model):
    """MLP should handle a single input point of shape (1, 2)."""
    x = torch.rand(1, 2)
    output = model(x)
    assert output.shape == (1, 1)


def test_mlp_different_batch_sizes(model):
    """MLP output shape should scale correctly with batch size."""
    for n in [1, 10, 100, 1000]:
        x = torch.rand(n, 2)
        output = model(x)
        assert output.shape == (n, 1), f"Failed for batch size {n}"