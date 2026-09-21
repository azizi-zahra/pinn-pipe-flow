"""
Neural network architectures for pinn-pipe-flow.

Provides the base PINN interface and concrete model implementations
such as the fully-connected MLP.
"""

from pinn_pipe.models.base import BasePINN
from pinn_pipe.models.mlp import MLP, SinActivation

__all__ = [
    "BasePINN",
    "MLP",
    "SinActivation",
]
