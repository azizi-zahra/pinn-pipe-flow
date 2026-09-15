"""
Hard Boundary Condition (Ansatz) model for pinn-pipe-flow.

Enforces Dirichlet wall boundary condition u(R) = 0 and Neumann centerline
symmetry boundary condition du/dr(0) = 0 strictly by architectural construction.
"""

import torch
import torch.nn as nn

from pinn_pipe.models.base import BasePINN
from pinn_pipe.models.mlp import MLP
from pinn_pipe.utils.config import ModelConfig


class HardBCMLP(BasePINN, nn.Module):
    """PINN model enforcing exact boundary conditions via a physics-informed ansatz.

    Ansatz formulation for Hagen-Poiseuille pipe flow:
        u_hat(r, u_max) = u_max * (1 - (r/R)^2) + (1 - (r/R)^2) * (r/R)^2 * NN(r, u_max)

    Properties:
        1. Outer wall (r = R):
           1 - (R/R)^2 = 0 ==> u_hat(R, u_max) = 0 (exact no-slip).
        2. Centerline symmetry (r = 0):
           u_hat(0, u_max) = u_max.
           d/dr [ (1 - (r/R)^2) * (r/R)^2 * NN ] at r=0 has a factor of r in every term,
           hence d(u_hat)/dr at r=0 is identically 0 (exact symmetry).

    This mathematically eliminates boundary condition error and allows setting
    loss_weight_bc_wall = 0 and loss_weight_bc_symmetry = 0.
    """

    def __init__(self, config: ModelConfig, R: float = 1.0) -> None:
        """Initializes HardBCMLP with the given ModelConfig and pipe radius.

        Args:
            config: ModelConfig specifying trunk network depth, width, and activation.
            R: Pipe radius (defaults to 1.0).
        """
        super().__init__()
        self.config = config
        self.R = float(getattr(config, "pipe_radius", R))
        self.trunk = MLP(config)

    def __repr__(self) -> str:
        """Returns a string summary of the model architecture."""
        return (
            f"HardBCMLP("
            f"R={self.R}, "
            f"trunk={self.trunk})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the predicted velocity satisfying hard boundary conditions.

        Args:
            x: Tensor of shape (N, 2) where each row is [r, u_max].

        Returns:
            Tensor of shape (N, 1) containing velocity predictions that identically
            satisfy u(R) = 0 and du/dr(0) = 0.
        """
        r = x[:, 0:1]
        u_max = x[:, 1:2]

        r_norm_sq = (r / self.R) ** 2
        envelope = 1.0 - r_norm_sq

        # Base analytical parabolic profile + boundary-regularized neural correction
        nn_out = self.trunk(x)
        ansatz = u_max * envelope + envelope * r_norm_sq * nn_out
        return ansatz
