"""
Loss functions for pinn-pipe-flow.

Each loss term is a separate function returning a scalar tensor.
The total_loss function combines them with configurable weights
and returns a dictionary of all named loss values for logging.
"""

import torch

from pinn_pipe.models import BasePINN
from pinn_pipe.physics import bc_symmetry, bc_wall, pde_residual
from pinn_pipe.utils import PhysicsConfig, TrainingConfig


def physics_loss(
    model: BasePINN,
    r: torch.Tensor,
    u_max: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the mean squared PDE residual over interior collocation points.

    Args:
        model: A BasePINN instance mapping (r, u_max) -> u.
        r: Tensor of shape (N, 1) containing interior collocation points.
        u_max: Tensor of shape (N, 1) containing sampled u_max values.
        config: PhysicsConfig containing R and mu.

    Returns:
        Scalar tensor containing the mean squared PDE residual.
    """
    return (pde_residual(model, r, u_max, config) ** 2).mean()


def bc_wall_loss(
    model: BasePINN,
    r_wall: torch.Tensor,
    u_max: torch.Tensor,
) -> torch.Tensor:
    """Computes the mean squared no-slip BC residual at the pipe wall.

    Args:
        model: A BasePINN instance mapping (r, u_max) -> u.
        r_wall: Tensor of shape (N, 1) containing r = R values.
        u_max: Tensor of shape (N, 1) containing sampled u_max values.

    Returns:
        Scalar tensor containing the mean squared wall BC residual.
    """
    return (bc_wall(model, r_wall, u_max) ** 2).mean()


def bc_symmetry_loss(
    model: BasePINN,
    r_sym: torch.Tensor,
    u_max: torch.Tensor,
) -> torch.Tensor:
    """Computes the mean squared symmetry BC residual at the pipe center.

    Args:
        model: A BasePINN instance mapping (r, u_max) -> u.
        r_sym: Tensor of shape (N, 1) containing r = 0 values.
        u_max: Tensor of shape (N, 1) containing sampled u_max values.

    Returns:
        Scalar tensor containing the mean squared symmetry BC residual.
    """
    return (bc_symmetry(model, r_sym, u_max) ** 2).mean()


def total_loss(
    model: BasePINN,
    r: torch.Tensor,
    u_max: torch.Tensor,
    r_wall: torch.Tensor,
    r_sym: torch.Tensor,
    u_max_bc: torch.Tensor,
    physics_config: PhysicsConfig,
    training_config: TrainingConfig,
) -> dict:
    """Computes the weighted total loss and all individual loss terms.

    Args:
        model: A BasePINN instance mapping (r, u_max) -> u.
        r: Tensor of shape (N, 1) containing interior collocation points.
        u_max: Tensor of shape (N, 1) containing sampled u_max values.
        r_wall: Tensor of shape (N, 1) containing r = R values.
        r_sym: Tensor of shape (N, 1) containing r = 0 values.
        u_max_bc: Tensor of shape (N, 1) containing u_max values for BC points.
        physics_config: PhysicsConfig containing R and mu.
        training_config: TrainingConfig containing loss weights.

    Returns:
        Dictionary with keys: loss_total_tensor, loss_total, loss_physics,
        loss_bc_wall, loss_bc_symmetry.
    """
    loss_pde = physics_loss(model, r, u_max, physics_config)
    loss_bc_wall = bc_wall_loss(model, r_wall, u_max_bc)
    loss_bc_sym = bc_symmetry_loss(model, r_sym, u_max_bc)

    loss_total = (
        training_config.loss_weight_physics * loss_pde
        + training_config.loss_weight_bc_wall * loss_bc_wall
        + training_config.loss_weight_bc_symmetry * loss_bc_sym
    )

    return {
        "loss_total_tensor": loss_total,        # for backward()
        "loss_total": loss_total.item(),
        "loss_physics": loss_pde.item(),
        "loss_bc_wall": loss_bc_wall.item(),
        "loss_bc_symmetry": loss_bc_sym.item(),
    }