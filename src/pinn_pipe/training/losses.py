"""
Loss functions for developing pipe flow.

Each loss term is a separate function returning a scalar tensor.
The total_loss function combines them with configurable weights
and returns a dictionary of all named loss values for logging.
"""

from typing import Any, Dict

import torch

from pinn_pipe.models.base import BasePINN
from pinn_pipe.physics.pipe_flow import (
    bc_inlet_u,
    bc_symmetry_u,
    bc_wall_u,
    bc_wall_v,
    pde_residual_continuity,
    pde_residual_momentum,
)
from pinn_pipe.utils.config import PhysicsConfig, TrainingConfig


def physics_loss_momentum(
    model: BasePINN,
    r: torch.Tensor,
    x: torch.Tensor,
    Re: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the mean squared axial momentum PDE residual."""
    residual = pde_residual_momentum(model, r, x, Re, config)
    return (residual ** 2).mean()


def physics_loss_continuity(
    model: BasePINN,
    r: torch.Tensor,
    x: torch.Tensor,
    Re: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the mean squared continuity equation residual."""
    residual = pde_residual_continuity(model, r, x, Re, config)
    return (residual ** 2).mean()


def bc_wall_u_loss(
    model: BasePINN,
    r_wall: torch.Tensor,
    x_bc: torch.Tensor,
    Re_bc: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the mean squared no-slip BC residual for u at r = R."""
    return (bc_wall_u(model, r_wall, x_bc, Re_bc, config) ** 2).mean()


def bc_wall_v_loss(
    model: BasePINN,
    r_wall: torch.Tensor,
    x_bc: torch.Tensor,
    Re_bc: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the mean squared no-penetration BC residual for v at r = R."""
    return (bc_wall_v(model, r_wall, x_bc, Re_bc, config) ** 2).mean()


def bc_symmetry_loss(
    model: BasePINN,
    r_sym: torch.Tensor,
    x_bc: torch.Tensor,
    Re_bc: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the mean squared symmetry BC residual du/dr at r = 0."""
    return (bc_symmetry_u(model, r_sym, x_bc, Re_bc, config) ** 2).mean()


def bc_inlet_loss(
    model: BasePINN,
    r_inlet: torch.Tensor,
    x_inlet: torch.Tensor,
    Re_bc: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the mean squared inlet profile BC residual at x = 0."""
    return (bc_inlet_u(model, r_inlet, x_inlet, Re_bc, config) ** 2).mean()


def total_loss(
    model: BasePINN,
    r: torch.Tensor,
    x: torch.Tensor,
    Re: torch.Tensor,
    r_wall: torch.Tensor,
    x_bc_wall: torch.Tensor,
    Re_bc_wall: torch.Tensor,
    r_sym: torch.Tensor,
    x_bc_sym: torch.Tensor,
    Re_bc_sym: torch.Tensor,
    r_inlet: torch.Tensor,
    x_inlet: torch.Tensor,
    Re_bc_inlet: torch.Tensor,
    physics_config: PhysicsConfig,
    training_config: TrainingConfig,
) -> Dict[str, Any]:
    """Computes the weighted total loss and all individual loss terms.

    Args:
        model: A BasePINN instance.
        r, x, Re: Interior collocation points.
        r_wall, x_bc_wall, Re_bc_wall: Wall boundary points.
        r_sym, x_bc_sym, Re_bc_sym: Symmetry boundary points.
        r_inlet, x_inlet, Re_bc_inlet: Inlet boundary points.
        physics_config: PhysicsConfig containing domain and fluid parameters.
        training_config: TrainingConfig containing loss weights.

    Returns:
        Dictionary with keys:
            loss_total_tensor -- tensor for backward()
            loss_total -- float
            loss_momentum -- float
            loss_continuity -- float
            loss_bc_wall -- float
            loss_bc_wall_v -- float
            loss_bc_symmetry -- float
            loss_bc_inlet -- float
    """
    loss_mom = physics_loss_momentum(model, r, x, Re, physics_config)
    loss_cont = physics_loss_continuity(model, r, x, Re, physics_config)
    loss_wall_u = bc_wall_u_loss(model, r_wall, x_bc_wall, Re_bc_wall, physics_config)
    loss_wall_v = bc_wall_v_loss(model, r_wall, x_bc_wall, Re_bc_wall, physics_config)
    loss_sym = bc_symmetry_loss(model, r_sym, x_bc_sym, Re_bc_sym, physics_config)
    loss_inlet = bc_inlet_loss(model, r_inlet, x_inlet, Re_bc_inlet, physics_config)

    loss_total_tensor = (
        training_config.loss_weight_physics * loss_mom
        + training_config.loss_weight_continuity * loss_cont
        + training_config.loss_weight_bc_wall * loss_wall_u
        + training_config.loss_weight_bc_wall_v * loss_wall_v
        + training_config.loss_weight_bc_symmetry * loss_sym
        + training_config.loss_weight_bc_inlet * loss_inlet
    )

    return {
        "loss_total_tensor": loss_total_tensor,
        "loss_total": float(loss_total_tensor.item()),
        "loss_momentum": float(loss_mom.item()),
        "loss_continuity": float(loss_cont.item()),
        "loss_bc_wall": float(loss_wall_u.item()),
        "loss_bc_wall_v": float(loss_wall_v.item()),
        "loss_bc_symmetry": float(loss_sym.item()),
        "loss_bc_inlet": float(loss_inlet.item()),
    }