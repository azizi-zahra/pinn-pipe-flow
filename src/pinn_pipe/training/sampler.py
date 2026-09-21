"""
Collocation point samplers for developing pipe flow.

Samples interior points for the PDE residuals and boundary points
for wall, symmetry, and inlet boundary conditions.
"""

from typing import Tuple

import torch

from pinn_pipe.utils.config import PhysicsConfig


def sample_interior(
    n: int,
    config: PhysicsConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples interior collocation points inside the pipe.

    Args:
        n: Number of points to sample.
        config: PhysicsConfig containing domain boundaries.

    Returns:
        r: Tensor of shape (N, 1) in (0.001*R, R), requires_grad=True.
        x: Tensor of shape (N, 1) in (1e-4, L), requires_grad=True.
        Re: Tensor of shape (N, 1) in (Re_min, Re_max), requires_grad=False.
    """
    r_min = 0.001 * config.R
    r = torch.rand(n, 1) * (config.R - r_min) + r_min
    r.requires_grad_(True)

    x_min = 1e-4
    x = torch.rand(n, 1) * (config.L - x_min) + x_min
    x.requires_grad_(True)

    Re = torch.rand(n, 1) * (config.Re_max - config.Re_min) + config.Re_min

    return r, x, Re


def sample_bc_wall(
    n: int,
    config: PhysicsConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples boundary points at the pipe wall r = R.

    Args:
        n: Number of points to sample.
        config: PhysicsConfig containing domain boundaries.

    Returns:
        r_wall: Tensor of shape (N, 1) with r = R, requires_grad=True.
        x_bc: Tensor of shape (N, 1) in (0, L), requires_grad=False.
        Re_bc: Tensor of shape (N, 1) in (Re_min, Re_max), requires_grad=False.
    """
    r_wall = torch.full((n, 1), float(config.R))
    r_wall.requires_grad_(True)

    x_bc = torch.rand(n, 1) * config.L
    Re_bc = torch.rand(n, 1) * (config.Re_max - config.Re_min) + config.Re_min

    return r_wall, x_bc, Re_bc


def sample_bc_symmetry(
    n: int,
    config: PhysicsConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples boundary points at the centerline r = 0.

    Args:
        n: Number of points to sample.
        config: PhysicsConfig containing domain boundaries.

    Returns:
        r_sym: Tensor of shape (N, 1) with r = 0, requires_grad=True.
        x_bc: Tensor of shape (N, 1) in (0, L), requires_grad=False.
        Re_bc: Tensor of shape (N, 1) in (Re_min, Re_max), requires_grad=False.
    """
    r_sym = torch.zeros(n, 1)
    r_sym.requires_grad_(True)

    x_bc = torch.rand(n, 1) * config.L
    Re_bc = torch.rand(n, 1) * (config.Re_max - config.Re_min) + config.Re_min

    return r_sym, x_bc, Re_bc


def sample_bc_inlet(
    n: int,
    config: PhysicsConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples boundary points at the inlet x = 0.

    Args:
        n: Number of points to sample.
        config: PhysicsConfig containing domain boundaries.

    Returns:
        r_inlet: Tensor of shape (N, 1) in (0, R), requires_grad=False.
        x_inlet: Tensor of shape (N, 1) with x = 0, requires_grad=False.
        Re_bc: Tensor of shape (N, 1) in (Re_min, Re_max), requires_grad=False.
    """
    r_inlet = torch.rand(n, 1) * config.R
    x_inlet = torch.zeros(n, 1)
    Re_bc = torch.rand(n, 1) * (config.Re_max - config.Re_min) + config.Re_min

    return r_inlet, x_inlet, Re_bc