"""
PDE residual and boundary condition residuals for Hagen-Poiseuille pipe flow.

Contains the governing equation residual, boundary condition residuals,
and the analytical solution used for evaluation.
"""

import torch

from pinn_pipe.models.base import BasePINN
from pinn_pipe.utils.config import PhysicsConfig
from pinn_pipe.utils.derivatives import grad, grad2


def pde_residual(model: BasePINN, r: torch.Tensor, u_max: torch.Tensor, config: PhysicsConfig) -> torch.Tensor:
    """Computes the PDE residual of the pipe flow governing equation.

    Evaluates the left hand side of the simplified Navier-Stokes equation
    at the given collocation points. A perfect model returns zero everywhere.

    Args:
        model: A BasePINN instance mapping (r, u_max) -> u.
        r: Tensor of shape (N, 1) containing interior collocation points.
        u_max: Tensor of shape (N, 1) containing sampled u_max values.
        config: PhysicsConfig containing R and mu.

    Returns:
        Tensor of shape (N, 1) containing the residual at each point.
    """
    x = torch.cat([r, u_max], dim=1)
    u = model(x)
    du_dr = grad(u, r)
    d2u_dr2 = grad2(u, r)

    dp_dz = -4.0 * config.mu * u_max / config.R ** 2
    residual = config.mu * (d2u_dr2 + (1 / r) * du_dr) - dp_dz

    return residual


def bc_wall(model: BasePINN, r_bc: torch.Tensor, u_max: torch.Tensor) -> torch.Tensor:
    """Computes the no-slip boundary condition residual at the pipe wall.

    The velocity must be zero at r = R. A perfect model returns zero.

    Args:
        model: A BasePINN instance mapping (r, u_max) -> u.
        r_bc: Tensor of shape (N, 1) containing r = R values.
        u_max: Tensor of shape (N, 1) containing sampled u_max values.

    Returns:
        Tensor of shape (N, 1) containing u(R) for each sample.
    """
    x = torch.cat([r_bc, u_max], dim=1)
    return model(x)


def bc_symmetry(model: BasePINN, r_bc: torch.Tensor, u_max: torch.Tensor) -> torch.Tensor:
    """Computes the symmetry boundary condition residual at the pipe center.

    The velocity gradient must be zero at r = 0. A perfect model returns zero.

    Args:
        model: A BasePINN instance mapping (r, u_max) -> u.
        r_bc: Tensor of shape (N, 1) containing r = 0 values.
        u_max: Tensor of shape (N, 1) containing sampled u_max values.

    Returns:
        Tensor of shape (N, 1) containing du/dr at r = 0 for each sample.
    """
    r_bc = r_bc.requires_grad_(True)
    x = torch.cat([r_bc, u_max], dim=1)
    u = model(x)
    return grad(u, r_bc)


def analytical_solution(r: torch.Tensor, u_max: torch.Tensor, R: float) -> torch.Tensor:
    """Computes the exact Hagen-Poiseuille velocity profile.

    Used during evaluation to compare against the model's predictions.

    Args:
        r: Tensor of shape (N, 1) containing radial positions.
        u_max: Tensor of shape (N, 1) containing maximum velocity values.
        R: Pipe radius.

    Returns:
        Tensor of shape (N, 1) containing the exact velocity at each point.
    """
    return u_max * (1 - r ** 2 / R ** 2)