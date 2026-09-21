"""
PDE residual and boundary condition residuals for developing pipe flow.

Contains the governing equation residuals (axial momentum and continuity)
for steady axisymmetric laminar boundary layer flow in cylindrical coordinates,
along with wall, symmetry, and inlet boundary condition residuals.
"""

from typing import Any

import numpy as np
import torch

from pinn_pipe.models.base import BasePINN
from pinn_pipe.utils.config import PhysicsConfig
from pinn_pipe.utils.derivatives import grad, grad2


def _get_u_in(Re: Any, nu: float, R: float) -> Any:
    """Computes the inlet velocity U_in from Reynolds number, viscosity, and radius.

    U_in = Re * nu / (2 * R) = Re * nu / D.

    Args:
        Re: Reynolds number (scalar or Tensor).
        nu: Kinematic viscosity.
        R: Pipe radius.

    Returns:
        Inlet velocity.
    """
    return Re * nu / (2.0 * R)


def pde_residual_momentum(
    model: BasePINN,
    r: torch.Tensor,
    x: torch.Tensor,
    Re: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the axial momentum PDE residual for developing pipe flow.

    Governing equation:
        u * du/dx + v * du/dr - (nu/r) * du/dr - nu * d2u/dr2 = 0

    Args:
        model: A BasePINN instance mapping (r/R, x/L, Re) -> (u, v).
        r: Tensor of shape (N, 1) containing radial coordinates (requires_grad=True).
        x: Tensor of shape (N, 1) containing axial coordinates (requires_grad=True).
        Re: Tensor of shape (N, 1) containing Reynolds numbers.
        config: PhysicsConfig containing R, L, and nu.

    Returns:
        Tensor of shape (N, 1) containing the momentum residual at each point.
    """
    inp = torch.cat([r / config.R, x / config.L, Re], dim=1)
    out = model(inp)
    u = out[:, 0:1]
    v = out[:, 1:2]

    du_dx = grad(u, x)
    du_dr = grad(u, r)
    d2u_dr2 = grad2(u, r)

    nu_val = config.nu
    residual = u * du_dx + v * du_dr - (nu_val / r) * du_dr - nu_val * d2u_dr2
    return residual


def pde_residual_continuity(
    model: BasePINN,
    r: torch.Tensor,
    x: torch.Tensor,
    Re: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the continuity equation residual for axisymmetric flow.

    Governing equation:
        du/dx + (1/r) * d(r*v)/dr = du/dx + v/r + dv/dr = 0

    Args:
        model: A BasePINN instance mapping (r/R, x/L, Re) -> (u, v).
        r: Tensor of shape (N, 1) containing radial coordinates (requires_grad=True).
        x: Tensor of shape (N, 1) containing axial coordinates (requires_grad=True).
        Re: Tensor of shape (N, 1) containing Reynolds numbers.
        config: PhysicsConfig containing R and L.

    Returns:
        Tensor of shape (N, 1) containing the continuity residual at each point.
    """
    inp = torch.cat([r / config.R, x / config.L, Re], dim=1)
    out = model(inp)
    u = out[:, 0:1]
    v = out[:, 1:2]

    du_dx = grad(u, x)
    dv_dr = grad(v, r)

    residual = du_dx + v / r + dv_dr
    return residual


def bc_wall_u(
    model: BasePINN,
    r_wall: torch.Tensor,
    x_bc: torch.Tensor,
    Re_bc: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the no-slip boundary condition residual for u at r = R.

    Args:
        model: A BasePINN instance mapping (r/R, x/L, Re) -> (u, v).
        r_wall: Tensor of shape (N, 1) containing r = R values.
        x_bc: Tensor of shape (N, 1) containing axial positions.
        Re_bc: Tensor of shape (N, 1) containing Reynolds numbers.
        config: PhysicsConfig containing R and L.

    Returns:
        Tensor of shape (N, 1) containing u(R, x) for each sample (target: 0).
    """
    inp = torch.cat([r_wall / config.R, x_bc / config.L, Re_bc], dim=1)
    out = model(inp)
    return out[:, 0:1]


def bc_wall_v(
    model: BasePINN,
    r_wall: torch.Tensor,
    x_bc: torch.Tensor,
    Re_bc: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the no-penetration boundary condition residual for v at r = R.

    Args:
        model: A BasePINN instance mapping (r/R, x/L, Re) -> (u, v).
        r_wall: Tensor of shape (N, 1) containing r = R values.
        x_bc: Tensor of shape (N, 1) containing axial positions.
        Re_bc: Tensor of shape (N, 1) containing Reynolds numbers.
        config: PhysicsConfig containing R and L.

    Returns:
        Tensor of shape (N, 1) containing v(R, x) for each sample (target: 0).
    """
    inp = torch.cat([r_wall / config.R, x_bc / config.L, Re_bc], dim=1)
    out = model(inp)
    return out[:, 1:2]


def bc_symmetry_u(
    model: BasePINN,
    r_sym: torch.Tensor,
    x_bc: torch.Tensor,
    Re_bc: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the symmetry boundary condition residual du/dr at r = 0.

    Args:
        model: A BasePINN instance mapping (r/R, x/L, Re) -> (u, v).
        r_sym: Tensor of shape (N, 1) containing r = 0 values (requires_grad=True).
        x_bc: Tensor of shape (N, 1) containing axial positions.
        Re_bc: Tensor of shape (N, 1) containing Reynolds numbers.
        config: PhysicsConfig containing R and L.

    Returns:
        Tensor of shape (N, 1) containing du/dr at r = 0 for each sample (target: 0).
    """
    r_sym = r_sym.requires_grad_(True)
    inp = torch.cat([r_sym / config.R, x_bc / config.L, Re_bc], dim=1)
    out = model(inp)
    u = out[:, 0:1]
    return grad(u, r_sym)


def bc_inlet_u(
    model: BasePINN,
    r_inlet: torch.Tensor,
    x_inlet: torch.Tensor,
    Re_bc: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Computes the flat inlet profile boundary condition residual at x = 0.

    Args:
        model: A BasePINN instance mapping (r/R, x/L, Re) -> (u, v).
        r_inlet: Tensor of shape (N, 1) containing radial positions in [0, R].
        x_inlet: Tensor of shape (N, 1) containing x = 0 values.
        Re_bc: Tensor of shape (N, 1) containing Reynolds numbers.
        config: PhysicsConfig containing R, L, and nu.

    Returns:
        Tensor of shape (N, 1) containing u(r, 0) - U_in for each sample (target: 0).
    """
    inp = torch.cat([r_inlet / config.R, x_inlet / config.L, Re_bc], dim=1)
    out = model(inp)
    u_pred = out[:, 0:1]
    u_in = _get_u_in(Re_bc, config.nu, config.R)
    return u_pred - u_in


def numerical_reference(
    r_norm: np.ndarray,
    x_norm: np.ndarray,
    Re_val: float,
    config: PhysicsConfig,
) -> np.ndarray:
    """Forward-marching finite difference reference solution for developing pipe flow.

    TODO: Implement numerical reference using forward-marching finite difference
    or solve_ivp on boundary layer ODE in a follow-up.

    Args:
        r_norm: Normalized radial coordinates r/R in [0, 1].
        x_norm: Normalized axial coordinates x/L in [0, 1].
        Re_val: Reynolds number.
        config: PhysicsConfig object.

    Raises:
        NotImplementedError: Always, as numerical reference is to be implemented in a follow-up.
    """
    raise NotImplementedError("Numerical reference not yet implemented -- evaluate visually")