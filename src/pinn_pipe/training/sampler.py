"""
Collocation point samplers for pinn-pipe-flow.

Samples interior points for the PDE residual and boundary points
for the wall and symmetry boundary conditions.
"""

import torch


def sample_interior(
    n: int,
    R: float,
    u_max_min: float,
    u_max_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples random interior collocation points inside the pipe.

    Args:
        n: Number of points to sample.
        R: Pipe radius. Points are sampled from (0, R).
        u_max_min: Minimum u_max value.
        u_max_max: Maximum u_max value.

    Returns:
        r: Tensor of shape (N, 1) with requires_grad=True.
        u_max: Tensor of shape (N, 1) with sampled u_max values.
    """
    r = torch.rand(n, 1) * (R - 0.01) + 0.01 # avoid sampling near zero
    r.requires_grad_(True)

    u_max = torch.rand(n, 1) * (u_max_max - u_max_min) + u_max_min

    return r, u_max


def sample_bc(
    n: int,
    R: float,
    u_max_min: float,
    u_max_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples boundary condition points at the wall and pipe center.

    Args:
        n: Number of points to sample per boundary.
        R: Pipe radius. Wall points are set to r = R, symmetry points to r = 0.
        u_max_min: Minimum u_max value.
        u_max_max: Maximum u_max value.

    Returns:
        r_wall: Tensor of shape (N, 1) with r = R, requires_grad=True.
        r_sym: Tensor of shape (N, 1) with r = 0, requires_grad=True.
        u_max_bc: Tensor of shape (N, 1) with sampled u_max values.
    """
    r_wall = torch.ones(n, 1) * R
    r_wall.requires_grad_(True)

    r_sym = torch.zeros(n, 1)
    r_sym.requires_grad_(True)

    u_max_bc = torch.rand(n, 1) * (u_max_max - u_max_min) + u_max_min

    return r_wall, r_sym, u_max_bc