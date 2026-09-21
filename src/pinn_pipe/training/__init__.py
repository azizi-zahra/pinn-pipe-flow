"""
Training components for pinn-pipe-flow.

Includes collocation point sampling, physics and boundary loss functions,
and the Trainer class orchestrating model training.
"""

from pinn_pipe.training.losses import (
    bc_inlet_loss,
    bc_symmetry_loss,
    bc_wall_u_loss,
    bc_wall_v_loss,
    physics_loss_continuity,
    physics_loss_momentum,
    total_loss,
)
from pinn_pipe.training.sampler import (
    sample_bc_inlet,
    sample_bc_symmetry,
    sample_bc_wall,
    sample_interior,
)
from pinn_pipe.training.scheduler import build_lr_scheduler
from pinn_pipe.training.trainer import Trainer

__all__ = [
    "Trainer",
    "physics_loss_momentum",
    "physics_loss_continuity",
    "bc_wall_u_loss",
    "bc_wall_v_loss",
    "bc_symmetry_loss",
    "bc_inlet_loss",
    "total_loss",
    "sample_interior",
    "sample_bc_wall",
    "sample_bc_symmetry",
    "sample_bc_inlet",
    "build_lr_scheduler",
]
