"""
Training components for pinn-pipe-flow.

Includes collocation point sampling, physics and boundary loss functions,
and the Trainer class orchestrating model training.
"""

from pinn_pipe.training.losses import (
    bc_symmetry_loss,
    bc_wall_loss,
    physics_loss,
    total_loss,
)
from pinn_pipe.training.sampler import sample_bc, sample_interior
from pinn_pipe.training.scheduler import build_lr_scheduler
from pinn_pipe.training.trainer import Trainer

__all__ = [
    "Trainer",
    "bc_symmetry_loss",
    "bc_wall_loss",
    "build_lr_scheduler",
    "physics_loss",
    "sample_bc",
    "sample_interior",
    "total_loss",
]
