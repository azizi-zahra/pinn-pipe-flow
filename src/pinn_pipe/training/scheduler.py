"""
Learning rate scheduler utilities for pinn-pipe-flow.

Provides factory functions to configure and instantiate PyTorch learning rate
schedulers such as CosineAnnealingLR, StepLR, MultiStepLR, and ReduceLROnPlateau.
"""

from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Optional[str],
    total_epochs: int,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Instantiates a PyTorch learning rate scheduler according to configuration.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        scheduler_type: Name of the scheduler ("cosine", "step", "multistep",
            "plateau", "exponential") or None / "" for no scheduler.
        total_epochs: Total number of training epochs (used as default T_max / milestones).
        params: Optional dictionary of scheduler-specific hyperparameters.

    Returns:
        An instantiated PyTorch lr_scheduler object, or None if no scheduler is configured.

    Raises:
        ValueError: If scheduler_type is not one of the supported types.
    """
    if not scheduler_type:
        return None

    name = scheduler_type.lower().strip()
    p = params or {}

    if name in ("cosine", "cosine_annealing"):
        t_max = int(p.get("T_max", total_epochs))
        eta_min = float(p.get("eta_min", 1e-6))
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    elif name == "step":
        step_size = int(p.get("step_size", max(1, total_epochs // 4)))
        gamma = float(p.get("gamma", 0.5))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif name in ("multistep", "multi_step"):
        milestones = p.get("milestones", [int(total_epochs * 0.5), int(total_epochs * 0.75)])
        gamma = float(p.get("gamma", 0.5))
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif name in ("plateau", "reduce_on_plateau"):
        mode = str(p.get("mode", "min"))
        factor = float(p.get("factor", 0.5))
        patience = int(p.get("patience", 1000))
        min_lr = float(p.get("min_lr", 1e-6))
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)

    elif name == "exponential":
        gamma = float(p.get("gamma", 0.9999))
        return ExponentialLR(optimizer, gamma=gamma)

    else:
        supported = ["cosine", "step", "multistep", "plateau", "exponential"]
        raise ValueError(
            f"Unsupported lr_scheduler: '{scheduler_type}'. Choose from {supported}."
        )
