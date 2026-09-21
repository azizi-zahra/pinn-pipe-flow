"""
Evaluation and visualization utilities for pinn-pipe-flow.

Provides functions to compute accuracy metrics against reference solutions
and generate plots for 2D velocity fields, velocity profiles, and loss curves.
"""

from pinn_pipe.evaluation.evaluator import (
    EVAL_RE_VALUES,
    compute_metrics,
    plot_loss_curve,
    plot_velocity_field,
    plot_velocity_profiles,
)

__all__ = [
    "EVAL_RE_VALUES",
    "compute_metrics",
    "plot_loss_curve",
    "plot_velocity_field",
    "plot_velocity_profiles",
]
