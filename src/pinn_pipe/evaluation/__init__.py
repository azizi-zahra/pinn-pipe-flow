"""
Evaluation and visualization utilities for pinn-pipe-flow.

Provides functions to compute accuracy metrics against analytical solutions
and generate plots for velocity profiles, error distributions, and loss curves.
"""

from pinn_pipe.evaluation.evaluator import (
    compute_metrics,
    plot_error,
    plot_loss_curve,
    plot_velocity_profile,
)

__all__ = [
    "compute_metrics",
    "plot_error",
    "plot_loss_curve",
    "plot_velocity_profile",
]
