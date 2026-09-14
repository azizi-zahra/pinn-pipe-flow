"""
Evaluation utilities for pinn-pipe-flow.

Computes error metrics and generates plots comparing the model's
predictions against the analytical Hagen-Poiseuille solution.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_pipe.models import BasePINN
from pinn_pipe.physics import analytical_solution
from pinn_pipe.utils import Config


# u_max values used for evaluation plots
EVAL_U_MAX_VALUES = [0.5, 1.0, 1.5, 2.0]


def compute_metrics(model: BasePINN, config: Config) -> dict:
    """Computes error metrics comparing model predictions to the analytical solution.

    Evaluates the model on a fine grid of r values for several u_max values
    and computes L2 error, maximum error, and relative L2 error.

    Args:
        model: A trained BasePINN instance.
        config: Fully populated Config object for this run.

    Returns:
        Dictionary containing l2_error, max_error, and relative_l2_error,
        each averaged across all evaluated u_max values.
    """
    model.eval()

    device = next(model.parameters()).device
    r = torch.linspace(0, config.physics.R, 1000).reshape(-1, 1).to(device)

    l2_errors = []
    max_errors = []
    relative_l2_errors = []

    with torch.no_grad():
        for u_max_val in EVAL_U_MAX_VALUES:
            u_max = torch.full_like(r, u_max_val)

            x = torch.cat([r, u_max], dim=1)
            u_pred = model(x)
            u_exact = analytical_solution(r, u_max, config.physics.R)

            error = (u_pred - u_exact).abs()

            l2_errors.append(torch.sqrt((error ** 2).mean()).item())
            max_errors.append(error.max().item())
            relative_l2_errors.append(
                (torch.sqrt((error ** 2).mean()) /
                 torch.sqrt((u_exact ** 2).mean())).item()
            )

    return {
        "l2_error": float(np.mean(l2_errors)),
        "max_error": float(np.mean(max_errors)),
        "relative_l2_error": float(np.mean(relative_l2_errors)),
    }


def plot_velocity_profile(
    model: BasePINN,
    config: Config,
    run_dir: str,
) -> None:
    """Plots predicted vs analytical velocity profiles for several u_max values.

    Args:
        model: A trained BasePINN instance.
        config: Fully populated Config object for this run.
        run_dir: Path to the run directory. Plot is saved to run_dir/plots/.
    """
    model.eval()

    device = next(model.parameters()).device
    r = torch.linspace(0, config.physics.R, 1000).reshape(-1, 1).to(device)
    r_np = r.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    with torch.no_grad():
        for u_max_val in EVAL_U_MAX_VALUES:
            u_max = torch.full_like(r, u_max_val)

            x = torch.cat([r, u_max], dim=1)
            u_pred = model(x).detach().cpu().numpy()
            u_exact = analytical_solution(r, u_max, config.physics.R).detach().cpu().numpy()

            color = ax._get_lines.get_next_color()
            ax.plot(r_np, u_exact, linestyle="--", color=color, label=f"Exact u_max={u_max_val}")
            ax.plot(r_np, u_pred, linestyle="-", color=color, label=f"PINN u_max={u_max_val}")

    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    ax.set_title("Velocity Profile: PINN vs Analytical")
    ax.legend()
    ax.grid(True)

    path = os.path.join(run_dir, "plots", "velocity_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved velocity profile plot to: {path}")


def plot_loss_curve(history: list[dict], run_dir: str) -> None:
    """Plots the training loss curves from the epoch history.

    Args:
        history: List of dicts, one per epoch, as produced by the Trainer.
        run_dir: Path to the run directory. Plot is saved to run_dir/plots/.
    """
    epochs = [h["epoch"] for h in history]
    loss_total = [h["loss_total"] for h in history]
    loss_physics = [h["loss_physics"] for h in history]
    loss_bc_wall = [h["loss_bc_wall"] for h in history]
    loss_bc_symmetry = [h["loss_bc_symmetry"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(epochs, loss_total, label="Total")
    ax.semilogy(epochs, loss_physics, label="Physics")
    ax.semilogy(epochs, loss_bc_wall, label="BC Wall")
    ax.semilogy(epochs, loss_bc_symmetry, label="BC Symmetry")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True)

    path = os.path.join(run_dir, "plots", "loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve plot to: {path}")


def plot_error(model: BasePINN, config: Config, run_dir: str) -> None:
    """Plots pointwise absolute error across r for several u_max values.

    Args:
        model: A trained BasePINN instance.
        config: Fully populated Config object for this run.
        run_dir: Path to the run directory. Plot is saved to run_dir/plots/.
    """
    model.eval()

    device = next(model.parameters()).device
    r = torch.linspace(0, config.physics.R, 1000).reshape(-1, 1).to(device)
    r_np = r.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    with torch.no_grad():
        for u_max_val in EVAL_U_MAX_VALUES:
            u_max = torch.full_like(r, u_max_val)

            x = torch.cat([r, u_max], dim=1)
            u_pred = model(x)
            u_exact = analytical_solution(r, u_max, config.physics.R)

            error = (u_pred - u_exact).abs().detach().cpu().numpy()
            ax.plot(r_np, error, label=f"u_max={u_max_val}")

    ax.set_xlabel("r")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Pointwise Absolute Error")
    ax.legend()
    ax.grid(True)

    path = os.path.join(run_dir, "plots", "error_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved error plot to: {path}")