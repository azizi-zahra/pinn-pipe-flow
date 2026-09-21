"""
Evaluation utilities for pinn-pipe-flow.

Generates 2D velocity heatmaps, axial profile evolution plots,
and loss curve plots for developing pipe flow.
"""

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_pipe.models.base import BasePINN
from pinn_pipe.utils.config import Config

EVAL_RE_VALUES = [100.0, 300.0, 500.0]


def compute_metrics(model: BasePINN, config: Config) -> Dict[str, Any]:
    """Computes error metrics comparing model predictions to a reference solution.

    For now, since numerical_reference is not implemented, returns an empty dict.

    Args:
        model: A trained BasePINN instance.
        config: Fully populated Config object for this run.

    Returns:
        Empty dictionary.
    """
    print("Warning: numerical reference not implemented. Metrics not computed.")
    return {}


def plot_velocity_field(
    model: BasePINN,
    config: Config,
    run_dir: str,
    Re_val: float,
) -> None:
    """Creates a 2D heatmap of axial velocity u(r, x) for a single fixed Re.

    Args:
        model: A trained BasePINN instance.
        config: Fully populated Config object for this run.
        run_dir: Path to the run directory.
        Re_val: Fixed Reynolds number to evaluate.
    """
    model.eval()
    device = next(model.parameters()).device

    R = config.physics.R
    L = config.physics.L

    r_grid = np.linspace(0, R, 100)
    x_grid = np.linspace(0, L, 200)
    R_mesh, X_mesh = np.meshgrid(r_grid, x_grid, indexing="ij")

    r_pts = torch.tensor(R_mesh.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
    x_pts = torch.tensor(X_mesh.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
    Re_pts = torch.full((100 * 200, 1), float(Re_val), dtype=torch.float32).to(device)

    inp = torch.cat([r_pts / R, x_pts / L, Re_pts], dim=1)

    with torch.no_grad():
        out = model(inp)
        u = out[:, 0:1].cpu().numpy().reshape(100, 200)

    fig, ax = plt.subplots(figsize=(8, 4))
    c = ax.pcolormesh(x_grid, r_grid, u, shading="auto", cmap="viridis")
    fig.colorbar(c, ax=ax, label="u(r, x) [m/s]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("r [m]")
    ax.set_title(f"Axial velocity u(r,x) at Re={Re_val:.0f}")

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    path = os.path.join(run_dir, "plots", f"velocity_field_Re{Re_val:.0f}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved velocity field plot to: {path}")


def plot_velocity_profiles(
    model: BasePINN,
    config: Config,
    run_dir: str,
    Re_val: float,
) -> None:
    """Plots u(r) at several x locations along with inlet and Hagen-Poiseuille references.

    Args:
        model: A trained BasePINN instance.
        config: Fully populated Config object for this run.
        run_dir: Path to the run directory.
        Re_val: Fixed Reynolds number to evaluate.
    """
    model.eval()
    device = next(model.parameters()).device

    R = config.physics.R
    L = config.physics.L
    u_in = Re_val * config.physics.nu / (2.0 * R)

    r_grid = np.linspace(0, R, 200)
    r_tensor = torch.tensor(r_grid, dtype=torch.float32).reshape(-1, 1).to(device)
    Re_tensor = torch.full_like(r_tensor, float(Re_val))

    x_fractions = [0.1, 0.3, 0.5, 0.7, 1.0]

    fig, ax = plt.subplots(figsize=(8, 5))

    with torch.no_grad():
        for x_frac in x_fractions:
            x_val = x_frac * L
            x_tensor = torch.full_like(r_tensor, float(x_val))
            inp = torch.cat([r_tensor / R, x_tensor / L, Re_tensor], dim=1)
            out = model(inp)
            u_pred = out[:, 0:1].cpu().numpy().flatten()
            ax.plot(r_grid, u_pred, label=f"x/L={x_frac:.1f}")

    ax.plot(r_grid, np.full_like(r_grid, u_in), "--", color="gray", label="Inlet (uniform)")
    hp = 2.0 * u_in * (1.0 - (r_grid / R) ** 2)
    ax.plot(r_grid, hp, "--", color="black", label="Fully developed (HP)")

    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    ax.set_title(f"Velocity Profiles at Re={Re_val:.0f}")
    ax.legend()
    ax.grid(True)

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    path = os.path.join(run_dir, "plots", f"velocity_profiles_Re{Re_val:.0f}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved velocity profiles plot to: {path}")


def plot_loss_curve(history: List[Dict[str, Any]], run_dir: str) -> None:
    """Plots the training loss curves from the epoch history.

    Args:
        history: List of dicts, one per epoch, as produced by the Trainer.
        run_dir: Path to the run directory.
    """
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(epochs, [h["loss_total"] for h in history], label="loss_total")
    ax.semilogy(epochs, [h["loss_momentum"] for h in history], label="loss_momentum")
    ax.semilogy(epochs, [h["loss_continuity"] for h in history], label="loss_continuity")
    ax.semilogy(epochs, [h["loss_bc_wall"] for h in history], label="loss_bc_wall")
    ax.semilogy(epochs, [h["loss_bc_wall_v"] for h in history], label="loss_bc_wall_v")
    ax.semilogy(epochs, [h["loss_bc_symmetry"] for h in history], label="loss_bc_symmetry")
    ax.semilogy(epochs, [h["loss_bc_inlet"] for h in history], label="loss_bc_inlet")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True)

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    path = os.path.join(run_dir, "plots", "loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve plot to: {path}")