"""
Streamlit interactive dashboard for pinn-pipe-flow.

Distinguishes between and fully supports:
1. 1D Hagen-Poiseuille Pipe Flow (exp_001–exp_028):
   (r, u_max) -> u, exact analytical Hagen-Poiseuille solution comparison.
2. 2D Developing Pipe Flow (exp_029+):
   (r/R, x/L, Re) -> (u, v), boundary layer growth, 2D velocity fields, entrance length.

Follows Streamlit best practices and current API standards.
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import yaml

from pinn_pipe.models import MLP
from pinn_pipe.utils import load_run_config
from pinn_pipe.utils.config import Config


# -----------------------------------------------------------------------------
# Streamlit page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PINN Pipe Flow Dashboard",
    page_icon=":material/waves:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Hugging Face Hub: download results/ once and cache the local path
# -----------------------------------------------------------------------------
try:
    HF_REPO = st.secrets.get("HF_REPO", "")
except Exception:
    HF_REPO = ""


@st.cache_resource(show_spinner=False)
def get_results_dir() -> str:
    """Returns the path to the results directory.

    Locally: returns "results/" if it exists.
    On Streamlit Cloud (or whenever HF_REPO is set): downloads model
    artifacts from Hugging Face Hub and returns the local cache path.
    """
    local = "results"
    if os.path.isdir(local):
        return local

    if not HF_REPO:
        st.error(
            "No `results/` directory found and `HF_REPO` is not set.\n\n"
            "Add `HF_REPO = 'your-username/your-repo'` to your Streamlit secrets "
            "or set it as an environment variable."
        )
        st.stop()

    try:
        from huggingface_hub import login, snapshot_download

        try:
            hf_token = st.secrets.get("HF_TOKEN", None)
        except Exception:
            hf_token = None

        if hf_token:
            login(token=hf_token)

        with st.spinner("Downloading model weights from Hugging Face Hub…"):
            path = snapshot_download(repo_id=HF_REPO, repo_type="model")
        return path

    except Exception as e:
        st.error(f"Failed to download models from Hugging Face Hub: {e}")
        st.stop()


RESULTS_DIR = get_results_dir()


# -----------------------------------------------------------------------------
# Helper: render plotly chart with current Streamlit width parameter
# -----------------------------------------------------------------------------
def render_plotly_chart(fig: go.Figure) -> None:
    """Renders a Plotly figure stretching to container width without deprecated parameters."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Legacy 1D Architecture Construction Helpers (exp_001 - exp_028)
# -----------------------------------------------------------------------------
def _build_1d_trunk(depth: int, width: int, activation_name: str) -> nn.Sequential:
    acts = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
        "gelu": nn.GELU(),
        "mish": nn.Mish(),
    }
    act = acts.get(activation_name.lower(), nn.Tanh())
    net = [nn.Linear(2, width)]
    for _ in range(depth):
        net.append(act)
        net.append(nn.Linear(width, width))
    net.append(act)
    net.append(nn.Linear(width, 1))
    return nn.Sequential(*net)


class Legacy1DMLP(nn.Module):
    """Standard 1D MLP for Hagen-Poiseuille flow (exp_001 - exp_027)."""

    def __init__(self, depth: int, width: int, activation_name: str) -> None:
        super().__init__()
        self.net = _build_1d_trunk(depth, width, activation_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LegacyHardBC(nn.Module):
    """Hard boundary condition ansatz model for 1D pipe flow (exp_028)."""

    def __init__(self, depth: int, width: int, activation_name: str, R: float = 1.0) -> None:
        super().__init__()
        self.R = R
        self.trunk = Legacy1DMLP(depth, width, activation_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x[:, 0:1]
        u_max = x[:, 1:2]
        r_norm_sq = (r / self.R) ** 2
        envelope = 1.0 - r_norm_sq
        return u_max * envelope + envelope * r_norm_sq * self.trunk(x)


# -----------------------------------------------------------------------------
# Run scanning & categorization
# -----------------------------------------------------------------------------
def get_available_runs(results_dir: str = RESULTS_DIR) -> Dict[str, Any]:
    """Scans the results directory and separates runs into 2D and 1D categories.

    Returns:
        Dict with keys: 'runs_2d', 'runs_1d', 'all_runs'.
    """
    if not os.path.exists(results_dir):
        return {"runs_2d": [], "runs_1d": [], "all_runs": []}

    latest_runs = {}  # base_name -> full_run_name
    for name in sorted(os.listdir(results_dir)):
        run_path = os.path.join(results_dir, name)
        if not os.path.isdir(run_path):
            continue
        has_model = os.path.exists(os.path.join(run_path, "model.pt"))
        has_config = os.path.exists(os.path.join(run_path, "config.yaml"))
        if not (has_model and has_config):
            continue

        parts = name.rsplit("_", 2)
        base_name = parts[0] if len(parts) == 3 else name
        latest_runs[base_name] = name

    runs_2d = []
    runs_1d = []

    for base_name, full_name in sorted(latest_runs.items()):
        cfg_path = os.path.join(results_dir, full_name, "config.yaml")
        try:
            with open(cfg_path, "r") as f:
                cfg_data = yaml.safe_load(f)
            physics_data = cfg_data.get("physics", {})
            if "L" in physics_data or "Re_range" in physics_data or "Re_min" in physics_data:
                runs_2d.append(full_name)
            else:
                runs_1d.append(full_name)
        except Exception:
            runs_1d.append(full_name)

    all_runs = sorted(runs_2d + runs_1d)
    return {"runs_2d": runs_2d, "runs_1d": runs_1d, "all_runs": all_runs}


@st.cache_resource(show_spinner=False)
def load_trained_model(run_name: str, results_dir: str = RESULTS_DIR):
    """Loads and caches a trained PINN model and its configuration.

    Determines whether the run is 2D developing pipe flow or 1D Hagen-Poiseuille.

    Returns:
        Tuple of (model, config, is_2d).
    """
    run_dir = os.path.join(results_dir, run_name)
    config = load_run_config(run_dir)

    model_path = os.path.join(run_dir, "model.pt")
    state_dict = torch.load(model_path, map_location="cpu")

    # Detect dimensionality from the first linear layer weight matrix
    first_weight_key = next((k for k in state_dict.keys() if k.endswith("0.weight")), None)
    in_features = state_dict[first_weight_key].shape[1] if first_weight_key else 3

    if in_features == 3:
        # 2D developing pipe flow: (r/R, x/L, Re) -> (u, v)
        model = MLP(config.model)
        model.load_state_dict(state_dict)
        model.eval()
        return model, config, True

    # 1D Hagen-Poiseuille flow: (r, u_max) -> u
    is_hard_bc = (config.model.type == "hard_bc_mlp") or getattr(config.model, "hard_bc", False)
    if is_hard_bc:
        model = LegacyHardBC(
            depth=config.model.hidden_layer_depth,
            width=config.model.hidden_layer_width,
            activation_name=config.model.activation,
            R=config.physics.R,
        )
    else:
        model = Legacy1DMLP(
            depth=config.model.hidden_layer_depth,
            width=config.model.hidden_layer_width,
            activation_name=config.model.activation,
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model, config, False


# -----------------------------------------------------------------------------
# Sidebar Selection & Flow Formulation Routing
# -----------------------------------------------------------------------------
run_collections = get_available_runs()
all_available = run_collections["all_runs"]

if not all_available:
    st.warning(
        "No trained models found in the `results/` directory. "
        "Please train a model first using `python scripts/train.py --config <config_path>`."
    )
    st.stop()

st.sidebar.header("Formulation suite")
suite_options = [
    f"2D Developing Flow ({len(run_collections['runs_2d'])} runs)",
    f"1D Hagen-Poiseuille ({len(run_collections['runs_1d'])} runs)",
    f"All Experiments ({len(all_available)} runs)",
]

# Set default suite to 2D Developing if available, else 1D
default_suite_idx = 0 if run_collections["runs_2d"] else 1
selected_suite = st.sidebar.radio(
    "Select experiment suite",
    options=suite_options,
    index=default_suite_idx,
    help="Toggle between the new 2D developing flow experiments and older 1D Hagen-Poiseuille experiments.",
)

if "2D Developing" in selected_suite:
    filtered_runs = run_collections["runs_2d"]
elif "1D Hagen-Poiseuille" in selected_suite:
    filtered_runs = run_collections["runs_1d"]
else:
    filtered_runs = all_available

if not filtered_runs:
    st.sidebar.warning("No runs found for the selected suite.")
    filtered_runs = all_available

# Experiment selection within suite
st.sidebar.header("Model selection")

# Maintain valid session state on suite change
if "selected_run_name" not in st.session_state or st.session_state["selected_run_name"] not in filtered_runs:
    st.session_state["selected_run_name"] = filtered_runs[0]

selected_run = st.sidebar.selectbox(
    "Select experiment run",
    options=filtered_runs,
    index=filtered_runs.index(st.session_state["selected_run_name"]),
    help="Select a trained experiment directory containing model.pt and config.yaml.",
)
st.session_state["selected_run_name"] = selected_run

try:
    load_button = st.sidebar.button(
        "Load model",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    )
except TypeError:
    load_button = st.sidebar.button(
        "Load model",
        type="primary",
        icon=":material/play_arrow:",
        use_container_width=True,
    )

if "loaded_run" not in st.session_state:
    st.session_state["loaded_run"] = selected_run

if load_button:
    st.session_state["loaded_run"] = selected_run

current_run = st.session_state["loaded_run"]

# Safety check: if loaded_run is missing or mismatched
if current_run not in all_available:
    current_run = selected_run
    st.session_state["loaded_run"] = current_run

with st.spinner(f"Loading model '{current_run}'..."):
    try:
        model, config, is_2d = load_trained_model(current_run)
    except Exception as e:
        st.error(f"Failed to load model from '{current_run}': {e}")
        st.stop()

st.sidebar.success(f"Active: `{current_run}`")
st.sidebar.markdown("---")

run_dir = os.path.join(RESULTS_DIR, current_run)


# =============================================================================
# SCENARIO A: 2D Developing Pipe Flow (exp_029+)
# =============================================================================
if is_2d:
    st.title("Physics-informed neural network: developing pipe flow dashboard")
    st.markdown(
        "Interactive exploration of **2D laminar developing pipe flow**. "
        "Evaluates PINN velocity predictions $(u, v)$ from normalized inputs $(r/R, x/L, Re)$ "
        "and illustrates hydrodynamic boundary layer growth from the inlet towards fully developed Hagen-Poiseuille flow."
    )

    R = float(config.physics.R)
    L = float(config.physics.L)
    nu = float(config.physics.nu)
    Re_min = float(config.physics.Re_min)
    Re_max = float(config.physics.Re_max)

    st.sidebar.header("Flow parameters")
    Re_val = st.sidebar.slider(
        "Reynolds number (Re)",
        min_value=float(Re_min),
        max_value=float(Re_max),
        value=float(min(max(300.0, Re_min), Re_max)),
        step=10.0,
        help="Reynolds number based on pipe diameter and average velocity.",
    )

    x_val = st.sidebar.slider(
        "Axial station x [m]",
        min_value=0.0,
        max_value=float(L),
        value=float(L * 0.25),
        step=0.1,
        help="Axial distance from the inlet along the pipe.",
    )

    # Physical quantities
    D = 2.0 * R
    u_in = (Re_val * nu) / D
    entry_length = 0.05 * Re_val * D
    is_fully_developed = x_val >= entry_length

    # Metrics
    st.subheader("Flow conditions and development status")
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Inlet velocity (U_in)", f"{u_in:.4f} m/s")
        m2.metric("Hydrodynamic entry length (Le)", f"{entry_length:.2f} m")
        m3.metric("Station ratio (x / Le)", f"{(x_val / entry_length):.2f}")
        m4.metric(
            "Regime at x",
            "Fully developed" if is_fully_developed else "Developing flow",
            delta=f"x = {x_val:.1f} m (L = {L:.1f} m)",
        )

    with st.expander("Experiment configuration details", expanded=False):
        cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5, cfg_c6 = st.columns(6)
        cfg_c1.metric("Model type", str(config.model.type))
        cfg_c2.metric("Depth", str(config.model.hidden_layer_depth))
        cfg_c3.metric("Width", str(config.model.hidden_layer_width))
        cfg_c4.metric("Activation", str(config.model.activation))
        cfg_c5.metric("Epochs", f"{config.training.epochs:,}")
        cfg_c6.metric("Optimizer", str(config.training.optimizer))

    # Inference for 2D Flow
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    # 2D Heatmap Grid
    n_r_field, n_x_field = 80, 160
    r_grid = np.linspace(0, R, n_r_field)
    x_grid = np.linspace(0, L, n_x_field)
    R_mesh, X_mesh = np.meshgrid(r_grid, x_grid, indexing="ij")

    r_flat = torch.tensor(R_mesh.flatten(), dtype=dtype, device=device).reshape(-1, 1)
    x_flat = torch.tensor(X_mesh.flatten(), dtype=dtype, device=device).reshape(-1, 1)
    re_flat = torch.full_like(r_flat, Re_val)
    inp_2d = torch.cat([r_flat / R, x_flat / L, re_flat], dim=1)

    with torch.no_grad():
        out_field = model(inp_2d)
        u_field = out_field[:, 0:1].cpu().numpy().reshape(n_r_field, n_x_field)
        v_field = out_field[:, 1:2].cpu().numpy().reshape(n_r_field, n_x_field)

    # 1D Radial Slice at selected x_val
    n_r_line = 200
    r_line = np.linspace(0, R, n_r_line)
    r_tensor = torch.tensor(r_line, dtype=dtype, device=device).reshape(-1, 1)
    x_tensor = torch.full_like(r_tensor, x_val)
    re_tensor = torch.full_like(r_tensor, Re_val)
    inp_line = torch.cat([r_tensor / R, x_tensor / L, re_tensor], dim=1)

    with torch.no_grad():
        out_line = model(inp_line)
        u_line = out_line[:, 0].cpu().numpy()
        v_line = out_line[:, 1].cpu().numpy()

    hp_exact = 2.0 * u_in * (1.0 - (r_line / R) ** 2)

    # Centerline velocity along pipe length
    n_x_line = 200
    x_center_line = np.linspace(0, L, n_x_line)
    r_center_tensor = torch.zeros(n_x_line, 1, dtype=dtype, device=device)
    x_center_tensor = torch.tensor(x_center_line, dtype=dtype, device=device).reshape(-1, 1)
    re_center_tensor = torch.full_like(r_center_tensor, Re_val)
    inp_center = torch.cat([r_center_tensor / R, x_center_tensor / L, re_center_tensor], dim=1)

    with torch.no_grad():
        out_center = model(inp_center)
        u_center = out_center[:, 0].cpu().numpy()

    # Visualizations
    st.markdown("---")
    st.subheader("Flow visualizations")

    # Plot 1: 2D Velocity Field Heatmap (mirrored across r=0)
    r_full = np.concatenate([-r_grid[::-1], r_grid[1:]])
    u_full = np.concatenate([u_field[::-1, :], u_field[1:, :]], axis=0)

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=u_full,
            x=x_grid,
            y=r_full,
            colorscale="Viridis",
            colorbar=dict(title="u(r, x) [m/s]"),
            hovertemplate="x: %{x:.2f} m<br>r: %{y:.2f} m<br>u: %{z:.4f} m/s<extra></extra>",
        )
    )
    fig_heatmap.add_vline(
        x=x_val,
        line_width=2.5,
        line_dash="dash",
        line_color="#ffffff",
        annotation_text=f"Station x = {x_val:.1f} m",
        annotation_position="top right",
    )
    if entry_length <= L:
        fig_heatmap.add_vline(
            x=entry_length,
            line_width=2,
            line_dash="dot",
            line_color="#ff7f0e",
            annotation_text=f"Entry length Le = {entry_length:.1f} m",
            annotation_position="bottom right",
        )

    fig_heatmap.update_layout(
        title=dict(
            text=f"Plot 1: 2D Axial Velocity Field u(r, x) at Re = {Re_val:.0f} (Full pipe diameter)",
            font=dict(size=16),
        ),
        xaxis_title="Axial position x [m]",
        yaxis_title="Radial position r [m]",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    render_plotly_chart(fig_heatmap)

    # Row 2: Radial Profiles and Centerline Evolution
    c1, c2 = st.columns(2)

    with c1:
        fig_prof = go.Figure()
        fig_prof.add_trace(
            go.Scatter(
                x=r_line,
                y=u_line,
                mode="lines",
                name=f"PINN u(r) at x={x_val:.1f}m",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig_prof.add_trace(
            go.Scatter(
                x=r_line,
                y=np.full_like(r_line, u_in),
                mode="lines",
                name="Uniform inlet profile (U_in)",
                line=dict(color="#7f7f7f", width=2, dash="dash"),
            )
        )
        fig_prof.add_trace(
            go.Scatter(
                x=r_line,
                y=hp_exact,
                mode="lines",
                name="Fully developed (Hagen-Poiseuille)",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
            )
        )
        fig_prof.update_layout(
            title=dict(
                text=f"Plot 2: Axial Velocity Profile u(r) at x = {x_val:.1f} m (Re = {Re_val:.0f})",
                font=dict(size=14),
            ),
            xaxis_title="Radial position r [m]",
            yaxis_title="Axial velocity u [m/s]",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        render_plotly_chart(fig_prof)

    with c2:
        fig_center = go.Figure()
        fig_center.add_trace(
            go.Scatter(
                x=x_center_line,
                y=u_center,
                mode="lines",
                name="PINN centerline u(0, x)",
                line=dict(color="#2ca02c", width=3),
            )
        )
        fig_center.add_trace(
            go.Scatter(
                x=x_center_line,
                y=np.full_like(x_center_line, 2.0 * u_in),
                mode="lines",
                name="Fully developed target (2 U_in)",
                line=dict(color="#d62728", width=2, dash="dash"),
            )
        )
        fig_center.add_vline(
            x=x_val,
            line_width=2,
            line_dash="dash",
            line_color="#1f77b4",
            annotation_text=f"Current x",
        )
        fig_center.update_layout(
            title=dict(
                text=f"Plot 3: Centerline Axial Velocity Evolution u(0, x) along pipe",
                font=dict(size=14),
            ),
            xaxis_title="Axial position x [m]",
            yaxis_title="Centerline velocity u(0, x) [m/s]",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        render_plotly_chart(fig_center)

    # Row 3: Multi-station profiles & Radial Velocity v(r)
    c3, c4 = st.columns(2)

    with c3:
        fig_multi = go.Figure()
        x_fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        palette = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725", "#ff7f0e"]

        for xf, color in zip(x_fractions, palette):
            x_st = xf * L
            x_st_tensor = torch.full_like(r_tensor, x_st)
            inp_st = torch.cat([r_tensor / R, x_st_tensor / L, re_tensor], dim=1)
            with torch.no_grad():
                u_st = model(inp_st)[:, 0].cpu().numpy()

            fig_multi.add_trace(
                go.Scatter(
                    x=r_line,
                    y=u_st,
                    mode="lines",
                    name=f"x/L = {xf:.2f} ({x_st:.1f}m)",
                    line=dict(color=color, width=2.5),
                )
            )

        fig_multi.update_layout(
            title=dict(
                text=f"Plot 4: Velocity Profiles Across Axial Stations (Re = {Re_val:.0f})",
                font=dict(size=14),
            ),
            xaxis_title="Radial position r [m]",
            yaxis_title="Velocity u(r) [m/s]",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        render_plotly_chart(fig_multi)

    with c4:
        fig_v = go.Figure()
        fig_v.add_trace(
            go.Scatter(
                x=r_line,
                y=v_line,
                mode="lines",
                name="Radial velocity v(r)",
                line=dict(color="#9467bd", width=2.5),
            )
        )
        fig_v.add_hline(y=0.0, line_dash="dash", line_color="gray")
        fig_v.update_layout(
            title=dict(
                text=f"Plot 5: Radial Velocity Component v(r) at x = {x_val:.1f} m",
                font=dict(size=14),
            ),
            xaxis_title="Radial position r [m]",
            yaxis_title="Radial velocity v [m/s]",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        render_plotly_chart(fig_v)


# =============================================================================
# SCENARIO B: 1D Fully Developed Hagen-Poiseuille Flow (exp_001–exp_028)
# =============================================================================
else:
    st.title("Physics-informed neural network: pipe flow dashboard")
    st.markdown(
        "Interactive exploration of laminar Hagen-Poiseuille flow in a circular pipe. "
        "Compare PINN model predictions with the exact analytical parabolic solution: "
        r"$u(r) = u_{\max} \cdot \left(1 - \frac{r^2}{R^2}\right)$."
    )

    pipe_radius_R = float(config.physics.R)

    st.sidebar.header("Flow parameters")
    u_max_val = st.sidebar.slider(
        "Centerline velocity (u_max)",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Maximum velocity at the centerline r = 0.",
    )

    pipe_radius_R = st.sidebar.slider(
        "Pipe radius (R)",
        min_value=0.1,
        max_value=2.0,
        value=float(config.physics.R),
        step=0.05,
        help="Outer radius of the circular pipe.",
    )

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    # Fine grid of 1000 radial points from 0 to R
    r_tensor = torch.linspace(0, pipe_radius_R, 1000, dtype=dtype, device=device).reshape(-1, 1)
    u_max_tensor = torch.full_like(r_tensor, u_max_val)
    x_input = torch.cat([r_tensor, u_max_tensor], dim=1)

    with torch.no_grad():
        u_pred_tensor = model(x_input)
        u_exact_tensor = u_max_tensor * (1.0 - (r_tensor / pipe_radius_R) ** 2)

    abs_error_tensor = torch.abs(u_pred_tensor - u_exact_tensor)
    l2_error = torch.sqrt(torch.mean(abs_error_tensor ** 2)).item()
    max_error = torch.max(abs_error_tensor).item()
    exact_l2 = torch.sqrt(torch.mean(u_exact_tensor ** 2)).item()
    relative_l2_error = (l2_error / exact_l2) if exact_l2 > 0 else 0.0

    r_np = r_tensor.squeeze().cpu().numpy()
    u_pred_np = u_pred_tensor.squeeze().cpu().numpy()
    u_exact_np = u_exact_tensor.squeeze().cpu().numpy()
    abs_error_np = abs_error_tensor.squeeze().cpu().numpy()

    # Metrics Panel
    st.subheader("Performance metrics and model configuration")

    with st.container(border=True):
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("L2 error", f"{l2_error:.4e}")
        metric_col2.metric("Max absolute error", f"{max_error:.4e}")
        metric_col3.metric("Relative L2 error", f"{relative_l2_error:.4e}")

    with st.expander("Experiment configuration details", expanded=True):
        cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5, cfg_c6 = st.columns(6)
        cfg_c1.metric("Model type", str(config.model.type))
        cfg_c2.metric("Depth", str(config.model.hidden_layer_depth))
        cfg_c3.metric("Width", str(config.model.hidden_layer_width))
        cfg_c4.metric("Activation", str(config.model.activation))
        cfg_c5.metric("Epochs", f"{config.training.epochs:,}")
        cfg_c6.metric("Optimizer", str(config.training.optimizer))

    # Flow Visualizations
    st.markdown("---")
    st.subheader("Flow visualizations")

    plot_col1, plot_col2 = st.columns(2)

    # Plot 1: Velocity Profile
    with plot_col1:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=r_np,
                y=u_pred_np,
                mode="lines",
                name="PINN prediction",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=r_np,
                y=u_exact_np,
                mode="lines",
                name="Analytical solution",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            )
        )
        fig1.update_layout(
            title=dict(
                text=f"Plot 1: Velocity profile (u_max = {u_max_val:.1f}, R = {pipe_radius_R:.2f})",
                font=dict(size=15),
            ),
            xaxis_title="Radial position r",
            yaxis_title="Velocity u(r)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        render_plotly_chart(fig1)

    # Plot 2: Pointwise Absolute Error
    with plot_col2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=r_np,
                y=abs_error_np,
                mode="lines",
                name="|u_pred - u_exact|",
                line=dict(color="#d62728", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(214, 39, 40, 0.1)",
            )
        )
        fig2.update_layout(
            title=dict(
                text=f"Plot 2: Pointwise absolute error (u_max = {u_max_val:.1f})",
                font=dict(size=15),
            ),
            xaxis_title="Radial position r",
            yaxis_title="|u_pred - u_exact|",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        render_plotly_chart(fig2)

    # Plot 3: Multi-condition Velocity Profile
    st.markdown("---")
    eval_u_max_values = [0.5, 1.0, 1.5, 2.0]
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig3 = go.Figure()

    for u_val, col in zip(eval_u_max_values, color_palette):
        u_val_tensor = torch.full_like(r_tensor, u_val)
        x_multi = torch.cat([r_tensor, u_val_tensor], dim=1)

        with torch.no_grad():
            u_p = model(x_multi).squeeze().cpu().numpy()
            u_e = (u_val * (1.0 - (r_tensor / pipe_radius_R) ** 2)).squeeze().cpu().numpy()

        fig3.add_trace(
            go.Scatter(
                x=r_np,
                y=u_p,
                mode="lines",
                name="PINN prediction",
                legendgroup=f"u_max_{u_val}",
                legendgrouptitle_text=f"<b>u_max = {u_val}</b>",
                line=dict(color=col, width=2.5),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=r_np,
                y=u_e,
                mode="lines",
                name="Analytical exact",
                legendgroup=f"u_max_{u_val}",
                line=dict(color=col, dash="dash", width=2),
            )
        )

    fig3.update_layout(
        title=dict(
            text=f"Plot 3: Velocity profiles across multiple u_max values [0.5, 1.0, 1.5, 2.0] (R = {pipe_radius_R:.2f})",
            font=dict(size=16),
        ),
        xaxis_title="Radial position r",
        yaxis_title="Velocity u(r)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            title=dict(text="<b>Flow conditions</b>"),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            groupclick="togglegroup",
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    render_plotly_chart(fig3)


# =============================================================================
# Training History (common to both modes if history.csv exists)
# =============================================================================
history_path = os.path.join(run_dir, "history.csv")
if os.path.exists(history_path):
    st.markdown("---")
    st.subheader("Training loss history")

    history_data = []
    with open(history_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history_data.append({k: float(v) for k, v in row.items() if v != ""})

    if history_data:
        epochs = [int(h["epoch"]) for h in history_data]
        loss_keys = [k for k in history_data[0].keys() if k != "epoch"]

        fig_loss = go.Figure()
        for key in loss_keys:
            vals = [h[key] for h in history_data]
            fig_loss.add_trace(go.Scatter(x=epochs, y=vals, mode="lines", name=key))

        fig_loss.update_layout(
            title=dict(text=f"Training Loss Trajectory: {current_run}", font=dict(size=15)),
            xaxis_title="Epoch",
            yaxis_title="Loss (log scale)",
            yaxis_type="log",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        render_plotly_chart(fig_loss)