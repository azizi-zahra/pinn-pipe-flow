"""
Streamlit interactive dashboard for pinn-pipe-flow.

Allows users to interact with trained Physics-Informed Neural Network (PINN)
models and visualize pipe flow velocity profiles against analytical Hagen-Poiseuille solutions.
Follows Streamlit best practices and current API standards.
"""

import os
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go

from pinn_pipe.models import MLP, HardBCMLP
from pinn_pipe.physics import analytical_solution
from pinn_pipe.utils import load_run_config


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
# Set HF_REPO to your repository, e.g. "your-username/pinn-pipe-flow".
# When running locally, results/ is used directly (no download needed).
# When running on Streamlit Cloud, weights are fetched from HF Hub.
HF_REPO = st.secrets.get("HF_REPO", "")          # set this in Streamlit secrets


@st.cache_resource(show_spinner=False)
def get_results_dir() -> str:
    """Returns the path to the results directory.

    Locally: returns "results/" if it exists.
    On Streamlit Cloud (or whenever HF_REPO is set): downloads the model
    artifacts from Hugging Face Hub and returns the local cache path.
    """
    local = "results"
    if os.path.isdir(local):
        # Running locally - use results/ directly, no download needed.
        return local

    if not HF_REPO:
        st.error(
            "No `results/` directory found and `HF_REPO` is not set.\n\n"
            "Add `HF_REPO = 'your-username/your-repo'` to your Streamlit secrets "
            "or set it as an environment variable."
        )
        st.stop()

    try:
        from huggingface_hub import snapshot_download, login

        # Log in if a token is provided (required for private repos).
        hf_token = st.secrets.get("HF_TOKEN", None)
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
# Run scanning & model loader
# -----------------------------------------------------------------------------
def get_available_runs(results_dir: str = RESULTS_DIR) -> list[str]:
    """Scans the results directory for valid experiment runs.

    A valid run must contain both 'model.pt' and 'config.yaml'.

    Args:
        results_dir: Path to directory containing experiment runs.

    Returns:
        Sorted list of experiment folder names.
    """
    if not os.path.exists(results_dir):
        return []

    runs = []
    for name in sorted(os.listdir(results_dir)):
        run_path = os.path.join(results_dir, name)
        if os.path.isdir(run_path):
            has_model = os.path.exists(os.path.join(run_path, "model.pt"))
            has_config = os.path.exists(os.path.join(run_path, "config.yaml"))
            if has_model and has_config:
                runs.append(name)
    return runs


@st.cache_resource(show_spinner=False)
def load_trained_model(run_name: str, results_dir: str = RESULTS_DIR):
    """Loads and caches a trained PINN model and its configuration.

    Args:
        run_name: Directory name of the experiment run.
        results_dir: Base results directory.

    Returns:
        Tuple of (model, config).
    """
    run_dir = os.path.join(results_dir, run_name)
    config = load_run_config(run_dir)

    # Build model architecture based on config
    if config.model.type == "hard_bc_mlp" or getattr(config.model, "hard_bc", False):
        model = HardBCMLP(config.model, R=config.physics.R)
    elif config.model.type == "mlp":
        model = MLP(config.model)
    else:
        raise ValueError(f"Unsupported model type: {config.model.type}")

    # Load weights
    model_path = os.path.join(run_dir, "model.pt")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


# -----------------------------------------------------------------------------
# UI Layout & Sidebar Controls
# -----------------------------------------------------------------------------
st.title("Physics-informed neural network: pipe flow dashboard")
st.markdown(
    "Interactive exploration of laminar Hagen-Poiseuille flow in a circular pipe. "
    "Compare PINN model predictions with the exact analytical parabolic solution: "
    r"$u(r) = u_{\max} \cdot \left(1 - \frac{r^2}{R^2}\right)$."
)

available_runs = get_available_runs()

if not available_runs:
    st.warning(
        "No trained models found in the `results/` directory. "
        "Please train a model first using `python scripts/train.py --config <config_path>`."
    )
    st.stop()

st.sidebar.header("Model selection")
selected_run = st.sidebar.selectbox(
    "Select experiment run",
    options=available_runs,
    index=0,
    help="Select a trained experiment directory containing model.pt and config.yaml.",
)

# Button to load model (supports modern width='stretch' with fallback)
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

# Store loaded model run in session state
if "loaded_run" not in st.session_state:
    st.session_state["loaded_run"] = selected_run

if load_button:
    st.session_state["loaded_run"] = selected_run

current_run = st.session_state["loaded_run"]

# Load model with spinner and error handling
with st.spinner(f"Loading model '{current_run}'..."):
    try:
        model, config = load_trained_model(current_run)
    except Exception as e:
        st.error(f"Failed to load model from '{current_run}': {e}")
        st.stop()

st.sidebar.success(f"Active: `{current_run}`")
st.sidebar.markdown("---")

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
    value=1.0,
    step=0.05,
    help="Outer radius of the circular pipe.",
)


# -----------------------------------------------------------------------------
# Inference & Metrics Calculation
# -----------------------------------------------------------------------------
dtype = next(model.parameters()).dtype
device = next(model.parameters()).device

# Fine grid of 1000 radial points from 0 to R
r_tensor = torch.linspace(0, pipe_radius_R, 1000, dtype=dtype, device=device).reshape(-1, 1)
u_max_tensor = torch.full_like(r_tensor, u_max_val)
x_input = torch.cat([r_tensor, u_max_tensor], dim=1)

with torch.no_grad():
    u_pred_tensor = model(x_input)
    u_exact_tensor = analytical_solution(r_tensor, u_max_tensor, pipe_radius_R)

# Errors for the current u_max
abs_error_tensor = torch.abs(u_pred_tensor - u_exact_tensor)
l2_error = torch.sqrt(torch.mean(abs_error_tensor ** 2)).item()
max_error = torch.max(abs_error_tensor).item()
exact_l2 = torch.sqrt(torch.mean(u_exact_tensor ** 2)).item()
relative_l2_error = (l2_error / exact_l2) if exact_l2 > 0 else 0.0

r_np = r_tensor.squeeze().cpu().numpy()
u_pred_np = u_pred_tensor.squeeze().cpu().numpy()
u_exact_np = u_exact_tensor.squeeze().cpu().numpy()
abs_error_np = abs_error_tensor.squeeze().cpu().numpy()


# -----------------------------------------------------------------------------
# Metrics Panel
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Plots Section
# -----------------------------------------------------------------------------
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
        u_e = analytical_solution(r_tensor, u_val_tensor, pipe_radius_R).squeeze().cpu().numpy()

    # PINN Prediction (solid)
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
    # Analytical (dashed)
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