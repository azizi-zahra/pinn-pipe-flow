# Codebase Overview: `pinn-pipe-flow`

> *Documentation created with Gemini 3.8 Flash.*

---

## 1. Overview & Purpose

This document provides a technical overview of the current codebase architecture for **reviewers and contributors**. It catalogs all modules, configurations, and core implementations completed to date for the `pinn-pipe-flow` Physics-Informed Neural Network (PINN) project, modeling steady laminar flow in a circular pipe (Hagen-Poiseuille).

---

## 2. Completed Files Overview

| Category | File Path | Status | Primary Responsibility |
| :--- | :--- | :---: | :--- |
| **Project & Build** | [`pyproject.toml`](../pyproject.toml) | Completed | Package metadata, dependencies (`numpy`, `matplotlib`, `pyyaml`, `scipy`), and testing configuration. |
| | [`.gitignore`](../.gitignore) | Completed | Excludes Python cache, Jupyter checkpoints, IDE files, and local run outputs. |
| **Configuration** | [`configs/base.yaml`](../configs/base.yaml) | Completed | Global default parameters for physics, network architecture, sampling, optimizer, and seeds. |
| | [`configs/experiments/exp_001_mlp_baseline.yaml`](../configs/experiments/exp_001_mlp_baseline.yaml) | Completed | Baseline experiment configuration inheriting from `base.yaml`. |
| **Models** | [`src/pinn_pipe/models/base.py`](../src/pinn_pipe/models/base.py) | Completed | Abstract base class (`BasePINN`) establishing interface contract `forward(x)`. |
| | [`src/pinn_pipe/models/mlp.py`](../src/pinn_pipe/models/mlp.py) | Completed | Fully-connected neural network (`MLP`) with configurable width, depth, and activation functions. |
| **Physics** | [`src/pinn_pipe/physics/pipe_flow.py`](../src/pinn_pipe/physics/pipe_flow.py) | Completed | Governing PDE residual, wall no-slip BC, symmetry BC, and exact analytical solution. |
| **Training** | [`src/pinn_pipe/training/sampler.py`](../src/pinn_pipe/training/sampler.py) | Completed | Uniform sampling for interior collocation points and boundary points with autograd tracking. |
| | [`src/pinn_pipe/training/losses.py`](../src/pinn_pipe/training/losses.py) | Completed | Mean squared residual loss terms (PDE, wall BC, symmetry BC) and weighted composite loss. |
| **Utilities** | [`src/pinn_pipe/utils/config.py`](../src/pinn_pipe/utils/config.py) | Completed | Strongly typed config dataclasses, hierarchical YAML loader, and sanity validation. |
| | [`src/pinn_pipe/utils/derivatives.py`](../src/pinn_pipe/utils/derivatives.py) | Completed | Autograd derivative wrappers for 1st (`grad`) and 2nd (`grad2`) order partial derivatives. |
| | [`src/pinn_pipe/utils/io.py`](../src/pinn_pipe/utils/io.py) | Completed | Run directory management, metric serialization (`json`), history tracking (`csv`), and model checkpoints. |
| | [`src/pinn_pipe/utils/reproducibility.py`](../src/pinn_pipe/utils/reproducibility.py) | Completed | Seed initialization across Python `random`, NumPy, and PyTorch (including cuDNN deterministic mode). |

*(Note: Incomplete/pending files in the repository: [`src/pinn_pipe/training/trainer.py`](../src/pinn_pipe/training/trainer.py) is currently empty; `src/pinn_pipe/evaluation/` is pending implementation; package `__init__.py` files are empty placeholder markers).*

---

## 3. Detailed Component Documentation

### 3.1 Project & Configuration

- [`pyproject.toml`](../pyproject.toml)
  - Configures packaging via `setuptools >= 68`.
  - Defines core dependencies: `numpy`, `matplotlib`, `pyyaml`, `scipy`.
  - Defines optional `dev` dependencies: `pytest`, `pytest-cov`.
  - Targets Python `>=3.10` and sets package discovery root to `src/`.

- [`configs/base.yaml`](../configs/base.yaml)
  - **Physics**: Pipe radius $R = 1.0$, dynamic viscosity $\mu = 1.0$, maximum velocity range $u_{\max} \in [0.5, 2.0]$.
  - **Model**: Architecture type `mlp`, depth `3`, width `32`, activation `tanh`.
  - **Sampling**: `1000` interior collocation points per iteration.
  - **Training**: `4000` epochs, learning rate `1e-3`, optimizer `adam`, loss weights (physics: 1.0, wall BC: 1.0, symmetry BC: 1.0).
  - **Run**: Seed `42`, precision `float64`.

- [`configs/experiments/exp_001_mlp_baseline.yaml`](../configs/experiments/exp_001_mlp_baseline.yaml)
  - Baseline experiment declaration with metadata `name: exp_001_mlp_baseline`.
  - Inherits all baseline defaults from `base.yaml`.

---

### 3.2 Neural Network Architecture (`pinn_pipe.models`)

- [`src/pinn_pipe/models/base.py`](../src/pinn_pipe/models/base.py)
  - Class: `BasePINN(ABC)`
  - Interface contract requiring `forward(x: torch.Tensor) -> torch.Tensor`, taking coordinate batch inputs of shape $(N, 2)$ where each row is $[r, u_{\max}]$ and returning predicted scalar velocity $u$ of shape $(N, 1)$.

- [`src/pinn_pipe/models/mlp.py`](../src/pinn_pipe/models/mlp.py)
  - Class: `MLP(BasePINN, nn.Module)`
  - Fully-connected architecture: `Linear(2, width) -> [Activation -> Linear(width, width)] * depth -> Activation -> Linear(width, 1)`.
  - Supports configurable activations: `tanh`, `relu`, `sigmoid`.
  - Implements `__repr__` for human-readable architecture inspection.

---

### 3.3 Physics & Governing Equations (`pinn_pipe.physics`)

- [`src/pinn_pipe/physics/pipe_flow.py`](../src/pinn_pipe/physics/pipe_flow.py)
  - `pde_residual(model, r, u_max, config)`: Evaluates the steady laminar Navier-Stokes momentum residual in cylindrical coordinates:
    $$\text{Residual} = \mu \left( \frac{\partial^2 u}{\partial r^2} + \frac{1}{r} \frac{\partial u}{\partial r} \right) - \frac{\partial p}{\partial z}$$
    where $\frac{\partial p}{\partial z} = -\frac{2\mu u_{\max}}{R^2}$.
  - `bc_wall(model, r_bc, u_max)`: Computes no-slip wall condition residual $u(R) = 0$.
  - `bc_symmetry(model, r_bc, u_max)`: Computes centerline symmetry condition residual $\left. \frac{\partial u}{\partial r} \right|_{r=0} = 0$.
  - `analytical_solution(r, u_max, R)`: Exact Hagen-Poiseuille parabolic profile:
    $$u_{\text{exact}}(r) = u_{\max} \left(1 - \frac{r^2}{R^2}\right)$$

---

### 3.4 Sampling & Losses (`pinn_pipe.training`)

- [`src/pinn_pipe/training/sampler.py`](../src/pinn_pipe/training/sampler.py)
  - `sample_interior(n, R, u_max_min, u_max_max)`: Uniformly samples $n$ interior points $r \in (0, R)$ with `requires_grad=True` and paired $u_{\max}$ values.
  - `sample_bc(n, R, u_max_min, u_max_max)`: Generates wall points ($r=R$), centerline symmetry points ($r=0$), and corresponding $u_{\max}$ inputs.

- [`src/pinn_pipe/training/losses.py`](../src/pinn_pipe/training/losses.py)
  - `physics_loss(...)`: Mean squared error (MSE) of `pde_residual`.
  - `bc_wall_loss(...)`: MSE of predicted velocity at the pipe wall.
  - `bc_symmetry_loss(...)`: MSE of first derivative $\frac{\partial u}{\partial r}$ at $r=0$.
  - `total_loss(...)`: Computes weighted sum:
    $$\mathcal{L}_{\text{total}} = w_{\text{pde}} \mathcal{L}_{\text{pde}} + w_{\text{wall}} \mathcal{L}_{\text{wall}} + w_{\text{sym}} \mathcal{L}_{\text{sym}}$$
    Returns PyTorch scalar tensor for gradient backpropagation and numerical floats for logging.

---

### 3.5 Utilities (`pinn_pipe.utils`)

- [`src/pinn_pipe/utils/config.py`](../src/pinn_pipe/utils/config.py)
  - Dataclasses: `PhysicsConfig`, `ModelConfig`, `SamplingConfig`, `TrainingConfig`, `RunConfig`, and root `Config`.
  - `load_config(base_path, experiment_path)`: Performs deep merge of experiment-level overrides into base settings.
  - `validate_config(config)`: Enforces positive geometric/fluid parameters, valid parameter bounds ($u_{\max,\min} < u_{\max,\max}$), and positive epochs/learning rates.

- [`src/pinn_pipe/utils/derivatives.py`](../src/pinn_pipe/utils/derivatives.py)
  - `grad(output, input)`: First-order derivative via `torch.autograd.grad` with `create_graph=True`.
  - `grad2(output, input)`: Second-order derivative computed by differentiating the first-order result.

- [`src/pinn_pipe/utils/io.py`](../src/pinn_pipe/utils/io.py)
  - `create_run_dir(experiment_name, results_dir)`: Creates timestamped experiment run directory (e.g., `results/exp_001_mlp_baseline_YYYYMMDD_HHMMSS/plots/`).
  - `save_config(config, run_dir)`: Exports run configuration to `config.yaml`.
  - `save_metrics(metrics, run_dir)`: Serializes evaluation metrics dictionary to `metrics.json`.
  - `save_history(history, run_dir)`: Writes epoch training log to `history.csv`.
  - `save_model(model, run_dir)` / `load_model(model, run_dir)`: Saves and loads model state dictionary (`model.pt`).

- [`src/pinn_pipe/utils/reproducibility.py`](../src/pinn_pipe/utils/reproducibility.py)
  - `set_seed(seed)`: Synchronously seeds Python standard `random`, `numpy`, PyTorch CPU/GPU generators, and sets `torch.backends.cudnn.deterministic = True`.
