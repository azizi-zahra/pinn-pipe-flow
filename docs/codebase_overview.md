# Codebase Overview: `pinn-pipe-flow`

> *Technical reference and architectural catalog for reviewers and developers.*

---

## 1. Overview & Purpose

This document provides a comprehensive technical overview of the codebase architecture for the `pinn-pipe-flow` project. The repository implements a Physics-Informed Neural Network (PINN) designed to solve the steady laminar flow of an incompressible Newtonian fluid through a straight circular pipe (governed by the Hagen-Poiseuille equations) using PyTorch.

The framework supports:
- Automated exact derivative computation via PyTorch autograd.
- Parametric modeling over a continuum of centerline velocities ($u_{\max}$).
- Hierarchical configuration inheritance with YAML deep merging and strict validation.
- End-to-end training, checkpointing, artifact serialization, and automated cross-experiment comparison.
- Package-level clean public interfaces across all subpackages.
- Full unit test coverage and technical documentation.

---

## 2. Completed Files Overview

| Category | File Path | Status | Primary Responsibility |
| :--- | :--- | :---: | :--- |
| **Project & Build** | [`README.md`](../README.md) | Completed | Project introduction, installation instructions, usage guide, and architecture diagram. |
| | [`CHANGELOG.md`](../CHANGELOG.md) | Completed | Project changelog tracking bug fixes, physical corrections, and experiment iterations. |
| | [`pyproject.toml`](../pyproject.toml) | Completed | Package metadata, dependencies (`numpy`, `matplotlib`, `pyyaml`, `scipy`), and pytest configuration. |
| | [`.gitignore`](../.gitignore) | Completed | Excludes Python bytecode, cache, virtual environments, local artifacts, and test outputs. |
| **Configuration** | [`configs/base.yaml`](../configs/base.yaml) | Completed | Global default parameters for physics, network architecture, sampling, optimizer, and seeds. |
| | [`configs/experiments/exp_001_mlp_baseline.yaml`](../configs/experiments/exp_001_mlp_baseline.yaml) | Completed | Baseline experiment configuration inheriting all defaults from `base.yaml`. |
| | [`configs/experiments/exp_002_lower_physics_weight.yaml`](../configs/experiments/exp_002_lower_physics_weight.yaml) | Completed | Experiment evaluating reduced physics loss weight (0.01), 8000 epochs, and lower learning rate (1e-4). |
| | [`configs/experiments/exp_003_double_epochs.yaml`](../configs/experiments/exp_003_double_epochs.yaml) | Completed | Experiment evaluating 2x baseline training epochs (8,000 epochs) to assess convergence. |
| | [`configs/experiments/exp_004_quadruple_epochs.yaml`](../configs/experiments/exp_004_quadruple_epochs.yaml) | Completed | Experiment evaluating 4x baseline training epochs (16,000 epochs) to evaluate error scaling. |
| | [`configs/experiments/exp_005_8x_epochs.yaml`](../configs/experiments/exp_005_8x_epochs.yaml) | Completed | Experiment evaluating 8x baseline training epochs (32,000 epochs) to find the convergence plateau. |
| | [`configs/experiments/exp_006_16x_epochs.yaml`](../configs/experiments/exp_006_16x_epochs.yaml) | Completed | Experiment evaluating 16x baseline training epochs (64,000 epochs) for extended asymptotic training. |
| | [`configs/experiments/exp_007_wider_network.yaml`](../configs/experiments/exp_007_wider_network.yaml) | Completed | Architecture test evaluating wider network (width=64, depth=3) at 64k epochs. |
| | [`configs/experiments/exp_008_widest_network.yaml`](../configs/experiments/exp_008_widest_network.yaml) | Completed | Architecture test evaluating widest network (width=128, depth=3) at 64k epochs. |
| | [`configs/experiments/exp_009_deeper_network.yaml`](../configs/experiments/exp_009_deeper_network.yaml) | Completed | Architecture test evaluating deeper network (width=32, depth=6) at 64k epochs (best performing). |
| | [`configs/experiments/exp_010_deepest_network.yaml`](../configs/experiments/exp_010_deepest_network.yaml) | Completed | Architecture test evaluating deepest network (width=32, depth=9) at 64k epochs. |
| | [`configs/experiments/exp_011_sgd_optimizer.yaml`](../configs/experiments/exp_011_sgd_optimizer.yaml) | Completed | Optimizer test evaluating SGD vs Adam with depth=6 network at 64k epochs. |
| | [`configs/experiments/exp_012_lbfgs_optimizer.yaml`](../configs/experiments/exp_012_lbfgs_optimizer.yaml) | Completed | Optimizer test evaluating L-BFGS vs Adam with depth=6 network at 64k epochs. |
| | [`configs/experiments/exp_013_sigmoid_activation.yaml`](../configs/experiments/exp_013_sigmoid_activation.yaml) | Completed | Activation test evaluating Sigmoid vs Tanh with depth=6 network at 64k epochs. |
| | [`configs/experiments/exp_014_more_collocation_points.yaml`](../configs/experiments/exp_014_more_collocation_points.yaml) | Completed | Sampling test evaluating 5,000 interior collocation points with depth=6 network at 64k epochs. |
| | [`configs/experiments/exp_015_fewer_collocation_points.yaml`](../configs/experiments/exp_015_fewer_collocation_points.yaml) | Completed | Sampling test evaluating 200 interior collocation points with depth=6 network at 64k epochs. |
| | [`configs/experiments/exp_016_higher_bc_weights.yaml`](../configs/experiments/exp_016_higher_bc_weights.yaml) | Completed | Loss weight test evaluating higher BC weights (10.0 wall, 10.0 symmetry) at 64k epochs. |
| | [`configs/experiments/exp_017_lower_physics_weight.yaml`](../configs/experiments/exp_017_lower_physics_weight.yaml) | Completed | Loss weight test evaluating lower physics residual weight (0.1) at 64k epochs. |
| **Package Init** | [`src/pinn_pipe/__init__.py`](../src/pinn_pipe/__init__.py) | Completed | Root package docstring describing the PINN pipe flow library. |
| | [`src/pinn_pipe/models/__init__.py`](../src/pinn_pipe/models/__init__.py) | Completed | Exposes public interfaces: `BasePINN`, `MLP`. |
| | [`src/pinn_pipe/physics/__init__.py`](../src/pinn_pipe/physics/__init__.py) | Completed | Exposes public interfaces: `analytical_solution`, `bc_symmetry`, `bc_wall`, `pde_residual`. |
| | [`src/pinn_pipe/training/__init__.py`](../src/pinn_pipe/training/__init__.py) | Completed | Exposes public interfaces: `Trainer`, `bc_symmetry_loss`, `bc_wall_loss`, `physics_loss`, `sample_bc`, `sample_interior`, `total_loss`. |
| | [`src/pinn_pipe/evaluation/__init__.py`](../src/pinn_pipe/evaluation/__init__.py) | Completed | Exposes public interfaces: `compute_metrics`, `plot_error`, `plot_loss_curve`, `plot_velocity_profile`. |
| | [`src/pinn_pipe/utils/__init__.py`](../src/pinn_pipe/utils/__init__.py) | Completed | Exposes public interfaces: `Config`, dataclasses (`ModelConfig`, `PhysicsConfig`, `RunConfig`, `SamplingConfig`, `TrainingConfig`), `load_config`, `validate_config`, `grad`, `grad2`, I/O utilities, and `set_seed`. |
| **Models** | [`src/pinn_pipe/models/base.py`](../src/pinn_pipe/models/base.py) | Completed | Abstract base class (`BasePINN`) defining model contract `forward(x) -> u`. |
| | [`src/pinn_pipe/models/mlp.py`](../src/pinn_pipe/models/mlp.py) | Completed | Fully connected network (`MLP`) with configurable width, depth, and activation functions. |
| **Physics** | [`src/pinn_pipe/physics/pipe_flow.py`](../src/pinn_pipe/physics/pipe_flow.py) | Completed | Governing Navier-Stokes PDE residual, wall no-slip BC, symmetry BC, and analytical solution. |
| **Training** | [`src/pinn_pipe/training/sampler.py`](../src/pinn_pipe/training/sampler.py) | Completed | Uniform sampling of interior collocation points and boundary points with autograd gradient tracking. |
| | [`src/pinn_pipe/training/losses.py`](../src/pinn_pipe/training/losses.py) | Completed | Mean squared residual loss terms (PDE, wall BC, symmetry BC) and composite weighted loss. |
| | [`src/pinn_pipe/training/trainer.py`](../src/pinn_pipe/training/trainer.py) | Completed | Training loop coordinator: optimizer execution, progress logging, and artifact persistence. |
| **Evaluation** | [`src/pinn_pipe/evaluation/evaluator.py`](../src/pinn_pipe/evaluation/evaluator.py) | Completed | Computes error metrics ($L_2$, maximum, relative $L_2$) and renders diagnostic plots. |
| **Utilities** | [`src/pinn_pipe/utils/config.py`](../src/pinn_pipe/utils/config.py) | Completed | Strongly typed config dataclasses, hierarchical YAML loader, and sanity validation. |
| | [`src/pinn_pipe/utils/derivatives.py`](../src/pinn_pipe/utils/derivatives.py) | Completed | Autograd derivative wrappers for first (`grad`) and second (`grad2`) partial derivatives. |
| | [`src/pinn_pipe/utils/io.py`](../src/pinn_pipe/utils/io.py) | Completed | Run directory management, metric serialization (`json`), history logging (`csv`), and model weights. |
| | [`src/pinn_pipe/utils/reproducibility.py`](../src/pinn_pipe/utils/reproducibility.py) | Completed | Synchronous seed initialization across Python `random`, NumPy, and PyTorch (CPU/cuDNN). |
| **Scripts** | [`scripts/train.py`](../scripts/train.py) | Completed | Command-line training pipeline: config loading, model building, training, evaluation, and saving. |
| | [`scripts/compare_runs.py`](../scripts/compare_runs.py) | Completed | Scans results directory, aggregates metrics and configs into `comparison.csv` and `comparison.png`. |
| **Test Suite** | [`tests/__init__.py`](../tests/__init__.py) | Completed | Test suite package initialization. |
| | [`tests/test_models.py`](../tests/test_models.py) | Completed | Unit tests for network construction, input/output tensors, shapes, and `__repr__`. |
| | [`tests/test_physics.py`](../tests/test_physics.py) | Completed | Unit tests for analytical solutions, boundary conditions, and PDE residual shapes. |
| | [`tests/test_sampler.py`](../tests/test_sampler.py) | Completed | Unit tests for interior and boundary sampling shapes, bounds, and autograd flags. |
| | [`tests/test_losses.py`](../tests/test_losses.py) | Completed | Unit tests verifying loss non-negativity, scalar types, and composite dictionary outputs. |
| **Documentation** | [`docs/codebase_overview.md`](codebase_overview.md) | Completed | Architectural and file-by-file catalog of the repository. |
| | [`docs/physics.md`](physics.md) | Completed | Detailed derivation of Hagen-Poiseuille flow, force balances, and boundary conditions. |
| | [`docs/experiments.md`](experiments.md) | Completed | Guide for creating, running, saving, and comparing experiment runs. |

---

## 3. Detailed Component Documentation

### 3.1 Project & Configuration

- **[`README.md`](../README.md)**
  - Comprehensive user-facing overview covering the physical problem, PINN methodology, directory tree, installation, usage, testing, and results placeholders.

- **[`pyproject.toml`](../pyproject.toml)**
  - Configures packaging via `setuptools >= 68` with editable install support (`pip install -e ".[dev]"`).
  - Specifies runtime dependencies: `numpy`, `matplotlib`, `pyyaml`, `scipy`.
  - Defines test dependencies: `pytest`, `pytest-cov`.
  - Configures pytest test discovery paths targeting `tests/`.

- **[`CHANGELOG.md`](../CHANGELOG.md)**
  - Documents project versions, changes, and fixes, notably documenting the resolution of the PDE residual $dp/dz$ factor (corrected from 2.0 to 4.0) and the boundary offset avoidance ($r \in [0.01, R]$) to eliminate the $1/r$ coordinate singularity.

- **[`configs/base.yaml`](../configs/base.yaml)**
  - **Physics**: Pipe radius $R = 1.0$, dynamic viscosity $\mu = 1.0$, maximum velocity range $u_{\max} \in [0.5, 2.0]$.
  - **Model**: Architecture type `mlp`, depth `3`, width `32`, activation `tanh`.
  - **Sampling**: `1000` interior collocation points per iteration.
  - **Training**: `4000` epochs, learning rate `1e-3`, optimizer `adam`, loss weights ($w_{\text{physics}} = 1.0$, $w_{\text{wall}} = 1.0$, $w_{\text{sym}} = 1.0$).
  - **Run**: Seed `42`, precision `float64`.

- **[`configs/experiments/exp_001_mlp_baseline.yaml`](../configs/experiments/exp_001_mlp_baseline.yaml)**
  - Baseline experiment configuration with metadata identifier `exp_001_mlp_baseline`.
  - Inherits all baseline defaults from `base.yaml`.

- **[`configs/experiments/exp_002_lower_physics_weight.yaml`](../configs/experiments/exp_002_lower_physics_weight.yaml)**
  - Tests whether scaling down the physics residual loss weight relative to boundary conditions stabilizes training.
  - Overrides: `loss_weight_physics: 0.01`, `epochs: 8000`, `learning_rate: 0.0001`.

- **[`configs/experiments/exp_003_double_epochs.yaml`](../configs/experiments/exp_003_double_epochs.yaml)**
  - Tests whether doubling baseline training iterations improves accuracy while retaining baseline weights and learning rate.
  - Overrides: `epochs: 8000`.

- **[`configs/experiments/exp_004_quadruple_epochs.yaml`](../configs/experiments/exp_004_quadruple_epochs.yaml)**
  - Evaluates error scaling behavior when quadrupling baseline training duration.
  - Overrides: `epochs: 16000`.

- **[`configs/experiments/exp_005_8x_epochs.yaml`](../configs/experiments/exp_005_8x_epochs.yaml)**
  - Evaluates performance at 8x baseline training duration to identify the empirical convergence plateau.
  - Overrides: `epochs: 32000`.

- **[`configs/experiments/exp_006_16x_epochs.yaml`](../configs/experiments/exp_006_16x_epochs.yaml)**
  - Assesses whether the power-law error halving persists through extended asymptotic training at 16x baseline duration.
  - Overrides: `epochs: 64000`.

- **[`configs/experiments/exp_007_wider_network.yaml`](../configs/experiments/exp_007_wider_network.yaml)**
  - Evaluates increasing layer width to 64 (depth=3) at 64,000 epochs.
  - Overrides: `model.hidden_layer_width: 64`, `training.epochs: 64000`.

- **[`configs/experiments/exp_008_widest_network.yaml`](../configs/experiments/exp_008_widest_network.yaml)**
  - Evaluates increasing layer width to 128 (depth=3) at 64,000 epochs.
  - Overrides: `model.hidden_layer_width: 128`, `training.epochs: 64000`.

- **[`configs/experiments/exp_009_deeper_network.yaml`](../configs/experiments/exp_009_deeper_network.yaml)**
  - Evaluates increasing depth to 6 (width=32) at 64,000 epochs (overall best performing configuration).
  - Overrides: `model.hidden_layer_depth: 6`, `training.epochs: 64000`.

- **[`configs/experiments/exp_010_deepest_network.yaml`](../configs/experiments/exp_010_deepest_network.yaml)**
  - Evaluates increasing depth to 9 (width=32) at 64,000 epochs.
  - Overrides: `model.hidden_layer_depth: 9`, `training.epochs: 64000`.

- **[`configs/experiments/exp_011_sgd_optimizer.yaml`](../configs/experiments/exp_011_sgd_optimizer.yaml)**
  - Evaluates standard SGD optimizer against Adam using depth=6 network at 64,000 epochs.
  - Overrides: `training.optimizer: "sgd"`, `model.hidden_layer_depth: 6`, `training.epochs: 64000`.

- **[`configs/experiments/exp_012_lbfgs_optimizer.yaml`](../configs/experiments/exp_012_lbfgs_optimizer.yaml)**
  - Evaluates quasi-Newton L-BFGS optimizer against Adam using depth=6 network at 64,000 epochs.
  - Overrides: `training.optimizer: "lbfgs"`, `model.hidden_layer_depth: 6`, `training.epochs: 64000`.

- **[`configs/experiments/exp_013_sigmoid_activation.yaml`](../configs/experiments/exp_013_sigmoid_activation.yaml)**
  - Evaluates Sigmoid activation function vs Tanh on depth=6 network at 64,000 epochs.
  - Overrides: `model.activation: "sigmoid"`, `model.hidden_layer_depth: 6`, `training.epochs: 64000`.

- **[`configs/experiments/exp_014_more_collocation_points.yaml`](../configs/experiments/exp_014_more_collocation_points.yaml)**
  - Evaluates 5,000 interior collocation points per epoch vs baseline 1,000.
  - Overrides: `sampling.num_interior_points: 5000`, `model.hidden_layer_depth: 6`, `training.epochs: 64000`.

- **[`configs/experiments/exp_015_fewer_collocation_points.yaml`](../configs/experiments/exp_015_fewer_collocation_points.yaml)**
  - Evaluates 200 interior collocation points per epoch vs baseline 1,000.
  - Overrides: `sampling.num_interior_points: 200`, `model.hidden_layer_depth: 6`, `training.epochs: 64000`.

- **[`configs/experiments/exp_016_higher_bc_weights.yaml`](../configs/experiments/exp_016_higher_bc_weights.yaml)**
  - Evaluates higher boundary condition loss weights ($w_{\text{wall}}=10.0, w_{\text{sym}}=10.0$).
  - Overrides: `training.loss_weights.bc_wall: 10.0`, `training.loss_weights.bc_symmetry: 10.0`, `model.hidden_layer_depth: 6`, `training.epochs: 64000`.

- **[`configs/experiments/exp_017_lower_physics_weight.yaml`](../configs/experiments/exp_017_lower_physics_weight.yaml)**
  - Evaluates reduced physics loss weight ($w_{\text{physics}}=0.1$).
  - Overrides: `training.loss_weights.physics: 0.1`, `model.hidden_layer_depth: 6`, `training.epochs: 64000`.

---

### 3.2 Package Interface Architecture (`src/pinn_pipe/`)

All subpackages implement explicit public API exposure via `__init__.py` files to enable short, clean imports:

```python
# Models
from pinn_pipe.models import BasePINN, MLP

# Physics
from pinn_pipe.physics import (
    analytical_solution,
    bc_symmetry,
    bc_wall,
    pde_residual,
)

# Training
from pinn_pipe.training import (
    Trainer,
    bc_symmetry_loss,
    bc_wall_loss,
    physics_loss,
    sample_bc,
    sample_interior,
    total_loss,
)

# Evaluation
from pinn_pipe.evaluation import (
    compute_metrics,
    plot_error,
    plot_loss_curve,
    plot_velocity_profile,
)

# Utilities
from pinn_pipe.utils import (
    Config,
    ModelConfig,
    PhysicsConfig,
    RunConfig,
    SamplingConfig,
    TrainingConfig,
    create_run_dir,
    grad,
    grad2,
    load_config,
    load_model,
    save_config,
    save_history,
    save_metrics,
    save_model,
    set_seed,
    validate_config,
)
```

---

### 3.3 Neural Network Architecture (`pinn_pipe.models`)

- **[`src/pinn_pipe/models/base.py`](../src/pinn_pipe/models/base.py)**
  - Class: `BasePINN(ABC)`
  - Defines the interface contract requiring `forward(x: torch.Tensor) -> torch.Tensor`.
  - Input: coordinate batch of shape $(N, 2)$ where each row is $[r, u_{\max}]$.
  - Output: predicted velocity tensor of shape $(N, 1)$.

- **[`src/pinn_pipe/models/mlp.py`](../src/pinn_pipe/models/mlp.py)**
  - Class: `MLP(BasePINN, nn.Module)`
  - Fully connected feedforward architecture:
    $$\text{Linear}(2, W) \to \left[ \sigma \to \text{Linear}(W, W) \right]^D \to \sigma \to \text{Linear}(W, 1)$$
    where $W$ is `hidden_layer_width` and $D$ is `hidden_layer_depth`.
  - Supported activation functions ($\sigma$): `tanh`, `relu`, `sigmoid`.
  - Implements `__repr__` for human-readable architecture inspection.

---

### 3.4 Physics & Governing Equations (`pinn_pipe.physics`)

- **[`src/pinn_pipe/physics/pipe_flow.py`](../src/pinn_pipe/physics/pipe_flow.py)**
  - `pde_residual(model, r, u_max, config)`: Evaluates the steady laminar Navier-Stokes momentum residual in cylindrical coordinates:
    $$\mathcal{R}_{\text{pde}} = \mu \left( \frac{\partial^2 u}{\partial r^2} + \frac{1}{r} \frac{\partial u}{\partial r} \right) - \frac{\partial p}{\partial z}$$
    where $\frac{\partial p}{\partial z} = -\frac{4\mu u_{\max}}{R^2}$.
  - `bc_wall(model, r_bc, u_max)`: Computes the no-slip boundary condition residual at the wall:
    $$\mathcal{R}_{\text{wall}} = u(R) - 0$$
  - `bc_symmetry(model, r_bc, u_max)`: Computes the centerline symmetry residual at $r = 0$:
    $$\mathcal{R}_{\text{sym}} = \left. \frac{\partial u}{\partial r} \right|_{r=0} - 0$$
  - `analytical_solution(r, u_max, R)`: Evaluates the exact parabolic Hagen-Poiseuille velocity profile:
    $$u_{\text{exact}}(r) = u_{\max} \left( 1 - \frac{r^2}{R^2} \right)$$

---

### 3.5 Training & Optimization (`pinn_pipe.training`)

- **[`src/pinn_pipe/training/sampler.py`](../src/pinn_pipe/training/sampler.py)**
  - `sample_interior(n, R, u_max_min, u_max_max)`: Samples $n$ radial points uniformly from $[0.01, R]$ (using a lower bound of $0.01$ to avoid the numerical $1/r$ coordinate singularity at the pipe axis) with `requires_grad=True` and paired $u_{\max} \in [u_{\max,\min}, u_{\max,\max}]$.
  - `sample_bc(n, R, u_max_min, u_max_max)`: Generates paired boundary points for the wall ($r = R$) and centerline ($r = 0$) with `requires_grad=True`.

- **[`src/pinn_pipe/training/losses.py`](../src/pinn_pipe/training/losses.py)**
  - `physics_loss(model, r, u_max, config)`: MSE of the PDE residual over interior collocation points.
  - `bc_wall_loss(model, r_wall, u_max)`: MSE of predicted velocities at $r = R$.
  - `bc_symmetry_loss(model, r_sym, u_max)`: MSE of velocity gradients $\frac{\partial u}{\partial r}$ at $r = 0$.
  - `total_loss(...)`: Weighted sum returning the scalar loss tensor for backpropagation along with a dictionary of detached float metrics:
    $$\mathcal{L}_{\text{total}} = w_{\text{pde}}\mathcal{L}_{\text{pde}} + w_{\text{wall}}\mathcal{L}_{\text{wall}} + w_{\text{sym}}\mathcal{L}_{\text{sym}}$$

- **[`src/pinn_pipe/training/trainer.py`](../src/pinn_pipe/training/trainer.py)**
  - Class: `Trainer`
  - Orchestrates the full training loop, Adam optimizer steps, epoch history logging, console progress reporting (every 500 epochs and epoch 1), and final artifact serialization (`save_config`, `save_history`, `save_model`).

---

### 3.6 Evaluation & Visualization (`pinn_pipe.evaluation`)

- **[`src/pinn_pipe/evaluation/evaluator.py`](../src/pinn_pipe/evaluation/evaluator.py)**
  - `compute_metrics(model, config)`: Compares model predictions against the exact analytical solution over a fine radial grid across multiple evaluation $u_{\max}$ values ($0.5, 1.0, 1.5, 2.0$), computing:
    - $L_2$ Error: $\sqrt{\frac{1}{N}\sum (\hat{u} - u_{\text{exact}})^2}$
    - Maximum Absolute Error: $\max |\hat{u} - u_{\text{exact}}|$
    - Relative $L_2$ Error: $\frac{\|\hat{u} - u_{\text{exact}}\|_2}{\|u_{\text{exact}}\|_2}$
  - `plot_velocity_profile(model, config, run_dir)`: Generates comparison curves of predicted vs. exact velocity profiles saved to `run_dir/plots/velocity_profile.png`.
  - `plot_loss_curve(history, run_dir)`: Plots semilog loss trajectories for total, physics, wall BC, and symmetry BC losses saved to `run_dir/plots/loss_curve.png`.
  - `plot_error(model, config, run_dir)`: Plots pointwise absolute errors across radius $r$ saved to `run_dir/plots/error_plot.png`.

---

### 3.7 Utilities (`pinn_pipe.utils`)

- **[`src/pinn_pipe/utils/config.py`](../src/pinn_pipe/utils/config.py)**
  - Strongly typed dataclasses: `PhysicsConfig`, `ModelConfig`, `SamplingConfig`, `TrainingConfig`, `RunConfig`, and `Config`.
  - `load_config(base_path, experiment_path)`: Deep-merges experiment-specific overrides into baseline configurations.
  - `validate_config(config)`: Enforces positivity of physical constants ($R > 0$, $\mu > 0$), valid parameter ranges ($u_{\max,\min} < u_{\max,\max}$), and positive epochs/learning rates.

- **[`src/pinn_pipe/utils/derivatives.py`](../src/pinn_pipe/utils/derivatives.py)**
  - `grad(output, input)`: First partial derivative $\frac{\partial u}{\partial r}$ via `torch.autograd.grad(create_graph=True)`.
  - `grad2(output, input)`: Second partial derivative $\frac{\partial^2 u}{\partial r^2}$ computed by differentiating the first derivative graph.

- **[`src/pinn_pipe/utils/io.py`](../src/pinn_pipe/utils/io.py)**
  - `create_run_dir(experiment_name, results_dir)`: Generates timestamped run directories (`results/<experiment>_YYYYMMDD_HHMMSS/plots/`).
  - `save_config(config, run_dir)`: Exports merged configuration to `config.yaml`.
  - `save_metrics(metrics, run_dir)`: Writes evaluation metrics dictionary to `metrics.json`.
  - `save_history(history, run_dir)`: Serializes epoch-by-epoch loss records to `history.csv`.
  - `save_model(model, run_dir)` / `load_model(model, run_dir)`: Saves and loads model state dictionary (`model.pt`).

- **[`src/pinn_pipe/utils/reproducibility.py`](../src/pinn_pipe/utils/reproducibility.py)**
  - `set_seed(seed)`: Synchronously seeds Python standard `random`, `numpy`, PyTorch CPU and CUDA generators, and enforces deterministic cuDNN execution.

---

### 3.8 Execution Scripts (`scripts/`)

- **[`scripts/train.py`](../scripts/train.py)**
  - Main training pipeline executable:
    ```bash
    python scripts/train.py --config configs/experiments/exp_001_mlp_baseline.yaml
    ```
  - Loads configuration, sets seeds, creates run directories, trains the model, renders evaluation plots, evaluates quantitative metrics, and saves all run artifacts.

- **[`scripts/compare_runs.py`](../scripts/compare_runs.py)**
  - Automated cross-experiment comparison utility:
    ```bash
    python scripts/compare_runs.py --results-dir results/
    ```
  - Scans `results/`, aggregates configs and metrics from completed runs, exports a side-by-side CSV summary table (`results/comparison.csv`), and renders a horizontal $L_2$ error ranking bar chart with exact value annotations and gridlines (`results/comparison.png`).

---

### 3.9 Test Suite (`tests/`)

The test suite covers unit verification across all subsystems:
- **[`tests/test_models.py`](../tests/test_models.py)**: Model forward pass shapes, repr strings, and invalid configuration guards.
- **[`tests/test_physics.py`](../tests/test_physics.py)**: Analytical solution boundary values ($u(0) = u_{\max}$, $u(R) = 0$), shape assertions, and PDE/BC residual tensor outputs.
- **[`tests/test_sampler.py`](../tests/test_sampler.py)**: Interior and boundary sampling tensor shapes, value ranges, and `requires_grad=True` verification.
- **[`tests/test_losses.py`](../tests/test_losses.py)**: Individual loss non-negativity, scalar output types, and composite dictionary structure.

---

### 3.10 Documentation Suite (`docs/`)

- **[`docs/physics.md`](physics.md)**: Thorough exposition of fluid mechanics concepts: shear, dynamic viscosity, no-slip condition, steady force balance, Navier-Stokes reduction in cylindrical coordinates, boundary conditions, and analytical parabolic solution.
- **[`docs/experiments.md`](experiments.md)**: Comprehensive guide on the two-tier configuration system, creating new experiments, artifact directory structure, and comparison procedures.
- **[`docs/codebase_overview.md`](codebase_overview.md)**: Master technical documentation and file catalog.

---

### 3.11 Experimental Results & Benchmark Runs (`results/`)

The framework outputs all completed experiment runs to timestamped folders under `results/`.

#### Benchmark Performance Summary

The table below summarizes the latest benchmark evaluations following the resolution of the physical $dp/dz$ factor and axis singularity:

| Run Directory / Experiment | Config File | Architecture (D, W, Act) | Epochs | Optimizer | Loss Weights ($w_{\text{pde}}, w_{\text{wall}}, w_{\text{sym}}$) | $L_2$ Error | Max Error | Relative $L_2$ Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exp_001_mlp_baseline` | `exp_001_mlp_baseline.yaml` | (3, 32, tanh) | 4,000 | Adam | (1.0, 1.0, 1.0) | 0.0241 | 0.0318 | 0.0491 (4.91%) |
| `exp_002_lower_physics_weight` | `exp_002_lower_physics_weight.yaml` | (3, 32, tanh) | 8,000 | Adam | (1.0, 1.0, 1.0) | 0.0391 | 0.0500 | 0.0707 (7.07%) |
| `exp_003_double_epochs` | `exp_003_double_epochs.yaml` | (3, 32, tanh) | 8,000 | Adam | (1.0, 1.0, 1.0) | 0.0143 | 0.0170 | 0.0294 (2.94%) |
| `exp_004_quardruple_epochs` | `exp_004_quadruple_epochs.yaml` | (3, 32, tanh) | 16,000 | Adam | (1.0, 1.0, 1.0) | 0.0077 | 0.0083 | 0.0144 (1.44%) |
| `exp_005_8x_epochs` | `exp_005_8x_epochs.yaml` | (3, 32, tanh) | 32,000 | Adam | (1.0, 1.0, 1.0) | 0.0041 | 0.0046 | 0.0079 (0.79%) |
| `exp_006_16x_epochs` | `exp_006_16x_epochs.yaml` | (3, 32, tanh) | 64,000 | Adam | (1.0, 1.0, 1.0) | 0.0021 | 0.0023 | 0.0052 (0.52%) |
| `exp_007_wider_network` | `exp_007_wider_network.yaml` | (3, 64, tanh) | 64,000 | Adam | (1.0, 1.0, 1.0) | 0.0034 | 0.0036 | 0.0064 (0.64%) |
| `exp_008_widest_network` | `exp_008_widest_network.yaml` | (3, 128, tanh) | 64,000 | Adam | (1.0, 1.0, 1.0) | 0.0081 | 0.0093 | 0.0154 (1.54%) |
| `exp_009_deeper_network` | `exp_009_deeper_network.yaml` | (6, 32, tanh) | 64,000 | Adam | (1.0, 1.0, 1.0) | **0.0015** | **0.0021** | **0.0024 (0.24%)** |
| `exp_010_deepest_network` | `exp_010_deepest_network.yaml` | (9, 32, tanh) | 64,000 | Adam | (1.0, 1.0, 1.0) | 0.0044 | 0.0068 | 0.0046 (0.46%) |
| `exp_011_sgd_optimizer` | `exp_011_sgd_optimizer.yaml` | (6, 32, tanh) | 64,000 | SGD | (1.0, 1.0, 1.0) | 0.0254 | 0.0279 | 0.0264 (2.64%) |
| `exp_012_lbfgs_optimizer` | `exp_012_lbfgs_optimizer.yaml` | (6, 32, tanh) | 64,000 | L-BFGS | (1.0, 1.0, 1.0) | 0.0291 | 0.0408 | 0.0399 (3.99%) |
| `exp_013_sigmoid_activation` | `exp_013_sigmoid_activation.yaml` | (6, 32, sigmoid) | 64,000 | Adam | (1.0, 1.0, 1.0) | 0.0213 | 0.0243 | 0.0359 (3.59%) |
| `exp_014_more_collocation_points` | `exp_014_more_collocation_points.yaml` | (6, 32, tanh) | 64,000 | Adam | (1.0, 1.0, 1.0) | 0.0018 | 0.0023 | 0.0025 (0.25%) |
| `exp_015_fewer_collocation_points` | `exp_015_fewer_collocation_points.yaml` | (6, 32, tanh) | 64,000 | Adam | (1.0, 1.0, 1.0) | 0.0067 | 0.0075 | 0.0114 (1.14%) |
| `exp_016_higher_bc_weights` | `exp_016_higher_bc_weights.yaml` | (6, 32, tanh) | 64,000 | Adam | (1.0, 10.0, 10.0) | 0.0016 | 0.0022 | 0.0031 (0.31%) |
| `exp_017_lower_physics_weight` | `exp_017_lower_physics_weight.yaml` | (6, 32, tanh) | 64,000 | Adam | (0.1, 1.0, 1.0) | 0.0021 | 0.0027 | 0.0039 (0.39%) |

#### Key Empirical Insights

1. **Epoch Scaling & Error Halving**:
   Successive doubling of training iterations ($4\text{k} \to 8\text{k} \to 16\text{k} \to 32\text{k} \to 64\text{k}$) exhibits consistent, power-law-like error halving, reaching 0.52% relative $L_2$ error on baseline architecture.
2. **Network Depth vs. Width**:
   Increasing layer width (64, 128) degraded performance on this 1D axisymmetric problem due to over-parameterization. Conversely, deepening to depth=6 yielded the overall best performing model (`exp_009`, 0.24% relative error), while depth=9 suffered from optimization degradation.
3. **Optimizer Selection**:
   Adam significantly outperformed both standard SGD (0.0254 error) and quasi-Newton L-BFGS (0.0291 error), which struggled with per-epoch collocation point resampling.
4. **Activation Functions**:
   Tanh demonstrated superior performance (0.0015 error) over Sigmoid (0.0213 error), owing to non-vanishing second-order derivatives in the Navier-Stokes residual.
5. **Collocation Point Density**:
   1,000 interior points per epoch represents an optimal efficiency sweet spot; increasing to 5,000 gave comparable error (0.0018), while dropping to 200 severely impaired accuracy (0.0067).
6. **Loss Weighting**:
   Equal loss weighting ($1.0, 1.0, 1.0$) was empirically confirmed as robust; higher BC weights slightly matched baseline (0.0016), while downweighting physics residual slightly degraded accuracy (0.0021).
7. **Physical Formula & Singularity Corrections**:
   Initial prototype runs (2026-09-12) had high error ($\approx 100\%$) due to two issues documented in [`CHANGELOG.md`](../CHANGELOG.md): an incorrect coefficient in $\partial p/\partial z$ (2 instead of 4) and numerical instability at $r \approx 0$ in the $1/r$ term. Addressing these reduced the baseline error by over $20\times$ immediately.

Each run folder contains:
- `config.yaml`: Merged configuration snapshot.
- `history.csv`: Per-epoch training losses.
- `metrics.json`: Quantitative error metrics against exact Hagen-Poiseuille solution.
- `model.pt`: Serialized PyTorch model weights.
- `plots/`: `velocity_profile.png`, `loss_curve.png`, and `error_plot.png`.
