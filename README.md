# pinn-pipe-flow

Physics-Informed Neural Network (PINN) for modeling steady laminar flow in a horizontal circular pipe (Hagen-Poiseuille flow) using PyTorch.

---

## Overview

The `pinn-pipe-flow` project uses a Physics-Informed Neural Network (PINN) to solve the classic fluid mechanics problem of steady, laminar flow through a horizontal circular pipe of radius $R$. In classical viscous pipe flow, friction against the wall and viscous shear forces across the fluid create a parabolic velocity profile.

Unlike conventional machine learning models that require large datasets of numerical or experimental observations, a **PINN is trained entirely from physical first principles**:
- The model minimizes the residual of the governing Navier-Stokes momentum equation evaluated at collocation points.
- Boundary conditions (no-slip at the wall, symmetry at the centerline) are enforced directly through penalization loss terms.
- No ground-truth velocity measurements or external training data are required.

The network is formulated as a **parametric model**: it takes two scalar inputs—the radial position $r \in [0, R]$ and the maximum centerline velocity $u_{\max}$—and outputs the predicted fluid velocity $u(r)$:

$$\hat{u} = \text{Model}(r, u_{\max})$$

By parameterizing across a continuous range of $u_{\max}$ values, a single trained network learns an entire family of flow solutions across varying pressure gradients. The codebase is designed with a modular architecture that can be readily extended to more advanced neural operator architectures, such as DeepONet.

---

## Physics

The flow is governed by the steady, axial Navier-Stokes momentum equation in cylindrical coordinates $(r, \theta, z)$:

$$\mu \left( \frac{d^2 u}{dr^2} + \frac{1}{r}\frac{du}{dr} \right) = \frac{dp}{dz}$$

where $\mu$ is the dynamic viscosity and $\frac{dp}{dz} < 0$ is the constant axial pressure gradient driving the flow.

### Boundary Conditions

A unique physical solution requires two boundary conditions:
1. **Wall No-Slip Condition**: Fluid adhering to the solid pipe wall is stationary:
   $$u(R) = 0$$
2. **Centerline Symmetry Condition**: Axisymmetric flow reaches zero slope at the pipe axis:
   $$\left. \frac{du}{dr} \right|_{r=0} = 0$$

### Analytical Solution

Integrating the differential equation with both boundary conditions yields the exact **Hagen-Poiseuille parabolic velocity profile**:

$$u(r) = u_{\max} \left( 1 - \frac{r^2}{R^2} \right)$$

where $u_{\max} = -\frac{R^2}{4\mu}\frac{dp}{dz}$ is the peak centerline velocity. This exact closed-form solution serves as the benchmark for evaluating PINN accuracy.

For the complete derivation, force balance diagrams, and discussion of physical assumptions, see [`docs/physics.md`](docs/physics.md).

---

## Project Structure

```
pinn-pipe-flow/
├── configs/                          # Configuration files for base settings and experiments
│   ├── base.yaml                     # Global default baseline parameters
│   └── experiments/                  # Experiment-specific override files
│       └── exp_001_mlp_baseline.yaml # Baseline MLP experiment configuration
├── docs/                             # Project documentation
│   ├── codebase_overview.md          # Full architectural catalog of all project modules
│   ├── experiments.md                # Guide for configuring, running, and comparing experiments
│   └── physics.md                    # Detailed mathematical derivation and physical concepts
├── scripts/                          # Command-line entry points
│   ├── train.py                      # Main training and evaluation execution script
│   └── compare_runs.py               # Cross-experiment comparison and metric aggregation
├── src/                              # Source code root
│   └── pinn_pipe/                    # Core Python package
│       ├── __init__.py               # Package docstring exposing top-level namespace
│       ├── models/                   # Neural network architectures (BasePINN, MLP)
│       ├── physics/                  # Governing equations, boundary conditions, and exact solution
│       ├── training/                 # Collocation point sampling, loss functions, and Trainer loop
│       ├── evaluation/               # Evaluation metrics (L2, max error) and visualization plots
│       └── utils/                    # Config loading, autograd derivatives, I/O, and seed setting
├── tests/                            # Unit test suite covering all core modules (pytest)
└── results/                          # Auto-generated experiment outputs (gitignored)
```

---

## Installation

### 1. Install PyTorch

PyTorch must be installed manually first to ensure compatibility with your system's hardware and CUDA drivers. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to find the correct installation command for your environment.

Example (CPU-only):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install Package Dependencies

From the repository root:

- **For development** (includes test runners and coverage tools):
  ```bash
  pip install -e ".[dev]"
  ```
- **For standard usage**:
  ```bash
  pip install -e "."
  ```

---

## Usage

### Running an Experiment

To train a model using an experiment configuration:

```bash
python scripts/train.py --config configs/experiments/exp_001_mlp_baseline.yaml
```

During execution, the script automatically:
1. Loads and validates the merged configuration (`configs/base.yaml` + experiment overrides).
2. Creates a timestamped directory under `results/<experiment_name>_<timestamp>/`.
3. Trains the neural network using the Adam optimizer with autograd-evaluated physics losses.
4. Generates diagnostic plots and saves all evaluation artifacts.

### Saved Artifacts

Each experiment run saves the following files into its run directory:

| File | Description |
| :--- | :--- |
| `config.yaml` | Exact snapshot of the fully merged run configuration. |
| `history.csv` | Per-epoch log of total, physics, wall BC, and symmetry BC losses. |
| `metrics.json` | Quantitative evaluation metrics ($L_2$ error, maximum error, relative $L_2$ error). |
| `model.pt` | Serialized PyTorch state dictionary containing trained network weights. |
| `plots/velocity_profile.png` | Comparison plot of predicted vs. exact analytical velocity profiles. |
| `plots/loss_curve.png` | Semilog plot of training loss convergence trajectories. |
| `plots/error_plot.png` | Pointwise absolute error across radius $r$ for test velocities. |

### Comparing Experiments

To aggregate and compare results across completed runs:

```bash
python scripts/compare_runs.py
```

This scans `results/` and generates:
- `results/comparison.csv`: Summary table listing hyperparameters and error metrics side by side.
- `results/comparison.png`: Bar chart ranking models by overall $L_2$ error.

For a comprehensive guide on designing configurations and overriding parameters, see [`docs/experiments.md`](docs/experiments.md).

---

## Running Tests

Run the unit test suite with `pytest`:

```bash
pytest
```

---

## Results

*Results will be added here after experiments are complete.*

Key evaluation figures from `results/` can be found in `docs/images/` once copied there.
