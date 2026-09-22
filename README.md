# pinn-pipe-flow

Physics-Informed Neural Networks (PINNs) in PyTorch for laminar pipe flow, spanning both **1D fully developed Hagen-Poiseuille flow** and **2D axisymmetric developing pipe flow**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pinn-pipe-flow.streamlit.app/)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

> **TL;DR**: Solves Navier-Stokes and boundary layer equations without simulation data by minimizing PDE residuals via PyTorch autograd. Supports soft boundary penalties and exact hard boundary conditions (`HardBCMLP`). The 1D SOTA model reaches **$5.34 \times 10^{-8}$ $L_2$ error** ($0.000007\%$ relative error). The 2D formulation models coupled velocity fields $(u, v)$ across Reynolds numbers $Re \in [100, 500]$ in the entrance region. Explore models live in the interactive web dashboard.

---

## 🌐 Interactive Dashboard

An interactive Streamlit dashboard is deployed and available online:

👉 **Live App**: [https://pinn-pipe-flow.streamlit.app/](https://pinn-pipe-flow.streamlit.app/)

Or launch it locally from the repository:

```bash
streamlit run dashboard/app.py
```

### Dashboard Features

The dashboard includes a **Formulation Suite Selector** to explore both physical regimes:

- **1D Hagen-Poiseuille Flow Suite (`exp_001`–`exp_028`)**:
  - **Live Analytical Comparison**: Evaluates PINN velocity predictions against the exact parabolic solution $u(r) = u_{\max} (1 - r^2/R^2)$.
  - **Interactive Controls**: Real-time sliders for pipe radius $R$ and centerline velocity $u_{\max} \in [0.5, 2.0]$.
  - **Quantitative Diagnostics**: $L_2$ error, maximum absolute error, relative $L_2$ error, pointwise residual plots, and parameter sensitivity curves.
- **2D Developing Pipe Flow Suite (`exp_029`–`exp_030`)**:
  - **Coupled 2D Velocity Fields**: Visualizes axial velocity $u(r, x)$ and radial velocity $v(r, x)$ over $(r/R, x/L, Re)$.
  - **Entrance Region Dynamics**: Inspects boundary layer thickness $\delta(x)$, cross-sectional velocity profile development from flat inlet $U_{\text{in}}$ towards parabolic $2 U_{\text{in}}$, and centerline acceleration toward the entry length $L_e \approx 0.06 \cdot Re \cdot D$.
  - **Interactive 2D Visualizations**: Heatmaps with streamlines, quiver vector fields, and axial slice profiles across customizable $Re \in [100, 500]$.
- **Cloud-Ready with Hugging Face Hub**:
  - Automatically fetches trained weights via `snapshot_download` on Streamlit Cloud when `HF_REPO` is configured in secrets/environment, with automatic fallback to local `results/`.
  - Easy sync script included: `python upload_to_hf.py`.

---

## 🚀 Quickstart

### 1. Installation

```bash
# 1. Install PyTorch for your hardware (https://pytorch.org)
pip install torch

# 2. Install package
pip install -e "."        # Standard
pip install -e ".[dev]"    # With test & development dependencies
```

### 2. Train a Model

**Train the 1D SOTA hard boundary condition model:**
```bash
python scripts/train.py --config configs/experiments/exp_028_hard_bc.yaml --device cuda
```

**Train the 2D developing pipe flow model:**
```bash
python scripts/train.py --config configs/experiments/exp_030_longer_training_depth6.yaml --device cuda
# Or baseline developing flow:
python scripts/train.py --config configs/experiments/exp_029_developing_flow_baseline.yaml --device cuda
```

### 3. Compare All Runs

Aggregate metrics and benchmark all completed runs:
```bash
python scripts/compare_runs.py
```
Outputs a summary table to `results/comparison.csv` and ranked bar charts to `results/comparison.png`.

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Run Test Suite

```bash
pytest
```

---

## 📁 Project Structure

```
pinn-pipe-flow/
├── configs/            # base.yaml + 30 experiment configs (exp_001 to exp_030)
├── dashboard/          # Streamlit web app (app.py, requirements.txt)
├── docs/               # Technical documentation
│   ├── codebase_overview.md  # Architecture, modules, and complete results tables
│   ├── experiments.md        # Experiment guide & phase-by-phase findings
│   └── physics.md            # Hagen-Poiseuille derivation & hard BC proof
├── scripts/            # CLI tools (train.py, compare_runs.py)
├── src/pinn_pipe/      # Core Python package
│   ├── models/         # Neural network architectures (BasePINN, MLP, HardBCMLP)
│   ├── physics/        # Governing equations (1D Hagen-Poiseuille & 2D developing flow)
│   ├── training/       # 1D/2D collocation samplers, multi-objective losses, schedulers, trainer
│   ├── evaluation/     # Metrics, 1D/2D profile and field evaluators
│   └── utils/          # Config parser, autograd derivatives, I/O, device helpers
├── tests/              # Pytest unit tests (losses, models, physics, sampler, schedulers)
├── upload_to_hf.py     # Artifact sync script for Hugging Face Hub
└── results/            # Run outputs (model.pt, metrics.json, plots/) [gitignored]
```

---

## 📐 Physics Formulations

The codebase implements two complementary physical formulations:

### 1. 1D Steady Fully-Developed Hagen-Poiseuille Flow

Axial Navier-Stokes in cylindrical coordinates $(r)$:

$$\mu \left( \frac{d^2 u}{dr^2} + \frac{1}{r}\frac{du}{dr} \right) = \frac{dp}{dz}, \quad \text{with } u(R)=0 \text{ (wall)}, \left.\frac{du}{dr}\right|_{r=0}=0 \text{ (symmetry)}$$

Exact analytical parabolic solution:

$$u(r) = u_{\max} \left( 1 - \frac{r^2}{R^2} \right), \quad \text{where } u_{\max} = -\frac{R^2}{4\mu}\frac{dp}{dz}$$

For the hard boundary condition proof guaranteeing $u(R)=0$ and $u'(0)=0$ by network construction, see [`docs/physics.md`](docs/physics.md).

### 2. 2D Axisymmetric Developing Pipe Flow (Entrance Region)

Coupled boundary layer equations with continuity in cylindrical coordinates $(r, x)$:

$$\text{Axial Momentum: } \quad u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial r} - \frac{\nu}{r}\frac{\partial u}{\partial r} - \nu \frac{\partial^2 u}{\partial r^2} = 0$$

$$\text{Continuity: } \quad \frac{\partial u}{\partial x} + \frac{v}{r} + \frac{\partial v}{\partial r} = 0$$

Boundary conditions:
- **Inlet ($x=0$)**: Flat uniform profile $u(r, 0) = U_{\text{in}} = \frac{Re \cdot \nu}{2R}, \quad v(r, 0) = 0$
- **Pipe Wall ($r=R$)**: No-slip and impermeable wall $u(R, x) = 0, \quad v(R, x) = 0$
- **Centerline ($r=0$)**: Axial symmetry $\left.\frac{\partial u}{\partial r}\right|_{r=0} = 0, \quad v(0, x) = 0$
- **Downstream Limit ($x \to \infty$)**: Fully developed parabolic profile $u(r) \to 2 U_{\text{in}} \left( 1 - \frac{r^2}{R^2} \right)$

---

## 📊 Results & Benchmarks

The repository documents **30 systematic experiments across 10 structured phases**:

| Experiment | Flow Formulation | Key Highlights | $L_2$ Error | Relative Error | Key Takeaway |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `exp_001` | 1D | Baseline MLP (4k epochs) | 0.0241 | 4.91% | Initial baseline convergence |
| `exp_006` | 1D | Extended training (64k epochs) | 0.0021 | 0.52% | Error halved every $2\times$ epochs |
| `exp_009` | 1D | Deep MLP (Depth 6, Width 32) | 0.0015 | 0.24% | Optimal network geometry |
| `exp_025` | 1D | Cosine annealing LR scheduler | 0.0019 | 0.22% | Smooth terminal convergence |
| `exp_026` | 1D | Hybrid Adam $\to$ L-BFGS | 0.0016 | 0.19% | **Best Soft-BC model** |
| `exp_028` | 1D | **Hard BC (`HardBCMLP`)** | **$5.34 \times 10^{-8}$** | **$0.000007\%$** | **1D SOTA**: Exact BCs eliminate 99.999% error |
| `exp_029` | 2D | Baseline Developing Flow | — | — | Coupled $(u, v)$ boundary layer solver |
| `exp_030` | 2D | Scaled Developing Flow (Depth 6, 64k epochs) | — | — | High-capacity non-linear boundary layer resolution |

For in-depth analysis and complete tabular results across all phases, see [`docs/experiments.md`](docs/experiments.md) and [`docs/codebase_overview.md`](docs/codebase_overview.md).
