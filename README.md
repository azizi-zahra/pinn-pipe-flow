# pinn-pipe-flow

Physics-Informed Neural Network (PINN) for steady laminar pipe flow (Hagen-Poiseuille flow) in PyTorch.

> **TL;DR**: Solves Navier-Stokes without training data by minimizing PDE residuals. Supports soft BC penalties and exact hard boundary conditions (`HardBCMLP`). SOTA model reaches **$5.34 \times 10^{-8}$ $L_2$ error** ($0.000007\%$ relative error). Includes an interactive Streamlit dashboard.

---

## Interactive Dashboard

Launch the Streamlit web dashboard to inspect models and visualize velocity profiles:

```bash
streamlit run dashboard/app.py
```

- **Live Comparison**: PINN predictions vs. exact analytical parabolic solution $u(r) = u_{\max} (1 - r^2/R^2)$.
- **Interactive Controls**: Sliders for pipe radius $R$ and centerline velocity $u_{\max} \in [0.5, 2.0]$.
- **Real-Time Diagnostics**: $L_2$ error, max error, pointwise residual plots, and run hyperparameter cards.
- **Cloud Ready**: Deployed on Streamlit Cloud using Hugging Face Hub (`snapshot_download`). Set `HF_REPO = "username/repo"` in `.streamlit/secrets.toml` or sync with `python upload_to_hf.py`.

---

## Quickstart

### 1. Install

```bash
# 1. Install PyTorch for your system (https://pytorch.org)
pip install torch

# 2. Install package
pip install -e "."        # Standard
pip install -e ".[dev]"    # With test dependencies
```

### 2. Train a Model

```bash
python scripts/train.py --config configs/experiments/exp_028_hard_bc.yaml --device cuda
```

### 3. Compare All Runs

```bash
python scripts/compare_runs.py
```
Generates `results/comparison.csv` and `results/comparison.png`.

### 4. Run Tests

```bash
pytest
```

---

## Project Structure

```
pinn-pipe-flow/
├── configs/            # base.yaml + 28 experiment configs (exp_001 to exp_028)
├── dashboard/          # Streamlit web app (app.py, requirements.txt)
├── docs/               # Technical documentation
│   ├── codebase_overview.md  # Architecture, modules, and full results table
│   ├── experiments.md        # Experiment guide & phase-by-phase findings
│   └── physics.md            # Hagen-Poiseuille derivation & hard BC proof
├── scripts/            # train.py (training), compare_runs.py (benchmarking)
├── src/pinn_pipe/      # Core package (models, physics, training, evaluation, utils)
├── tests/              # Pytest unit tests (models, physics, samplers, losses, schedulers)
├── upload_to_hf.py     # Uploads model artifacts to Hugging Face Hub
└── results/            # Run outputs (model.pt, metrics.json, plots/) [gitignored]
```

---

## Physics Summary

Steady, axial Navier-Stokes in cylindrical coordinates:

$$\mu \left( \frac{d^2 u}{dr^2} + \frac{1}{r}\frac{du}{dr} \right) = \frac{dp}{dz}, \quad \text{with } u(R)=0 \text{ (wall)}, \left.\frac{du}{dr}\right|_{r=0}=0 \text{ (symmetry)}$$

Exact analytical solution:

$$u(r) = u_{\max} \left( 1 - \frac{r^2}{R^2} \right), \quad \text{where } u_{\max} = -\frac{R^2}{4\mu}\frac{dp}{dz}$$

For full derivations and the `HardBCMLP` algebraic proof, see [`docs/physics.md`](docs/physics.md).

---

## Results & Benchmarks

28 experiments across 9 systematic phases:

| Experiment | Highlights | $L_2$ Error | Relative Error | Key Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| `exp_001` | Baseline (4k epochs) | 0.0241 | 4.91% | Initial convergence |
| `exp_006` | Extended (64k epochs) | 0.0021 | 0.52% | Error halved every $2\times$ epochs |
| `exp_009` | Deep MLP (Depth 6, Width 32) | 0.0015 | 0.24% | Optimal network geometry |
| `exp_025` | Cosine LR scheduler | 0.0019 | 0.22% | Smoother late convergence |
| `exp_026` | Hybrid Adam $\to$ L-BFGS | 0.0016 | 0.19% | **Best Soft-BC model** |
| `exp_028` | **Hard BC (`HardBCMLP`)** | **$5.34 \times 10^{-8}$** | **$0.000007\%$** | **SOTA**: Exact BCs eliminate 99.999% error |

Detailed tables and insights: [`docs/experiments.md`](docs/experiments.md) & [`docs/codebase_overview.md`](docs/codebase_overview.md).
