# Running and Comparing Experiments

This guide explains how to configure, run, and compare Physics-Informed Neural Network (PINN) experiments in the `pinn-pipe-flow` project.

---

## 1. Overview of the Experiment Workflow

The project is designed around a modular, reproducible experiment workflow:

```
+------------------------+      +-----------------------------------------+
|   configs/base.yaml    |  +   | configs/experiments/exp_*.yaml      |
|   (Default parameters) |      | (Experiment-specific overrides)         |
+------------------------+      +-----------------------------------------+
                    |                                |
                    +--------------------------------+
                                    |
                            [ load_config() ]
                                    |
                                    v
                         [ scripts/train.py ]
                                    |
                                    v
+-------------------------------------------------------------------------+
| results/<experiment_name>_<timestamp>/                                  |
|   ├── config.yaml          (Full merged configuration)                  |
|   ├── history.csv          (Per-epoch training losses)                  |
|   ├── metrics.json         (Final L2, max, relative errors)             |
|   ├── model.pt             (Trained PyTorch weights)                    |
|   └── plots/                                                            |
|       ├── velocity_profile.png (PINN vs Exact profiles)                 |
|       ├── loss_curve.png       (Loss trajectories on log scale)         |
|       └── error_plot.png       (Pointwise absolute error)               |
+-------------------------------------------------------------------------+
                                    |
                                    v
                       [ scripts/compare_runs.py ]
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
         results/comparison.csv          results/comparison.png
         (Summary metric table)          (L2 error bar chart)
```

---

## 2. How the Configuration System Works

Experiment configurations follow a two-tier hierarchical structure:

1. **`configs/base.yaml` (Base Configuration)**:
   Defines standard default values for all parameters across physics, architecture, sampling, optimizer, and reproducibility settings.
2. **`configs/experiments/<name>.yaml` (Experiment Overrides)**:
   Contains metadata about the experiment and only the specific parameter sections that differ from `base.yaml`.

### 2.1 The Base Configuration (`configs/base.yaml`)

```yaml
physics:
  R: 1.0            # Pipe radius
  mu: 1.0           # Dynamic viscosity
  u_max_range:
    min: 0.5        # Minimum u_max sampled during training
    max: 2.0        # Maximum u_max sampled during training

model:
  type: "mlp"       # Architecture name
  hidden_layer_depth: 3
  hidden_layer_width: 32
  activation: "tanh"

sampling:
  num_interior_points: 1000   # Collocation points inside the pipe

training:
  epochs: 4000
  learning_rate: 0.001
  optimizer: "adam"
  loss_weight_physics: 1.0      # Weight for PDE residual loss
  loss_weight_bc_wall: 1.0      # Weight for no-slip BC at r=R
  loss_weight_bc_symmetry: 1.0  # Weight for symmetry BC at r=0

run:
  seed: 42
  dtype: "float64"
```

### 2.2 How Overrides are Merged

When an experiment is launched, `load_config("configs/base.yaml", experiment_yaml)` reads both files:
- The `overrides` section of the experiment file is merged section-by-section onto the baseline settings using Python dictionary union (`base_data[section] | values`).
- Any key not explicitly overridden retains its default value from `base.yaml`.
- The merged configuration is validated by `validate_config()` to guarantee physical consistency (e.g., $R > 0$, $\mu > 0$, $u_{\max,\min} < u_{\max,\max}$, $\text{epochs} > 0$, $\text{learning\_rate} > 0$).

---

## 3. How to Create a New Experiment Config File

All experiment configurations live in `configs/experiments/`.

### Step-by-Step Instructions

1. Create a new YAML file using the naming pattern `exp_<number>_<short_description>.yaml` (e.g., `configs/experiments/exp_002_deep_mlp.yaml`).
2. Add a `metadata` block with `name` and `description`. The `name` must match the file's basename (without `.yaml`).
3. Add an `overrides` block defining only the parameters you want to change.

### Example 1: Testing a Deeper Network

```yaml
# configs/experiments/exp_002_deep_mlp.yaml
metadata:
  name: "exp_002_deep_mlp"
  description: "Test a deeper 5-layer MLP architecture with wider hidden layers."

overrides:
  model:
    hidden_layer_depth: 5
    hidden_layer_width: 64
    activation: "tanh"
```

### Example 2: Adjusting Loss Weights and Learning Rate

```yaml
# configs/experiments/exp_003_high_bc_weight.yaml
metadata:
  name: "exp_003_high_bc_weight"
  description: "Heavily penalize boundary condition violations to improve near-wall accuracy."

overrides:
  training:
    epochs: 5000
    learning_rate: 0.0005
    loss_weight_physics: 1.0
    loss_weight_bc_wall: 10.0
    loss_weight_bc_symmetry: 5.0
```

---

## 4. How to Run an Experiment

Run training from the repository root using `scripts/train.py`:

```bash
python scripts/train.py --config configs/experiments/exp_001_mlp_baseline.yaml
```

### What Happens During Execution:

1. **Config Loading & Validation**: Loads `base.yaml`, applies experiment overrides, and validates inputs.
2. **Reproducibility Initialization**: Sets random seeds across Python `random`, NumPy, and PyTorch CPU/GPU.
3. **Run Directory Creation**: Automatically generates a unique timestamped directory under `results/`.
4. **Model Instantiation**: Builds the specified network architecture (e.g., `MLP`).
5. **Training Execution**: Evaluates collocation samples, computes PDE residuals and boundary losses, performs Adam gradient steps, and logs progress every 100 epochs.
6. **Artifact Persistence**: Saves model weights, training history, and merged config.
7. **Post-Training Evaluation**: Computes accuracy metrics against the exact Hagen-Poiseuille analytical solution and generates diagnostic plots.
8. **Summary Output**: Prints quantitative error metrics to the console.

---

## 5. Artifacts Saved in the Results Directory

Every run creates a timestamped folder:

```
results/<experiment_name>_YYYYMMDD_HHMMSS/
```

### File Contents

| File Path | Format | Description |
| :--- | :---: | :--- |
| `config.yaml` | YAML | The complete, merged configuration used for this exact run. Ensures full reproducibility. |
| `history.csv` | CSV | Epoch-by-epoch training logs with columns: `epoch`, `loss_total`, `loss_physics`, `loss_bc_wall`, `loss_bc_symmetry`. |
| `metrics.json` | JSON | Final evaluation metrics averaged across test velocities: `l2_error`, `max_error`, and `relative_l2_error`. |
| `model.pt` | PyTorch Binary | Serialized state dictionary (`model.state_dict()`) containing all trained network weights and biases. |
| `plots/velocity_profile.png` | PNG | Compares predicted PINN profiles $\hat{u}(r)$ against exact analytical curves $u(r) = u_{\max}(1 - r^2/R^2)$ across multiple test velocities ($u_{\max} \in \{0.5, 1.0, 1.5, 2.0\}$). |
| `plots/loss_curve.png` | PNG | Semilog plot displaying the convergence history of `total`, `physics`, `bc_wall`, and `bc_symmetry` losses over all epochs. |
| `plots/error_plot.png` | PNG | Pointwise absolute error $|\hat{u}(r) - u_{\text{exact}}(r)|$ plotted as a function of radial coordinate $r \in [0, R]$ for each test velocity. |

---

## 6. How to Compare Multiple Runs

Once multiple experiment configurations have been trained, compare them using `scripts/compare_runs.py`:

```bash
python scripts/compare_runs.py
```

To scan a custom results directory:

```bash
python scripts/compare_runs.py --results-dir results/
```

### How the Comparison Tool Works

The script iterates through all subdirectories in `results/`, reading `config.yaml` and `metrics.json` from each completed run. It compiles the data into two summary artifacts:
- `results/comparison.csv`
- `results/comparison.png`

---

## 7. Understanding the Comparison Table and Plot

### 7.1 The Comparison Table (`results/comparison.csv`)

The CSV table enables side-by-side comparison of hyperparameters and performance metrics:

| Column Name | Description |
| :--- | :--- |
| `run_name` | The timestamped experiment directory name. |
| `model_type` | Network architecture (e.g., `mlp`). |
| `hidden_layer_depth` | Number of hidden layers in the network. |
| `hidden_layer_width` | Number of neurons per hidden layer. |
| `activation` | Non-linear activation function (e.g., `tanh`). |
| `epochs` | Total training iterations. |
| `learning_rate` | Optimizer learning rate. |
| `optimizer` | Optimization algorithm (e.g., `adam`). |
| `loss_weight_physics` | Weight factor $w_{\text{pde}}$ applied to PDE residual loss. |
| `loss_weight_bc_wall` | Weight factor $w_{\text{wall}}$ applied to no-slip wall condition. |
| `loss_weight_bc_symmetry`| Weight factor $w_{\text{sym}}$ applied to centerline symmetry condition. |
| `l2_error` | Mean absolute $L_2$ error: $\sqrt{\frac{1}{N}\sum (\hat{u} - u_{\text{exact}})^2}$. |
| `max_error` | Maximum absolute pointwise error: $\max |\hat{u} - u_{\text{exact}}|$. |
| `relative_l2_error` | Normalized $L_2$ error: $\frac{\|\hat{u} - u_{\text{exact}}\|_2}{\|u_{\text{exact}}\|_2}$. |

### 7.2 The Comparison Plot (`results/comparison.png`)

A horizontal bar chart visualizing `l2_error` across all scanned runs:
- **Horizontal orientation**: Experiment names are arranged along the vertical axis from top to bottom, avoiding rotated text and ensuring long experiment identifiers remain completely legible.
- **Dynamic height**: Scales automatically with the number of experiment runs so bars never crowd or overlap.
- **Value annotations**: Direct numeric $L_2$ error values (formatted to 4 decimal places) are rendered at the tip of each bar.
- **Major and minor grids**: Dual-frequency vertical gridlines along the error axis and horizontal guide lines per experiment row provide precise visual reference.
- **Model ranking**: Enables immediate visual ranking to determine how architecture, optimization, or weighting changes impact accuracy.

---

## 8. Tips for Designing Effective Experiments

When formulating hypotheses and configuring experiments, consider the following trade-offs:

### 8.1 Network Architecture (`model`)

- **Activation Function**:
  - Always use smooth, infinitely differentiable ($C^\infty$) activation functions like `tanh` for PINNs.
  - **Avoid `relu`**: The second derivative of ReLU is zero almost everywhere ($\frac{d^2}{dr^2}\text{ReLU}(x) = 0$). Since the PDE residual depends directly on the second derivative $\frac{d^2 u}{dr^2}$, ReLU networks cannot propagate physics gradients effectively.
- **Depth vs. Width**:
  - For simple Hagen-Poiseuille flow, a shallow network (depth 3, width 32 to 64) is typically sufficient to capture the parabolic profile.
  - Increasing depth beyond 5–6 layers without skip connections can cause optimization slowdowns without accuracy gains.

### 8.2 Loss Weighting (`training.loss_weight_*`)

PINN loss surfaces often suffer from gradient competition between the PDE residual and boundary conditions:
- **High Wall Errors**: If the model predicts non-zero velocity at the wall ($r = R$), increase `loss_weight_bc_wall` from `1.0` to `5.0` or `10.0`.
- **Under-Fitting the Profile Curvature**: If boundary conditions are satisfied but the profile fails to match the parabolic curvature in the interior, increase `loss_weight_physics`.

### 8.3 Collocation Point Density (`sampling.num_interior_points`)

- Collocation points are resampled dynamically every training epoch.
- 1,000 points per epoch provides good radial coverage for 1D pipe flow.
- Increasing to 2,000–5,000 points can help reduce variance if the loss fluctuates between epochs, at the cost of marginally slower epoch execution.

### 8.4 Optimization & Learning Rate (`training`)

- **Learning Rate**: For Adam, `1e-3` is a robust baseline. If the loss plateaus early or oscillates erratically, try reducing to `5e-4` or `1e-4` with longer epochs (e.g., 6,000–8,000).
- **Epoch Count**: 4,000 epochs is typically enough for baseline convergence. Monitor `plots/loss_curve.png` to ensure losses have flattened before concluding training.

---

## 9. Completed Experiments

### Phase 1 -- Epochs

Epochs were varied systematically from 4,000 to 64,000 to analyze training convergence and error scaling. The $L_2$ error consistently halved with each doubling of epochs, and no convergence plateau was observed within this range. As a result, 64,000 epochs was chosen as the standard training duration for subsequent phases.

| Experiment | Epochs | L2 Error |
| :--- | :---: | :---: |
| `exp_001` | 4000 | 0.0241 |
| `exp_003` | 8000 | 0.0143 |
| `exp_004` | 16000 | 0.0077 |
| `exp_005` | 32000 | 0.0041 |
| `exp_006` | 64000 | 0.0021 |

### Phase 2 -- Architecture

Network width and depth were evaluated independently while fixing training duration at 64,000 epochs:
- **Width Variation**: Wider networks performed worse than the baseline width of 32. Because the 1D Hagen-Poiseuille problem is physically low-dimensional, wider layers introduced unnecessary capacity that impaired optimization.
- **Depth Variation**: Increasing depth improved performance up to depth 6, beyond which accuracy degraded (likely due to vanishing gradients in deeper un-residualized architectures).

| Experiment | Width | Depth | L2 Error |
| :--- | :---: | :---: | :---: |
| `exp_006` | 32 | 3 | 0.0021 |
| `exp_007` | 64 | 3 | 0.0034 |
| `exp_008` | 128 | 3 | 0.0081 |
| `exp_009` | 32 | 6 | 0.0015 |
| `exp_010` | 32 | 9 | 0.0044 |

### Phase 3 -- Optimizers

Using the best architecture (width=32, depth=6) at 64,000 epochs, alternative optimizers were evaluated against Adam:
- **SGD (`exp_011`)**: Significantly underperformed Adam ($L_2$ error 0.0254 vs 0.0015), as standard gradient descent struggled with the non-convex PINN composite loss surface.
- **L-BFGS (`exp_012`)**: Yielded $L_2$ error 0.0291, struggling with full-batch step sizes across varying $u_{\max}$ conditioning.

| Experiment | Optimizer | L2 Error | Max Error | Relative L2 Error |
| :--- | :---: | :---: | :---: | :---: |
| `exp_009` (Baseline) | Adam | 0.0015 | 0.0021 | 0.24% |
| `exp_011` | SGD | 0.0254 | 0.0279 | 2.64% |
| `exp_012` | L-BFGS | 0.0291 | 0.0408 | 3.99% |

### Phase 4 -- Activation Functions

- **Sigmoid (`exp_013`)**: Compared against Tanh on depth=6. Sigmoid exhibited vanishing gradient issues in higher-order autograd derivatives ($\partial^2 u/\partial r^2$), resulting in an $L_2$ error of 0.0213 (over $14\times$ worse than Tanh).

| Experiment | Activation | L2 Error | Max Error | Relative L2 Error |
| :--- | :---: | :---: | :---: | :---: |
| `exp_009` (Baseline) | Tanh | 0.0015 | 0.0021 | 0.24% |
| `exp_013` | Sigmoid | 0.0213 | 0.0243 | 3.59% |

### Phase 5 -- Collocation Points

Evaluated the impact of interior collocation point density (baseline: 1,000):
- **5,000 Points (`exp_014`)**: Achieved competitive accuracy ($L_2$ error 0.0018), confirming 1,000 points was already near the point of diminishing returns for 1D radial sampling.
- **200 Points (`exp_015`)**: Sparsely sampled interior led to degraded accuracy ($L_2$ error 0.0067, $\sim 4.5\times$ higher error), demonstrating the necessity of adequate radial resolution.

| Experiment | Collocation Points | L2 Error | Max Error | Relative L2 Error |
| :--- | :---: | :---: | :---: | :---: |
| `exp_015` | 200 | 0.0067 | 0.0075 | 1.14% |
| `exp_009` (Baseline) | 1,000 | 0.0015 | 0.0021 | 0.24% |
| `exp_014` | 5,000 | 0.0018 | 0.0023 | 0.25% |

### Phase 6 -- Loss Weighting

Investigated relative loss weighting between PDE residuals and boundary conditions:
- **Higher BC Weights (`exp_016`)**: Increasing $w_{\text{wall}} = 10.0$ and $w_{\text{sym}} = 10.0$ maintained very high accuracy ($L_2$ error 0.0016), nearly matching the baseline.
- **Lower Physics Weight (`exp_017`)**: Reducing $w_{\text{physics}} = 0.1$ increased error slightly ($L_2$ error 0.0021), confirming equal weighting ($1.0, 1.0, 1.0$) remains optimal.

| Experiment | $w_{\text{physics}}$ | $w_{\text{wall}}$ | $w_{\text{sym}}$ | L2 Error | Max Error | Relative L2 Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `exp_009` (Baseline) | 1.0 | 1.0 | 1.0 | 0.0015 | 0.0021 | 0.24% |
| `exp_016` | 1.0 | 10.0 | 10.0 | 0.0016 | 0.0022 | 0.31% |
| `exp_017` | 0.1 | 1.0 | 1.0 | 0.0021 | 0.0027 | 0.39% |

### Current Best Configuration

Across all 17 experiments, the optimal configuration is **`exp_009_deeper_network`**:
- **Architecture**: MLP (Width = 32, Depth = 6, Activation = Tanh)
- **Training**: Epochs = 64,000, Optimizer = Adam ($\text{lr} = 10^{-3}$)
- **Collocation Points**: 1,000 interior points per epoch
- **Loss Weights**: $w_{\text{physics}} = 1.0, w_{\text{wall}} = 1.0, w_{\text{sym}} = 1.0$
- **Performance**: $L_2\text{ Error} = 0.00155$, $\text{Max Error} = 0.00212$, $\text{Relative } L_2\text{ Error} = 0.24\%$ (sub-0.25% error across all $u_{\max} \in [0.5, 2.0]$)
