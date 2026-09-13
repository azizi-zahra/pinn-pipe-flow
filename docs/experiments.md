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
  loss_weights:
    physics: 1.0      # Weight for PDE residual loss
    bc_wall: 1.0      # Weight for no-slip BC at r=R
    bc_symmetry: 1.0  # Weight for symmetry BC at r=0

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
    loss_weights:
      physics: 1.0
      bc_wall: 10.0
      bc_symmetry: 5.0
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

A bar chart visualizing `l2_error` across all scanned runs on the vertical axis:
- Allows immediate visual ranking of model performance.
- Quickly identifies whether an architectural or weighting change improved or degraded physical accuracy.

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

### 8.2 Loss Weighting (`training.loss_weights`)

PINN loss surfaces often suffer from gradient competition between the PDE residual and boundary conditions:
- **High Wall Errors**: If the model predicts non-zero velocity at the wall ($r = R$), increase `loss_weights.bc_wall` from `1.0` to `5.0` or `10.0`.
- **Under-Fitting the Profile Curvature**: If boundary conditions are satisfied but the profile fails to match the parabolic curvature in the interior, increase `loss_weights.physics`.

### 8.3 Collocation Point Density (`sampling.num_interior_points`)

- Collocation points are resampled dynamically every training epoch.
- 1,000 points per epoch provides good radial coverage for 1D pipe flow.
- Increasing to 2,000–5,000 points can help reduce variance if the loss fluctuates between epochs, at the cost of marginally slower epoch execution.

### 8.4 Optimization & Learning Rate (`training`)

- **Learning Rate**: For Adam, `1e-3` is a robust baseline. If the loss plateaus early or oscillates erratically, try reducing to `5e-4` or `1e-4` with longer epochs (e.g., 6,000–8,000).
- **Epoch Count**: 4,000 epochs is typically enough for baseline convergence. Monitor `plots/loss_curve.png` to ensure losses have flattened before concluding training.
