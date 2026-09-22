# AGENTS.md — pinn-pipe-flow

Read this file before taking any action in this project.
This file is the source of truth for conventions, current project state,
and agent behavior rules.

---

## Mandatory first steps

1. Read this file completely.
2. If the task involves creating a new experiment config, read `docs/experiments.md`
   to confirm the next experiment number before writing any file.
3. If the task involves any source file, read that file before editing it.

---

## Project overview

A Physics-Informed Neural Network (PINN) that models developing laminar pipe flow.

**Problem:** Steady axisymmetric incompressible flow in a circular pipe, from
flat inlet profile to developing Hagen-Poiseuille profile.

**Governing equations:** Boundary layer momentum equation + continuity equation.

**Network:** MLP with inputs `(r/R, x/L, Re_norm)` and outputs `(u, v)`.
- `r/R` — normalized radial position, in [0, 1]
- `x/L` — normalized axial position, in [0, 1]
- `Re_norm` — Reynolds number normalized to [0, 1] via `(Re - Re_min) / (Re_max - Re_min)`
- `u` — axial velocity
- `v` — radial velocity

**Important:** Re must be normalized before being passed to the network.
Raw Re values (100–500) are two orders of magnitude larger than the other
inputs and will cause training failure if passed unnormalized.

---

## Environment

- Language: Python 3.12
- Framework: PyTorch
- OS: Ubuntu Linux
- Activate virtual environment: `pytorch` (zsh alias)
- Run python: `pyt` (zsh alias)
- Install project: `pip install -e ".[dev]"` from project root
- Run tests: `torch-pytest` (aliased to use venv python)
- GPU flag: `--device cuda` (defaults to cpu)
- Run training: `python scripts/train.py --config configs/experiments/<name>.yaml --device cuda`

---

## File structure (key files only)

```
configs/
  base.yaml                        default config for all experiments
  experiments/exp_NNN_<name>.yaml  one file per experiment
src/pinn_pipe/
  models/mlp.py                    MLP: 3 inputs, 2 outputs
  physics/pipe_flow.py             PDE residuals and BC functions
  training/sampler.py              samples (r, x, Re) for interior and BCs
  training/losses.py               all loss terms and total_loss
  training/trainer.py              Trainer class
  evaluation/evaluator.py          plots and compute_metrics
  utils/config.py                  dataclasses and load_config
  utils/io.py                      save/load utilities
scripts/train.py                   training entry point
scripts/compare_runs.py            compares all runs, produces comparison.csv
docs/experiments.md                experiment tracking and conventions
AGENTS.md                          this file
```

---

## Rules — always follow these

**Never do these:**
- Do not run `git add`, or `git commit`. `git status` is ok.
  Tell the user what to commit at the end instead.
- Do not run the training script.
- Do not run `compare_runs.py`.
- Do not install packages unless the task explicitly requires it. If needed, tell the use.
- Do not edit `results/` — it is gitignored and runtime-generated.
- Do not create files outside the project structure without asking.

**Always do these:**
- Read a file before editing it. Never edit from memory.
- After any code change, run `torch-pytest tests/ -v` and report the result.
- After completing a task, tell the user exactly which files to commit
  and provide a commit message (see format below).
- If a task is ambiguous, ask one clarifying question before starting.
- If a task would require changing more than 5 files, summarize the plan
  first and wait for confirmation before making changes.

---

## Commit message format

```
<type(scope)>: <short description (imperative, under 72 chars)>
```

**Types:**
- `feat` — new feature or capability
- `fix` — bug fix
- `refactor` — restructuring without behavior change
- `experiment` — new experiment config or results
- `test` — test changes only
- `chore` — tooling, config, dependencies
- `docs` — documentation only

**Examples:**

```
feat(sampler): normalize Re input to [0,1] in sampler
```

```
experiment: add exp_030 with Re normalization and larger network
```

```
fix: enforce wall BC after mass correction in numerical_reference
```

One commit per logical unit of work. Do not batch unrelated changes.

---

## Code style

- Google-style docstrings on every public function and class.
- PEP8 import order: standard library, third party, local. Two blank lines
  between top-level definitions.
- Type hints on all function signatures.
- All `.numpy()` calls must use `.detach().cpu().numpy()` for GPU compatibility.
- Device obtained via `next(model.parameters()).device` in evaluation code.
- No em dashes in any output or comments.

---

## Experiment config conventions

- File name: `configs/experiments/exp_NNN_<short_description>.yaml`
- NNN is zero-padded to 3 digits.
- Experiment configs only specify fields that differ from `base.yaml`.
- Always include `experiment_name` and `description` fields.

**Example:**

```
# =============================================================================
# exp_028_hard_bc.yaml -- tests hard boundary condition enforcement (ansatz).
# Guarantees no-slip at wall and symmetry at centerline by network construction,
# allowing boundary condition loss weights to be eliminated (0.0).
# =============================================================================
metadata:
  name: "exp_028_hard_bc"
  description: "Test HardBCMLP ansatz strictly enforcing wall and symmetry boundary conditions."

overrides:
  model:
    type: "hard_bc_mlp"
    hidden_layer_depth: 6
  training:
    epochs: 64000
    loss_weight_physics: 1.0
    loss_weight_bc_wall: 0.0
    loss_weight_bc_symmetry: 0.0
```
---

## Loss terms (current formulation)

| Key in history dict    | What it measures                        |
|------------------------|-----------------------------------------|
| `loss_momentum`        | Axial momentum PDE residual             |
| `loss_continuity`      | Continuity equation residual            |
| `loss_bc_wall`         | No-slip: u(R, x) = 0                   |
| `loss_bc_wall_v`       | No radial flow: v(R, x) = 0            |
| `loss_bc_symmetry`     | Symmetry: du/dr(0, x) = 0              |
| `loss_bc_inlet`        | Flat inlet: u(r, 0) = U_in             |
| `loss_total`           | Weighted sum of all above               |

`U_in` is derived from Re: `U_in = Re * nu / (2 * R)`

---

## Evaluation Re values

`EVAL_RE_VALUES = [100.0, 300.0, 500.0]`

These are the Re values used for all evaluation plots and metrics.
Defined as a module-level constant in `src/pinn_pipe/evaluation/evaluator.py`.