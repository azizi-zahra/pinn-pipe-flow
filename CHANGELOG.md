# Changelog

All notable changes to this project will be documented here.

---

## [Unreleased]

### Added
- Hard boundary condition ansatz model (`HardBCMLP`) strictly enforcing wall no-slip $u(R) = 0$ and centerline symmetry $\partial u/\partial r(0) = 0$ by architectural construction
- Extended activation functions in `MLP`: `SinActivation` (SIREN-style sinusoidal), `SiLU` (`Swish`), `GELU`, and `Mish`
- Learning rate scheduler utility `build_lr_scheduler` supporting `CosineAnnealingLR`, `StepLR`, `MultiStepLR`, `ExponentialLR`, and `ReduceLROnPlateau`
- Hybrid optimizer support in `Trainer`: initial training with Adam followed by L-BFGS second-order fine-tuning at configurable `hybrid_switch_epoch`
- Device management utility `get_device` with `--device` CLI flag (`cuda`, `cpu`) in `scripts/train.py`
- Training time is now printed at the end of training and saved to metrics.json as `training_time_seconds`
- `scripts/compare_runs.py` now aggregates the most recent run per experiment and renders an enhanced horizontal bar chart with annotations and gridlines
- Comprehensive unit tests: `tests/test_schedulers.py` and exact boundary condition / activation tests in `tests/test_models.py`

### Fixed
- Sampler now avoids r values close to zero (minimum 0.01) to prevent 1/r singularity in PDE residual which was causing physics loss to explode
- Corrected dp_dz factor in PDE residual from 2.0 to 4.0 -- the wrong factor was causing the model to predict exactly half the correct velocity for all u_max values
- Flattened `loss_weights` configuration mapping to match `TrainingConfig` dataclass attributes

### Experiments
- Phase 1 (Epochs): Ran exp_001 through exp_006. Error consistently halved with each doubling of epochs. No plateau found up to 64000 epochs. Best result: L2 error 0.0021 at 64000 epochs.
- Phase 2 (Architecture): Ran exp_007 through exp_010. Wider networks performed worse. Deeper networks helped up to depth 6, beyond which performance degraded. Best architecture: width=32, depth=6, L2 error 0.0015.
- Phase 3 (Optimizers): Ran exp_011 through exp_012. SGD and LBFGS both performed significantly worse than Adam. Adam remains the best optimizer for this problem. LBFGS underperformed likely due to resampling points every epoch which destabilizes its line search.
- Phase 4 (Activation Functions): Ran exp_013. Sigmoid activation performed much worse than tanh (L2 error 0.021 vs 0.0015). Tanh remains the best activation for this problem due to its smooth higher-order derivatives needed for the PDE residual.
- Phase 5 (Sampling): Ran exp_014 through exp_015. More collocation points (5000) barely improved over 1000. Fewer points (200) hurt significantly. 1000 points is the sweet spot for this problem.
- Phase 6 (Loss Weights): Ran exp_016 through exp_017. Higher BC weights and lower physics weight both had minimal effect. Equal weights of 1.0 remain the best choice.
- Phase 7 (Learning Rate & Epoch Scaling): Ran exp_018, exp_019, exp_021. Lower learning rate (1e-4) converged slower with higher error (0.0053). Higher learning rate (1e-2) diverged with unstable gradients (L2 error 0.913). Longer training (128,000 epochs) plateaued around 0.0019, confirming 64k epochs with lr=1e-3 is the optimal baseline.
- Phase 8 (Physics Loss Weight Sweep): Ran exp_020, exp_022, exp_023, exp_024 ($w_{\text{physics}} \in \{5, 10, 50, 100\}$). Heavy upweighting of physics residuals progressively degraded accuracy (L2 error worsened from 0.0062 at $w=5$ to 0.0544 at $w=100$) due to stiff gradient competition against boundary condition terms.
- Phase 9 (Advanced Optimization & Architectural Innovation): Added exp_025 through exp_028 testing CosineAnnealingLR scheduling (exp_025), hybrid two-stage Adam -> L-BFGS optimization (exp_026), SiLU non-saturating smooth activation (exp_027), and exact hard boundary condition enforcement via HardBCMLP ansatz (exp_028).