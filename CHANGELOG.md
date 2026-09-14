# Changelog

All notable changes to this project will be documented here.

---

## [Unreleased]

### Added
- Training time is now printed at the end of training and saved to metrics.json as `training_time_seconds`
- compare_runs.py now shows only the most recent run per experiment instead of all runs
- Enhanced `scripts/compare_runs.py` comparison plot: converted to horizontal bar chart with dynamic scaling, bar value annotations, and dual-level gridlines for improved label readability.

### Fixed
- Sampler now avoids r values close to zero (minimum 0.01) to prevent 1/r singularity in PDE residual which was causing physics loss to explode
- Corrected dp_dz factor in PDE residual from 2.0 to 4.0 -- the wrong factor was causing the model to predict exactly half the correct velocity for all u_max values

### Experiments
- Phase 1 (Epochs): Ran exp_001 through exp_006. Error consistently halved with each doubling of epochs. No plateau found up to 64000 epochs. Best result: L2 error 0.0021 at 64000 epochs.
- Phase 2 (Architecture): Ran exp_007 through exp_010. Wider networks performed worse. Deeper networks helped up to depth 6, beyond which performance degraded. Best architecture: width=32, depth=6, L2 error 0.0015.
- Phase 3 (Optimizers): Ran exp_011 through exp_012. SGD and LBFGS both performed significantly worse than Adam. Adam remains the best optimizer for this problem. LBFGS underperformed likely due to resampling points every epoch which destabilizes its line search.
- Phase 4 (Activation Functions): Ran exp_013. Sigmoid activation performed much worse than tanh (L2 error 0.021 vs 0.0015). Tanh remains the best activation for this problem due to its smooth higher-order derivatives needed for the PDE residual.
- Phase 5 (Sampling): Ran exp_014 through exp_015. More collocation points (5000) barely improved over 1000. Fewer points (200) hurt significantly. 1000 points is the sweet spot for this problem.
- Phase 6 (Loss Weights): Ran exp_016 through exp_017. Higher BC weights and lower physics weight both had minimal effect. Equal weights of 1.0 remain the best choice.