# Changelog

All notable changes to this project will be documented here.

---

## [Unreleased]

### Added
- Training time is now printed at the end of training and saved to metrics.json as `training_time_seconds`
- compare_runs.py now shows only the most recent run per experiment instead of all runs
- compare_runs.py now uses experiment name without timestamp on x-axis for cleaner plots

### Fixed
- Sampler now avoids r values close to zero (minimum 0.01) to prevent 1/r singularity in PDE residual which was causing physics loss to explode
- Corrected dp_dz factor in PDE residual from 2.0 to 4.0 -- the wrong factor was causing the model to predict exactly half the correct velocity for all u_max values

### Experiments
- Phase 1 (Epochs): Ran exp_001 through exp_006. Error consistently halved with each doubling of epochs. No plateau found up to 64000 epochs. Best result: L2 error 0.0021 at 64000 epochs.
- Phase 2 (Architecture): Ran exp_007 through exp_010. Wider networks performed worse. Deeper networks helped up to depth 6, beyond which performance degraded. Best architecture: width=32, depth=6, L2 error 0.0015.