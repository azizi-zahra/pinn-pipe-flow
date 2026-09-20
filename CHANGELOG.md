# Changelog

All notable changes to this project will be documented here.

---

## [Unreleased]

### Added
- **Streamlit Dashboard** (`dashboard/app.py`): Interactive web UI to compare PINN predictions vs. analytical solutions across radial positions and $u_{\max}$ flow conditions.
- **Hugging Face Hub Integration**: Cloud deployment support (`snapshot_download`) with local fallback and `upload_to_hf.py` model sync script.
- **Run Config Loader** (`load_run_config`): Utility to load saved `config.yaml` files directly without base config merging.
- **Hard BC Architecture** (`HardBCMLP`): Exact wall no-slip and centerline symmetry enforcement by construction.
- **Extended Activations & Schedulers**: Added `SiLU`, `GELU`, `Mish`, `SinActivation`, and learning rate scheduler factory (`CosineAnnealingLR`, `StepLR`, etc.).
- **Hybrid Optimizer**: Two-stage Adam $\to$ L-BFGS training support.

### Fixed
- **Hybrid L-BFGS Stability**: Fixed collocation points during L-BFGS phase to prevent line search destabilization from stochastic resampling; lowered L-BFGS lr to 0.1.
- **Git Tracking**: Removed `results/` from Git tracking for clean cloud deployment with HF Hub.
- **Coordinate Singularity**: Sampler avoids $r < 0.01$ to eliminate $1/r$ residual explosion.
- **Physics Factor**: Corrected $dp/dz$ factor in PDE residual from 2.0 to 4.0.

### Experiments Summary
- **Phases 1–8**: Found optimal soft-BC baseline: 64k epochs, depth 6, width 32, Tanh, Adam ($\text{lr}=10^{-3}$), 1k collocation points, equal loss weights ($w=1.0$). Best: `exp_009` ($L_2 = 0.00155$, rel err 0.24%).
- **Phase 9 (Advanced)**:
  - `exp_025` (Cosine LR): $L_2 = 0.00196$ (0.22% rel err). Smooth late convergence.
  - `exp_026` (Hybrid Adam $\to$ L-BFGS): $L_2 = 0.00165$ (0.19% rel err). Best soft-BC model.
  - `exp_027` (SiLU): $L_2 = 0.00246$ (0.46% rel err). Stable, slightly behind Tanh.
  - `exp_028` (Hard BC): $L_2 = 5.34 \times 10^{-8}$ ($7.34 \times 10^{-6}\%$ rel err). **New SOTA**, 5 orders of magnitude improvement over soft BCs.