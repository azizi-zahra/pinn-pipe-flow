# Changelog

All notable changes to this project will be documented here.

---

## [Unreleased]

### Fixed
- Sampler now avoids r values close to zero to prevent 1/r singularity in the PDE residual, which was causing the physics loss to explode during training

- Corrected dp_dz factor in PDE residual from 2.0 to 4.0. The previous value was causing the model to consistently predict half the correct velocity value for all u_max inputs.