# Changelog

All notable changes to this project will be documented here.

---

## [Unreleased]

### Fixed
- Sampler now avoids r values close to zero to prevent 1/r singularity in the PDE residual, which was causing the physics loss to explode during training