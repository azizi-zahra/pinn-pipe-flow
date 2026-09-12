"""
Script to train the model for pinn-pipe-flow.

Training starts from here. Loads config, builds model, runs trainer,
evaluates results, and saves all artifacts to the run directory.

Usage:
    python scripts/train.py --config configs/experiments/exp_001_mlp_baseline.yaml
"""

import argparse
import os

from pinn_pipe.evaluation import (
    compute_metrics,
    plot_error,
    plot_loss_curve,
    plot_velocity_profile,
)
from pinn_pipe.models import MLP
from pinn_pipe.training import Trainer
from pinn_pipe.utils import (
    create_run_dir,
    load_config,
    save_metrics,
    set_seed,
    validate_config,
)


def main() -> None:
    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train a PINN model for pipe flow.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config yaml file.",
    )
    args = parser.parse_args()
    print(f"Loading config: {args.config}")

    # -------------------------------------------------------------------------
    # Load and validate config
    # -------------------------------------------------------------------------
    config = load_config("configs/base.yaml", args.config)
    validate_config(config)

    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    # -------------------------------------------------------------------------
    set_seed(config.run.seed)

    # -------------------------------------------------------------------------
    # Create timestamped run directory
    # -------------------------------------------------------------------------
    experiment_name = os.path.splitext(os.path.basename(args.config))[0]
    run_dir = create_run_dir(experiment_name)
    print(f"Run directory: {run_dir}")

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    supported_models = {
        "mlp": MLP,
    }

    if config.model.type not in supported_models:
        raise ValueError(
            f"Unsupported model type: {config.model.type}. "
            f"Choose from {list(supported_models.keys())}"
        )

    model = supported_models[config.model.type](config.model)
    print(f"Model: {model}")

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    trainer = Trainer(model, config, run_dir)
    trainer.train()
    trainer.save()

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    plot_velocity_profile(model, config, run_dir)
    plot_loss_curve(trainer.history, run_dir)
    plot_error(model, config, run_dir)

    # -------------------------------------------------------------------------
    # Compute and save metrics
    # -------------------------------------------------------------------------
    metrics = compute_metrics(model, config)
    save_metrics(metrics, run_dir)

    # -------------------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------------------
    print("\n--- Results ---")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"\nAll artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()