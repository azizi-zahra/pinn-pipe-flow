"""
Script to train the model for pinn-pipe-flow.

Training starts from here. Loads config, builds model, runs trainer,
evaluates results, and saves all artifacts to the run directory.

Usage:
    python scripts/train.py --config configs/experiments/exp_001_mlp_baseline.yaml
    python scripts/train.py --config configs/experiments/exp_001_mlp_baseline.yaml --device cuda
"""

import argparse
import os

from plyer import notification

from pinn_pipe.evaluation import (
    compute_metrics,
    plot_error,
    plot_loss_curve,
    plot_velocity_profile,
)
from pinn_pipe.models import HardBCMLP, MLP
from pinn_pipe.training import Trainer
from pinn_pipe.utils import (
    create_run_dir,
    get_device,
    load_config,
    save_config,
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu'. Defaults to cpu.",
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
    # Create timestamped run directory and save config
    # -------------------------------------------------------------------------
    experiment_name = os.path.splitext(os.path.basename(args.config))[0]
    run_dir = create_run_dir(experiment_name)
    save_config(config, run_dir)
    print(f"Run directory: {run_dir}")

    # -------------------------------------------------------------------------
    # Build model and move to device
    # -------------------------------------------------------------------------
    supported_models = {
        "mlp": MLP,
        "hard_bc_mlp": HardBCMLP,
    }

    if config.model.type not in supported_models:
        raise ValueError(
            f"Unsupported model type: {config.model.type}. "
            f"Choose from {list(supported_models.keys())}"
        )

    device = get_device(args.device)
    if config.model.type == "hard_bc_mlp" or getattr(config.model, "hard_bc", False):
        model = HardBCMLP(config.model, R=config.physics.R)
    else:
        model = supported_models[config.model.type](config.model)
    model = model.to(device)
    print(f"Model: {model}")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    trainer = Trainer(model, config, run_dir, device)
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
    metrics["training_time_seconds"] = trainer.training_time
    metrics["device"] = str(device)
    save_metrics(metrics, run_dir)

    # -------------------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------------------
    print("\n--- Results ---")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nAll artifacts saved to: {run_dir}")

    # -------------------------------------------------------------------------
    # Notify user that training is complete
    # -------------------------------------------------------------------------
    try:
        notification.notify(
            title="pinn-pipe-flow",
            message=f"Training complete: {experiment_name}",
        )
    except NotImplementedError:
        pass
    print("\a")


if __name__ == "__main__":
    main()