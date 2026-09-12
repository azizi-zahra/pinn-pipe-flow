"""
Trainer class for pinn-pipe-flow.

Manages the training loop, loss logging, progress printing,
and saving of all run artifacts.
"""

import torch

from pinn_pipe.models.base import BasePINN
from pinn_pipe.training.losses import total_loss
from pinn_pipe.training.sampler import sample_bc, sample_interior
from pinn_pipe.utils.config import Config
from pinn_pipe.utils.io import save_config, save_history, save_model


class Trainer:
    """Manages the PINN training loop for pipe flow.

    Handles optimizer setup, sampling, loss computation, history
    logging, and saving of all run artifacts.
    """

    def __init__(self, model: BasePINN, config: Config, run_dir: str) -> None:
        """Initializes the Trainer with model, config, and run directory.

        Args:
            model: A BasePINN instance to train.
            config: Fully populated Config object for this run.
            run_dir: Path to the timestamped results directory for this run.

        Raises:
            ValueError: If the optimizer name in config is not supported.
        """
        self.model = model
        self.config = config
        self.run_dir = run_dir
        self.history = []

        supported_optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "lbfgs": torch.optim.LBFGS,
        }

        if config.training.optimizer not in supported_optimizers:
            raise ValueError(
                f"Unsupported optimizer: {config.training.optimizer}. "
                f"Choose from {list(supported_optimizers.keys())}"
            )

        self.optimizer = supported_optimizers[config.training.optimizer](
            self.model.parameters(),
            lr=config.training.learning_rate,
        )

    def train(self) -> None:
        """Runs the full training loop for the configured number of epochs.

        At each epoch:
        - Samples fresh interior and BC collocation points
        - Computes total loss and all individual loss terms
        - Backpropagates and steps the optimizer
        - Logs loss values to history
        - Prints progress every 500 epochs
        """
        self.model.train()

        for epoch in range(1, self.config.training.epochs + 1):

            # sample fresh points every epoch
            r, u_max = sample_interior(
                n=self.config.sampling.num_interior_points,
                R=self.config.physics.R,
                u_max_min=self.config.physics.u_max_min,
                u_max_max=self.config.physics.u_max_max,
            )

            r_wall, r_sym, u_max_bc = sample_bc(
                n=self.config.sampling.num_interior_points,
                R=self.config.physics.R,
                u_max_min=self.config.physics.u_max_min,
                u_max_max=self.config.physics.u_max_max,
            )

            # compute losses
            losses = total_loss(
                model=self.model,
                r=r,
                u_max=u_max,
                r_wall=r_wall,
                r_sym=r_sym,
                u_max_bc=u_max_bc,
                physics_config=self.config.physics,
                training_config=self.config.training,
            )

            # backpropagation
            self.optimizer.zero_grad()
            losses["loss_total_tensor"].backward()
            self.optimizer.step()

            # log history
            self.history.append({
                "epoch": epoch,
                "loss_total": losses["loss_total"],
                "loss_physics": losses["loss_physics"],
                "loss_bc_wall": losses["loss_bc_wall"],
                "loss_bc_symmetry": losses["loss_bc_symmetry"],
            })

            # print progress every 500 epochs
            if epoch % 500 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:>5}/{self.config.training.epochs} | "
                    f"Total: {losses['loss_total']:.4e} | "
                    f"Physics: {losses['loss_physics']:.4e} | "
                    f"BC Wall: {losses['loss_bc_wall']:.4e} | "
                    f"BC Sym: {losses['loss_bc_symmetry']:.4e}"
                )

    def save(self) -> None:
        """Saves model weights, training history, and config to the run directory."""
        save_model(self.model, self.run_dir)
        save_history(self.history, self.run_dir)
        save_config(self.config, self.run_dir)
        print(f"Run artifacts saved to: {self.run_dir}")