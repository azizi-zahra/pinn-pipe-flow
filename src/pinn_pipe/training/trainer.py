"""
Trainer class for pinn-pipe-flow.

Manages the training loop, loss logging, progress printing,
and saving of all run artifacts.
"""

import time
from typing import Optional

import torch

from pinn_pipe.models import BasePINN
from pinn_pipe.training.losses import total_loss
from pinn_pipe.training.sampler import sample_bc, sample_interior
from pinn_pipe.training.scheduler import build_lr_scheduler
from pinn_pipe.utils import Config, save_config, save_history, save_model


class Trainer:
    """Manages the PINN training loop for pipe flow.

    Handles optimizer setup, LR scheduling, hybrid optimization, sampling,
    loss computation, history logging, and saving of all run artifacts.
    """

    def __init__(self, model: BasePINN, config: Config, run_dir: str, device: torch.device) -> None:
        """Initializes the Trainer with model, config, and run directory.

        Args:
            model: A BasePINN instance to train.
            config: Fully populated Config object for this run.
            run_dir: Path to the timestamped results directory for this run.
            device: torch.device to use for training.

        Raises:
            ValueError: If the optimizer name in config is not supported.
        """
        self.model = model
        self.config = config
        self.run_dir = run_dir
        self.history = []
        self.device = device

        self.is_hybrid = config.training.optimizer in ("hybrid", "hybrid_adam_lbfgs")
        self.switch_epoch = (
            config.training.hybrid_switch_epoch
            if config.training.hybrid_switch_epoch is not None
            else int(config.training.epochs * 0.8)
        )

        if self.is_hybrid:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.training.learning_rate,
            )
            self.scheduler = build_lr_scheduler(
                self.optimizer,
                config.training.lr_scheduler,
                self.switch_epoch,
                config.training.lr_scheduler_params,
            )
        else:
            supported_optimizers = {
                "adam": torch.optim.Adam,
                "sgd": torch.optim.SGD,
                "lbfgs": torch.optim.LBFGS,
            }

            if config.training.optimizer not in supported_optimizers:
                raise ValueError(
                    f"Unsupported optimizer: {config.training.optimizer}. "
                    f"Choose from {list(supported_optimizers.keys()) + ['hybrid']}"
                )

            self.optimizer = supported_optimizers[config.training.optimizer](
                self.model.parameters(),
                lr=config.training.learning_rate,
            )
            self.scheduler = build_lr_scheduler(
                self.optimizer,
                config.training.lr_scheduler,
                config.training.epochs,
                config.training.lr_scheduler_params,
            )

    def train(self) -> None:
        """Runs the full training loop for the configured number of epochs.

        At each epoch:
        - Handles hybrid optimizer switch from Adam to L-BFGS if configured
        - Samples fresh interior and BC collocation points
        - Computes total loss and all individual loss terms
        - Backpropagates and steps the optimizer and LR scheduler
        - Logs loss values and learning rate to history
        - Prints progress every 500 epochs
        """
        start_time = time.time()
        self.model.train()

        for epoch in range(1, self.config.training.epochs + 1):
            # Hybrid optimizer transition: switch from Adam to L-BFGS
            if self.is_hybrid and epoch == self.switch_epoch + 1:
                lbfgs_lr = (
                    self.config.training.lbfgs_learning_rate
                    if self.config.training.lbfgs_learning_rate is not None
                    else 1.0
                )
                print(
                    f"\n>>> [Hybrid Optimizer] Switching from Adam to L-BFGS at epoch {epoch} "
                    f"(lr={lbfgs_lr}, remaining epochs: {self.config.training.epochs - self.switch_epoch})"
                )
                self.optimizer = torch.optim.LBFGS(
                    self.model.parameters(),
                    lr=lbfgs_lr,
                    max_iter=20,
                    history_size=50,
                )
                self.scheduler = None  # L-BFGS relies on internal line search

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
            
            # move tensors to device
            r = r.to(self.device)
            u_max = u_max.to(self.device)
            r_wall = r_wall.to(self.device)
            r_sym = r_sym.to(self.device)
            u_max_bc = u_max_bc.to(self.device)

            if isinstance(self.optimizer, torch.optim.LBFGS):
                def closure():
                    self.optimizer.zero_grad()
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
                    losses["loss_total_tensor"].backward()
                    return losses["loss_total_tensor"]
                self.optimizer.step(closure)
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
            else:    
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

            # Step learning rate scheduler if configured
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(losses["loss_total"])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # log history
            self.history.append({
                "epoch": epoch,
                "loss_total": losses["loss_total"],
                "loss_physics": losses["loss_physics"],
                "loss_bc_wall": losses["loss_bc_wall"],
                "loss_bc_symmetry": losses["loss_bc_symmetry"],
                "lr": current_lr,
            })

            # print progress every 500 epochs
            if epoch % 500 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:>5}/{self.config.training.epochs} | "
                    f"Total: {losses['loss_total']:.4e} | "
                    f"Physics: {losses['loss_physics']:.4e} | "
                    f"BC Wall: {losses['loss_bc_wall']:.4e} | "
                    f"BC Sym: {losses['loss_bc_symmetry']:.4e} | "
                    f"LR: {current_lr:.2e}"
                )
                
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")

    def save(self) -> None:
        """Saves model weights, training history, and config to the run directory."""
        save_model(self.model, self.run_dir)
        save_history(self.history, self.run_dir)
        save_config(self.config, self.run_dir)
        print(f"Run artifacts saved to: {self.run_dir}")