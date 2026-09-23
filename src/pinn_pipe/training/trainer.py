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
from pinn_pipe.training.sampler import (
    sample_bc_inlet,
    sample_bc_symmetry,
    sample_bc_wall,
    sample_interior,
)
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
        - Samples fresh interior and BC collocation points (fixed during LBFGS phase)
        - Computes total loss and all individual loss terms
        - Backpropagates and steps the optimizer and LR scheduler
        - Logs loss values and learning rate to history
        - Prints progress every 500 epochs
        """
        start_time = time.time()
        self.model.train()

        # fixed points for LBFGS phase -- populated at switch epoch
        r_fixed: Optional[torch.Tensor] = None
        x_fixed: Optional[torch.Tensor] = None
        Re_fixed: Optional[torch.Tensor] = None
        r_wall_fixed: Optional[torch.Tensor] = None
        x_bc_wall_fixed: Optional[torch.Tensor] = None
        Re_bc_wall_fixed: Optional[torch.Tensor] = None
        r_sym_fixed: Optional[torch.Tensor] = None
        x_bc_sym_fixed: Optional[torch.Tensor] = None
        Re_bc_sym_fixed: Optional[torch.Tensor] = None
        r_inlet_fixed: Optional[torch.Tensor] = None
        x_inlet_fixed: Optional[torch.Tensor] = None
        Re_bc_inlet_fixed: Optional[torch.Tensor] = None

        for epoch in range(1, self.config.training.epochs + 1):

            # Hybrid optimizer transition: switch from Adam to L-BFGS
            if self.is_hybrid and epoch == self.switch_epoch + 1:
                lbfgs_lr = (
                    self.config.training.lbfgs_learning_rate
                    if self.config.training.lbfgs_learning_rate is not None
                    else 0.1
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
                self.scheduler = None

                # fix sampling points for entire LBFGS phase
                r_fixed, x_fixed, Re_fixed = sample_interior(
                    n=self.config.sampling.num_interior_points,
                    config=self.config.physics,
                )
                r_wall_fixed, x_bc_wall_fixed, Re_bc_wall_fixed = sample_bc_wall(
                    n=self.config.sampling.num_bc_wall_points,
                    config=self.config.physics,
                )
                r_sym_fixed, x_bc_sym_fixed, Re_bc_sym_fixed = sample_bc_symmetry(
                    n=self.config.sampling.num_bc_symmetry_points,
                    config=self.config.physics,
                )
                r_inlet_fixed, x_inlet_fixed, Re_bc_inlet_fixed = sample_bc_inlet(
                    n=self.config.sampling.num_bc_inlet_points,
                    config=self.config.physics,
                )

                r_fixed = r_fixed.to(self.device)
                x_fixed = x_fixed.to(self.device)
                Re_fixed = Re_fixed.to(self.device)
                r_wall_fixed = r_wall_fixed.to(self.device)
                x_bc_wall_fixed = x_bc_wall_fixed.to(self.device)
                Re_bc_wall_fixed = Re_bc_wall_fixed.to(self.device)
                r_sym_fixed = r_sym_fixed.to(self.device)
                x_bc_sym_fixed = x_bc_sym_fixed.to(self.device)
                Re_bc_sym_fixed = Re_bc_sym_fixed.to(self.device)
                r_inlet_fixed = r_inlet_fixed.to(self.device)
                x_inlet_fixed = x_inlet_fixed.to(self.device)
                Re_bc_inlet_fixed = Re_bc_inlet_fixed.to(self.device)

            # use fixed points during LBFGS phase, fresh points during Adam phase
            if self.is_hybrid and epoch > self.switch_epoch:
                r = r_fixed
                x = x_fixed
                Re = Re_fixed
                r_wall = r_wall_fixed
                x_bc_wall = x_bc_wall_fixed
                Re_bc_wall = Re_bc_wall_fixed
                r_sym = r_sym_fixed
                x_bc_sym = x_bc_sym_fixed
                Re_bc_sym = Re_bc_sym_fixed
                r_inlet = r_inlet_fixed
                x_inlet = x_inlet_fixed
                Re_bc_inlet = Re_bc_inlet_fixed
            else:
                r, x, Re = sample_interior(
                    n=self.config.sampling.num_interior_points,
                    config=self.config.physics,
                )
                r_wall, x_bc_wall, Re_bc_wall = sample_bc_wall(
                    n=self.config.sampling.num_bc_wall_points,
                    config=self.config.physics,
                )
                r_sym, x_bc_sym, Re_bc_sym = sample_bc_symmetry(
                    n=self.config.sampling.num_bc_symmetry_points,
                    config=self.config.physics,
                )
                r_inlet, x_inlet, Re_bc_inlet = sample_bc_inlet(
                    n=self.config.sampling.num_bc_inlet_points,
                    config=self.config.physics,
                )

                r = r.to(self.device)
                x = x.to(self.device)
                Re = Re.to(self.device)
                r_wall = r_wall.to(self.device)
                x_bc_wall = x_bc_wall.to(self.device)
                Re_bc_wall = Re_bc_wall.to(self.device)
                r_sym = r_sym.to(self.device)
                x_bc_sym = x_bc_sym.to(self.device)
                Re_bc_sym = Re_bc_sym.to(self.device)
                r_inlet = r_inlet.to(self.device)
                x_inlet = x_inlet.to(self.device)
                Re_bc_inlet = Re_bc_inlet.to(self.device)

            if isinstance(self.optimizer, torch.optim.LBFGS):
                def closure():
                    self.optimizer.zero_grad()
                    losses = total_loss(
                        model=self.model,
                        r=r,
                        x=x,
                        Re=Re,
                        r_wall=r_wall,
                        x_bc_wall=x_bc_wall,
                        Re_bc_wall=Re_bc_wall,
                        r_sym=r_sym,
                        x_bc_sym=x_bc_sym,
                        Re_bc_sym=Re_bc_sym,
                        r_inlet=r_inlet,
                        x_inlet=x_inlet,
                        Re_bc_inlet=Re_bc_inlet,
                        physics_config=self.config.physics,
                        training_config=self.config.training,
                    )
                    losses["loss_total_tensor"].backward()
                    return losses["loss_total_tensor"]

                self.optimizer.step(closure)
                losses = total_loss(
                    model=self.model,
                    r=r,
                    x=x,
                    Re=Re,
                    r_wall=r_wall,
                    x_bc_wall=x_bc_wall,
                    Re_bc_wall=Re_bc_wall,
                    r_sym=r_sym,
                    x_bc_sym=x_bc_sym,
                    Re_bc_sym=Re_bc_sym,
                    r_inlet=r_inlet,
                    x_inlet=x_inlet,
                    Re_bc_inlet=Re_bc_inlet,
                    physics_config=self.config.physics,
                    training_config=self.config.training,
                )
            else:
                losses = total_loss(
                    model=self.model,
                    r=r,
                    x=x,
                    Re=Re,
                    r_wall=r_wall,
                    x_bc_wall=x_bc_wall,
                    Re_bc_wall=Re_bc_wall,
                    r_sym=r_sym,
                    x_bc_sym=x_bc_sym,
                    Re_bc_sym=Re_bc_sym,
                    r_inlet=r_inlet,
                    x_inlet=x_inlet,
                    Re_bc_inlet=Re_bc_inlet,
                    physics_config=self.config.physics,
                    training_config=self.config.training,
                )
                self.optimizer.zero_grad()
                losses["loss_total_tensor"].backward()
                self.optimizer.step()

            # step learning rate scheduler if configured
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
                "loss_momentum": losses["loss_momentum"],
                "loss_continuity": losses["loss_continuity"],
                "loss_bc_wall": losses["loss_bc_wall"],
                "loss_bc_wall_v": losses["loss_bc_wall_v"],
                "loss_bc_symmetry": losses["loss_bc_symmetry"],
                "loss_bc_inlet": losses["loss_bc_inlet"],
                "lr": current_lr,
            })

            # print progress every 500 epochs
            if epoch % 500 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:>5}/{self.config.training.epochs} | "
                    f"Total: {losses['loss_total']:.4e} | "
                    f"Momentum: {losses['loss_momentum']:.4e} | "
                    f"Continuity: {losses['loss_continuity']:.4e} | "
                    f"Inlet: {losses['loss_bc_inlet']:.4e} | "
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