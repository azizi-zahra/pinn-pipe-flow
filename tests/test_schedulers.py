"""
Tests for training/scheduler.py and hybrid optimizer functionality in Trainer.
"""

import pytest
import torch
import torch.nn as nn

from pinn_pipe.models import MLP
from pinn_pipe.training import Trainer, build_lr_scheduler
from pinn_pipe.utils import (
    Config,
    ModelConfig,
    PhysicsConfig,
    RunConfig,
    SamplingConfig,
    TrainingConfig,
)


@pytest.fixture
def dummy_model():
    """Simple linear model for optimizer and scheduler tests."""
    return nn.Linear(2, 1)


@pytest.fixture
def dummy_config():
    """Returns a valid Config object for trainer testing."""
    return Config(
        physics=PhysicsConfig(R=1.0, mu=1.0, u_max_min=0.5, u_max_max=2.0),
        model=ModelConfig(
            type="mlp",
            hidden_layer_depth=2,
            hidden_layer_width=16,
            activation="tanh",
        ),
        sampling=SamplingConfig(num_interior_points=50),
        training=TrainingConfig(
            epochs=10,
            learning_rate=0.001,
            optimizer="adam",
            loss_weight_physics=1.0,
            loss_weight_bc_wall=1.0,
            loss_weight_bc_symmetry=1.0,
        ),
        run=RunConfig(seed=42, dtype="float64"),
    )


def test_build_lr_scheduler_none(dummy_model):
    """Scheduler factory should return None when scheduler_type is None."""
    opt = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    sched = build_lr_scheduler(opt, None, 100)
    assert sched is None


def test_build_lr_scheduler_cosine(dummy_model):
    """Cosine scheduler should be instantiated and step learning rate downwards."""
    opt = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    sched = build_lr_scheduler(opt, "cosine", 100, {"eta_min": 1e-6})
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    initial_lr = opt.param_groups[0]["lr"]
    sched.step()
    stepped_lr = opt.param_groups[0]["lr"]
    assert stepped_lr <= initial_lr


def test_build_lr_scheduler_step(dummy_model):
    """StepLR scheduler should be instantiated properly."""
    opt = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    sched = build_lr_scheduler(opt, "step", 100, {"step_size": 25, "gamma": 0.5})
    assert isinstance(sched, torch.optim.lr_scheduler.StepLR)


def test_build_lr_scheduler_plateau(dummy_model):
    """ReduceLROnPlateau scheduler should be instantiated properly."""
    opt = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    sched = build_lr_scheduler(opt, "plateau", 100, {"patience": 10})
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_build_lr_scheduler_invalid(dummy_model):
    """Invalid scheduler name should raise ValueError."""
    opt = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        build_lr_scheduler(opt, "unsupported_scheduler_name", 100)


def test_trainer_with_lr_scheduler(dummy_config, tmp_path):
    """Trainer should successfully initialize and run with a cosine scheduler."""
    dummy_config.training.epochs = 5
    dummy_config.training.lr_scheduler = "cosine"
    dummy_config.training.lr_scheduler_params = {"eta_min": 1e-5}

    model = MLP(dummy_config.model)
    trainer = Trainer(model, dummy_config, str(tmp_path), torch.device("cpu"))
    assert trainer.scheduler is not None
    trainer.train()

    assert len(trainer.history) == 5
    assert "lr" in trainer.history[0]


def test_trainer_with_hybrid_optimizer(dummy_config, tmp_path):
    """Trainer should successfully switch from Adam to L-BFGS at switch_epoch."""
    dummy_config.training.epochs = 4
    dummy_config.training.optimizer = "hybrid"
    dummy_config.training.hybrid_switch_epoch = 2
    dummy_config.training.lbfgs_learning_rate = 0.5

    model = MLP(dummy_config.model)
    trainer = Trainer(model, dummy_config, str(tmp_path), torch.device("cpu"))

    assert trainer.is_hybrid
    assert isinstance(trainer.optimizer, torch.optim.Adam)

    trainer.train()

    # After training past switch_epoch (epoch 2), optimizer should be L-BFGS
    assert isinstance(trainer.optimizer, torch.optim.LBFGS)
    assert len(trainer.history) == 4
