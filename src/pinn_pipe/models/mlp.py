"""
MLP model for pinn-pipe-flow.

Implements a fully connected feedforward neural network that inherits 
from BasePINN. Takes normalized coordinates (r/R, x/L, Re) as
input and predicts the fluid velocities (u, v) at that point.
"""

import torch
import torch.nn as nn

from pinn_pipe.models.base import BasePINN
from pinn_pipe.utils.config import ModelConfig


class SinActivation(nn.Module):
    """Sinusoidal activation function for PINNs (SIREN-style).
    
    Computes sin(omega * x), providing non-attenuating higher-order derivatives
    suitable for physics-informed neural networks.
    """
    def __init__(self, omega: float = 1.0) -> None:
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


class MLP(BasePINN, nn.Module):
    """Fully connected feedforward neural network for pipe flow velocity prediction.
    
    Architecture: Linear -> [Activation -> Linear] * depth -> Activation -> Linear
    Input size: 3 (r/R, x/L, Re)
    Output size: 2 (u, v)
    """
    
    def __init__(self, config: ModelConfig) -> None:
        """Builds the network layers from the provided model config.

        Args:
            config: ModelConfig containing architecture hyperparameters:
                    hidden_layer_depth, hidden_layer_width, and activation.

        Raises:
            ValueError: If the activation function is not supported.
        """
        super().__init__()
        self.config = config
        
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
            "sin": SinActivation(),
            "sine": SinActivation(),
            "mish": nn.Mish(),
        }
        
        if config.activation not in activations:
            raise ValueError(
                f"Unsupported activation: {config.activation}. " 
                f"Choose from {list(activations.keys())}"
            )
            
        activation = activations[config.activation]
        
        net = []
        net.append(nn.Linear(3, config.hidden_layer_width))
        for _ in range(config.hidden_layer_depth):
            net.append(activation)
            net.append(nn.Linear(config.hidden_layer_width, config.hidden_layer_width))        
        net.append(activation)
        net.append(nn.Linear(config.hidden_layer_width, 2))
        
        self.net = nn.Sequential(*net)
        
    def __repr__(self) -> str:
        """Returns a string summary of the model architecture."""
        return (
            f"MLP("
            f"input=3, "
            f"hidden_layers={self.config.hidden_layer_depth}, "
            f"width={self.config.hidden_layer_width}, "
            f"activation={self.config.activation}, "
            f"output=2)"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the predicted velocity for a batch of input points.

        Args:
            x: Tensor of shape (N, 3) where each row is [r/R, x/L, Re].

        Returns:
            Tensor of shape (N, 2) where columns are [u, v].
        """
        return self.net(x)