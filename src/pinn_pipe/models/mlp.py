"""
MLP model for pinn-pipe-flow.

Implements a fully connected feedforward  neural network that inherits 
from BasePINN. Takas radial position r and maximum velocity u_max as
input and predicts the fluid velocity at that point.
"""

import torch
import torch.nn as nn

from pinn_pipe.models.base import BasePINN
from pinn_pipe.utils.config import ModelConfig


class MLP(BasePINN, nn.Module):
    """Fully connected feedforward neural network for pipe flow velocity prediction.
    
    Architecture: Linear -> [Activation -> Linear] * depth -> Activation -> Linear
    Input size: 2 (r, u_max)
    Output size: 1 (predicted velocity)
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
            "sigmoid": nn.Sigmoid()
        }
        
        if config.activation not in activations:
            raise ValueError(
                f"Unsupported activation: {config.activation}. " 
                f"Choose from {list(activations.keys())}"
            )
            
        activation = activations[config.activation]
        
        net = []
        net.append(nn.Linear(2, config.hidden_layer_width))
        for _ in range(config.hidden_layer_depth):
            net.append(activation)
            net.append(nn.Linear(config.hidden_layer_width, config.hidden_layer_width))        
        net.append(activation)
        net.append(nn.Linear(config.hidden_layer_width, 1))
        
        self.net = nn.Sequential(*net)
        
    def __repr__(self) -> str:
        """Returns a string summary of the model architecture."""
        return(
            f"MLP("
            f"input=2, "
            f"hidden_layers={self.config.hidden_layer_depth}, "
            f"width={self.config.hidden_layer_width}, "
            f"activation={self.config.activation}, "
            f"output=1)"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the predicted velocity for a batch of input points.

        Args:
            x: Tensor of shape (N, 2) where each row is [r, u_max].

        Returns:
            Tensor of shape (N, 1) containing the predicted velocity at each point.
        """
        return self.net(x)