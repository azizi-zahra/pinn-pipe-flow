"""
Abstract base class for all PINN models in pinn-pipe-flow.

Any new model must inherit from BasePINN and implement the forward method.
This ensures all models are interchangeable throughout the pipeline.
"""

from abc import ABC, abstractmethod

import torch


class BasePINN(ABC):
    """Abstract base class that all PINN models must inherit from."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the predicted velocity for a batch of input points.

        Args:
            x: Tensor of shape (N, 2) where each row is [r, u_max].

        Returns:
            Tensor of shape (N, 1) containing the predicted velocity at each point.
        """
        pass