"""
Derivative utilities for physics residual computation.

Provides short helper functions for first and second order derivatives
using torch.autograd. Used in physics/pipe_flow.py to keep the
residual code clean and readable.
"""

import torch


def grad(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """Computes the first derivative of output with respect to input.

    Args:
        output: Model output tensor, typically predicted velocity u.
        input: Input tensor to differentiate with respect to, typically r.

    Returns:
        Tensor of the same shape as output containing du/dr.
    """
    return torch.autograd.grad(
        output, input,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]
    
      
def grad2(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """Computes the second derivative of output with respect to input.

    Args:
        output: Model output tensor, typically predicted velocity u.
        input: Input tensor to differentiate with respect to, typically r.

    Returns:
        Tensor of the same shape as output containing d²u/dr².
    """
    first = grad(output, input)
    return grad(first, input)