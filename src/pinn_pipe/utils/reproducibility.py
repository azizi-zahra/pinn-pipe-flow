"""
Reproducibility utilities.

Sets all random seeds to ensure experiments are fully reproducible
on the same machine and environment.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Sets random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value. Use the value from the run config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True