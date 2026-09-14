"""
Device utilities for pinn-pipe-flow.

Handles device selection between CPU and GPU
"""

import torch


def get_device(device_str: str = None) -> torch.device:
    """Returns torch device to use for training.
    
    Args:
        device_str: Device string, either 'cuda' or 'cpu'.
                    If None, defaults to cpu.
    
    Returns:
        torch.device.object
        
    Raises:
        ValueError: If device_str is not 'cuda' or 'cpu'
    """
    if device_str is None:
        return torch.device("cpu")
    
    if device_str not in ["cuda", "cpu"]:
        raise ValueError(f"Invalid device: {device_str}. Choose 'cuda' or 'cpu'.")

    if device_str == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available on this machine.")

    return torch.device(device_str)