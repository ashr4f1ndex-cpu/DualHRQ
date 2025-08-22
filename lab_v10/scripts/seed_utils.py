"""
Deterministic seeding utilities for reproducible HRM training.

This module provides utilities to set all random seeds across different libraries
to ensure reproducible results in HRM training and backtesting.
"""

import os
import random


def set_all_seeds(seed: int = 1337) -> None:
    """
    Set all random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python random
    random.seed(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_deterministic_config() -> dict:
    """
    Get configuration for deterministic training.
    
    Returns:
        Configuration dict with deterministic settings
    """
    return {
        "seed": 1337,
        "deterministic": True,
        "benchmark": False,
        "num_workers": 0,  # Disable multiprocessing for determinism
    }


def verify_determinism(func, *args, seed: int = 1337, **kwargs) -> bool:
    """
    Verify that a function produces deterministic results.
    
    Args:
        func: Function to test
        *args: Function arguments
        seed: Random seed
        **kwargs: Function keyword arguments
    
    Returns:
        True if function is deterministic, False otherwise
    """
    # First run
    set_all_seeds(seed)
    result1 = func(*args, **kwargs)

    # Second run with same seed
    set_all_seeds(seed)
    result2 = func(*args, **kwargs)

    # Compare results
    try:
        import numpy as np
        if hasattr(result1, 'numpy'):  # PyTorch tensor
            return np.allclose(result1.numpy(), result2.numpy())
        elif isinstance(result1, np.ndarray):
            return np.allclose(result1, result2)
        else:
            return result1 == result2
    except ImportError:
        return result1 == result2
