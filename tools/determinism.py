#!/usr/bin/env python3
"""
determinism.py - Determinism Validation Framework
=================================================

Framework for ensuring bit-exact reproducibility across runs.
"""

import random
import numpy as np
import torch
import os
from typing import Dict, Any, Callable


def set_all_seeds(seed: int = 42):
    """Set all random seeds for deterministic behavior."""
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)


def validate_determinism(func: Callable, test_data: Dict[str, Any], n_runs: int = 3) -> Dict[str, Any]:
    """Validate that a function produces deterministic results."""
    results = []
    
    for i in range(n_runs):
        set_all_seeds(42)  # Same seed each time
        result = func(test_data)
        results.append(result)
    
    # Check if all results are identical
    first_result = results[0]
    is_deterministic = all(
        torch.allclose(result, first_result) if isinstance(result, torch.Tensor) 
        else result == first_result
        for result in results[1:]
    )
    
    return {
        'is_deterministic': is_deterministic,
        'n_runs': n_runs,
        'results': results
    }


class ReproducibilityValidator:
    """Validator for reproducibility testing."""
    
    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance
    
    def validate_reproducibility(self, model_func: Callable, test_data: Dict[str, Any], 
                               n_runs: int = 3) -> Dict[str, Any]:
        """Validate reproducibility of model function."""
        results = []
        
        for i in range(n_runs):
            set_all_seeds(42)
            result = model_func(test_data)
            results.append(result)
        
        # Check consistency
        first_result = results[0]
        is_reproducible = True
        
        for result in results[1:]:
            if isinstance(result, torch.Tensor):
                if not torch.allclose(result, first_result, atol=self.tolerance):
                    is_reproducible = False
                    break
            elif isinstance(result, (int, float)):
                if abs(result - first_result) > self.tolerance:
                    is_reproducible = False
                    break
            elif result != first_result:
                is_reproducible = False
                break
        
        return {
            'is_reproducible': is_reproducible,
            'n_runs': n_runs,
            'consistency_check': 'passed' if is_reproducible else 'failed',
            'tolerance': self.tolerance
        }


if __name__ == '__main__':
    # Basic validation
    def test_func(data):
        return torch.randn(5).sum().item()
    
    validator = ReproducibilityValidator()
    result = validator.validate_reproducibility(test_func, {})
    print(f"Determinism validation: {result}")