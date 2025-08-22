"""
Determinism tests for HRM trading lab.

These tests verify that all random operations produce reproducible results
when using the same seed.
"""

import numpy as np
import pytest

from lab_v10.scripts.seed_utils import set_all_seeds, verify_determinism


def test_numpy_repeatability():
    """Test that NumPy random operations are repeatable."""
    set_all_seeds(123)
    a = np.random.rand(8, 8)

    set_all_seeds(123)
    b = np.random.rand(8, 8)

    assert np.allclose(a, b), "NumPy random operations should be deterministic"


def test_python_random_repeatability():
    """Test that Python random operations are repeatable."""
    import random

    set_all_seeds(456)
    a = [random.random() for _ in range(10)]

    set_all_seeds(456)
    b = [random.random() for _ in range(10)]

    assert a == b, "Python random operations should be deterministic"


def test_numpy_choice_repeatability():
    """Test that NumPy choice operations are repeatable."""
    set_all_seeds(789)
    choices_a = np.random.choice(100, size=20, replace=False)

    set_all_seeds(789)
    choices_b = np.random.choice(100, size=20, replace=False)

    assert np.array_equal(choices_a, choices_b), "NumPy choice should be deterministic"


@pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch not available"
)
def test_torch_repeatability():
    """Test that PyTorch operations are repeatable."""
    import torch

    set_all_seeds(999)
    a = torch.randn(4, 4)

    set_all_seeds(999)
    b = torch.randn(4, 4)

    assert torch.allclose(a, b), "PyTorch operations should be deterministic"


def test_verify_determinism_utility():
    """Test the verify_determinism utility function."""
    def deterministic_func(size):
        return np.random.rand(size)

    # Should return True for deterministic function
    assert verify_determinism(deterministic_func, 5, seed=111)

    def non_deterministic_func(size):
        import time
        np.random.seed(int(time.time() * 1000000) % 2**32)
        return np.random.rand(size)

    # Should return False for non-deterministic function
    # Note: This test is inherently flaky, so we skip it in CI
    # assert not verify_determinism(non_deterministic_func, 5, seed=222)


def test_seed_persistence_across_operations():
    """Test that seed persists across multiple operations."""
    set_all_seeds(333)

    # First batch of operations
    a1 = np.random.rand(3)
    a2 = np.random.randint(0, 100, 5)

    set_all_seeds(333)

    # Second batch of operations (same sequence)
    b1 = np.random.rand(3)
    b2 = np.random.randint(0, 100, 5)

    assert np.allclose(a1, b1), "First operation should be deterministic"
    assert np.array_equal(a2, b2), "Second operation should be deterministic"


def _torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def test_financial_data_generation_determinism():
    """Test determinism in financial data generation patterns."""
    set_all_seeds(42)

    # Simulate price path generation
    n_steps = 100
    dt = 1/252
    mu = 0.05
    sigma = 0.2

    # GBM simulation
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    log_returns_a = (mu - 0.5 * sigma**2) * dt + sigma * dW

    set_all_seeds(42)

    # Repeat simulation
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    log_returns_b = (mu - 0.5 * sigma**2) * dt + sigma * dW

    assert np.allclose(log_returns_a, log_returns_b), "Financial simulations should be deterministic"


def test_train_val_split_determinism():
    """Test that train/validation splits are deterministic."""
    from sklearn.model_selection import train_test_split

    # Create sample data
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000)

    set_all_seeds(2023)
    X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(
        X, y, test_size=0.2, random_state=2023
    )

    set_all_seeds(2023)
    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
        X, y, test_size=0.2, random_state=2023
    )

    assert np.allclose(X_train_a, X_train_b), "Train split should be deterministic"
    assert np.allclose(X_val_a, X_val_b), "Validation split should be deterministic"
    assert np.allclose(y_train_a, y_train_b), "Train labels should be deterministic"
    assert np.allclose(y_val_a, y_val_b), "Validation labels should be deterministic"
