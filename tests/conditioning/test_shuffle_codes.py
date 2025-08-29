"""
test_shuffle_codes.py
=====================

TDD tests for DRQ-002: Shuffle Test for Label Dependency Validation
These tests MUST be written first and will initially FAIL.
Implementation should make these tests pass.

CRITICAL: Model performance should drop >50% on shuffled labels.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Callable, Any
import time

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from tools.check_no_static_ids import ShuffleTest
except ImportError:
    # These will fail initially - that's expected in TDD
    pass


class TestShuffleValidation:
    """Tests for shuffle test validation of model dependency on labels."""
    
    def test_shuffle_degradation_requirement(self):
        """Test that model performance drops >50% on shuffled labels."""
        # Mock model function that should depend on labels
        def good_model_func(X, y, train_idx, test_idx):
            """Model that legitimately depends on labels - should show degradation."""
            # Simple correlation-based model
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]
            
            # Calculate feature-target correlations on training data
            correlations = []
            for i in range(X_train.shape[1]):
                corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
                correlations.append(0 if np.isnan(corr) else corr)
            
            # Predict using correlations
            predictions = np.sum(X_test * correlations, axis=1)
            actual = y[test_idx]
            
            # Return correlation as performance metric
            return np.corrcoef(predictions, actual)[0, 1] if len(actual) > 1 else 0
        
        # Generate synthetic data
        n_samples = 1000
        n_features = 10
        np.random.seed(42)
        
        X = np.random.randn(n_samples, n_features)
        # y depends on X features (legitimate relationship)
        y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.1
        
        train_idx = np.arange(700)
        test_idx = np.arange(700, 1000)
        
        # This will fail initially until ShuffleTest is implemented
        shuffle_tester = ShuffleTest(n_shuffles=10)
        results = shuffle_tester.test_label_shuffling(good_model_func, X, y, train_idx, test_idx)
        
        # Performance should drop significantly when labels are shuffled
        assert results['degradation_sufficient'], \
            f"Performance should drop >50%, got {results['relative_performance_drop']:.1%}"
        assert results['relative_performance_drop'] > 0.5, \
            f"Relative drop should be >50%, got {results['relative_performance_drop']:.1%}"
        assert results['original_score'] > results['mean_shuffled_score'], \
            "Original model should outperform shuffled labels"
    
    def test_detects_memorization_model(self):
        """Test that shuffle test detects memorization-based models."""
        # Mock model that memorizes training examples (bad model)
        training_memo = {}
        
        def memorization_model_func(X, y, train_idx, test_idx):
            """Model that memorizes training data - should NOT show degradation."""
            nonlocal training_memo
            
            # "Train" by memorizing all training examples
            for i, idx in enumerate(train_idx):
                key = tuple(X[idx])  # Use features as key
                training_memo[key] = y[idx]
            
            # "Predict" by looking up memorized values
            predictions = []
            for idx in test_idx:
                key = tuple(X[idx])
                if key in training_memo:
                    predictions.append(training_memo[key])
                else:
                    predictions.append(np.mean(y[train_idx]))  # Default to mean
            
            predictions = np.array(predictions)
            actual = y[test_idx]
            
            return np.corrcoef(predictions, actual)[0, 1] if len(actual) > 1 else 0
        
        # Create data where test set has some overlap with training set (enabling memorization)
        n_samples = 500
        n_features = 5
        np.random.seed(123)
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Make some test examples identical to training examples
        train_idx = np.arange(300)
        test_idx = np.arange(300, 500)
        
        # Copy some training examples to test set (enables memorization)
        for i in range(50):
            test_idx_pos = 300 + i
            train_idx_pos = i
            X[test_idx_pos] = X[train_idx_pos].copy()
            y[test_idx_pos] = y[train_idx_pos].copy()
        
        shuffle_tester = ShuffleTest(n_shuffles=10)
        results = shuffle_tester.test_label_shuffling(memorization_model_func, X, y, train_idx, test_idx)
        
        # Memorization model should NOT show sufficient degradation
        # (This indicates a problem with the model)
        assert not results['degradation_sufficient'], \
            f"Memorization model should not show sufficient degradation, got {results['relative_performance_drop']:.1%}"
        assert results['relative_performance_drop'] < 0.3, \
            f"Memorization should show <30% drop, got {results['relative_performance_drop']:.1%}"
    
    def test_shuffle_preserves_distribution(self):
        """Test that label shuffling preserves the label distribution."""
        n_samples = 1000
        
        # Create labels with specific distribution
        original_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
        train_idx = np.arange(700)
        
        shuffle_tester = ShuffleTest(n_shuffles=5)
        
        # Check that shuffled labels maintain distribution
        for _ in range(5):
            shuffled_labels = original_labels.copy()
            shuffled_labels[train_idx] = np.random.permutation(shuffled_labels[train_idx])
            
            # Distribution should be preserved in training set
            original_dist = np.bincount(original_labels[train_idx]) / len(train_idx)
            shuffled_dist = np.bincount(shuffled_labels[train_idx]) / len(train_idx)
            
            # Distributions should be identical (same counts, different order)
            np.testing.assert_array_equal(np.sort(original_dist), np.sort(shuffled_dist))
    
    def test_multiple_shuffle_consistency(self):
        """Test that multiple shuffle runs produce consistent results."""
        def consistent_model_func(X, y, train_idx, test_idx):
            """Simple linear model for consistency testing."""
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]
            
            # Simple linear regression (X @ beta = y)
            if X_train.shape[0] > X_train.shape[1]:  # Avoid singular matrix
                beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                predictions = X_test @ beta
                actual = y[test_idx]
                return np.corrcoef(predictions, actual)[0, 1] if len(actual) > 1 else 0
            else:
                return 0
        
        # Generate consistent test data
        n_samples = 800
        n_features = 5
        np.random.seed(456)
        
        X = np.random.randn(n_samples, n_features)
        y = X @ np.random.randn(n_features) + np.random.randn(n_samples) * 0.1
        
        train_idx = np.arange(600)
        test_idx = np.arange(600, 800)
        
        # Run shuffle test multiple times
        shuffle_tester = ShuffleTest(n_shuffles=10)
        
        results_1 = shuffle_tester.test_label_shuffling(consistent_model_func, X, y, train_idx, test_idx)
        results_2 = shuffle_tester.test_label_shuffling(consistent_model_func, X, y, train_idx, test_idx)
        
        # Results should be reasonably consistent (within 20% relative difference)
        relative_diff = abs(results_1['relative_performance_drop'] - results_2['relative_performance_drop'])
        relative_diff /= max(results_1['relative_performance_drop'], results_2['relative_performance_drop'])
        
        assert relative_diff < 0.2, \
            f"Shuffle test results should be consistent, got {relative_diff:.1%} difference"
    
    def test_edge_case_handling(self):
        """Test shuffle test handles edge cases gracefully."""
        # Test with very small dataset
        def simple_model_func(X, y, train_idx, test_idx):
            if len(train_idx) == 0 or len(test_idx) == 0:
                return 0
            return np.corrcoef(y[train_idx], y[test_idx])[0, 1] if len(test_idx) > 1 else 0
        
        # Tiny dataset
        X_small = np.random.randn(10, 3)
        y_small = np.random.randn(10)
        train_idx_small = np.arange(7)
        test_idx_small = np.arange(7, 10)
        
        shuffle_tester = ShuffleTest(n_shuffles=3)
        results = shuffle_tester.test_label_shuffling(simple_model_func, X_small, y_small, 
                                                     train_idx_small, test_idx_small)
        
        # Should handle small dataset without crashing
        assert 'original_score' in results, "Should return results for small dataset"
        assert 'mean_shuffled_score' in results, "Should return shuffled results"
        
        # Test with empty test set
        empty_test_idx = np.array([], dtype=int)
        try:
            results_empty = shuffle_tester.test_label_shuffling(simple_model_func, X_small, y_small,
                                                               train_idx_small, empty_test_idx)
            # Should handle gracefully, not crash
        except Exception as e:
            pytest.fail(f"Should handle empty test set gracefully: {e}")
    
    def test_performance_timing(self):
        """Test that shuffle test completes within reasonable time."""
        def slow_model_func(X, y, train_idx, test_idx):
            """Artificially slow model for timing test."""
            time.sleep(0.01)  # 10ms delay per call
            return np.mean(y[test_idx]) if len(test_idx) > 0 else 0
        
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        y = np.random.randn(n_samples)
        train_idx = np.arange(150)
        test_idx = np.arange(150, 200)
        
        shuffle_tester = ShuffleTest(n_shuffles=5)  # Small number for timing
        
        start_time = time.time()
        results = shuffle_tester.test_label_shuffling(slow_model_func, X, y, train_idx, test_idx)
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time (5 shuffles × 10ms + overhead < 5 seconds)
        assert elapsed_time < 5.0, \
            f"Shuffle test should complete quickly, took {elapsed_time:.2f}s"
    
    def test_statistical_significance(self):
        """Test that shuffle test can detect statistical significance."""
        def strong_model_func(X, y, train_idx, test_idx):
            """Model with strong legitimate signal."""
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]
            
            # Strong linear relationship
            if len(X_train) > 0:
                mean_coeff = np.mean([np.corrcoef(X_train[:, i], y_train)[0, 1] 
                                     for i in range(X_train.shape[1]) 
                                     if not np.isnan(np.corrcoef(X_train[:, i], y_train)[0, 1])])
                predictions = np.mean(X_test, axis=1) * mean_coeff
            else:
                predictions = np.zeros(len(test_idx))
            
            actual = y[test_idx]
            return np.corrcoef(predictions, actual)[0, 1] if len(actual) > 1 else 0
        
        # Create data with strong signal
        n_samples = 1000
        X = np.random.randn(n_samples, 8)
        y = np.sum(X, axis=1) + np.random.randn(n_samples) * 0.05  # Strong signal, low noise
        
        train_idx = np.arange(700)
        test_idx = np.arange(700, 1000)
        
        shuffle_tester = ShuffleTest(n_shuffles=20)  # More shuffles for statistical power
        results = shuffle_tester.test_label_shuffling(strong_model_func, X, y, train_idx, test_idx)
        
        # Strong model should show very significant degradation
        assert results['relative_performance_drop'] > 0.7, \
            f"Strong model should show >70% degradation, got {results['relative_performance_drop']:.1%}"
        
        # Performance difference should be statistically significant
        performance_diff = results['original_score'] - results['mean_shuffled_score']
        shuffled_std = np.std(results['shuffled_scores']) if len(results['shuffled_scores']) > 1 else 1
        
        # Simple z-test approximation
        z_score = performance_diff / (shuffled_std + 1e-8)
        assert abs(z_score) > 2, \
            f"Performance difference should be statistically significant, z={z_score:.2f}"


# This will run when pytest is called and should initially FAIL
# Implementation should make these tests pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])