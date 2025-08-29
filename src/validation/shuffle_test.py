"""
shuffle_test.py
==============

Implementation of shuffle test for label dependency validation.

This module tests whether a model legitimately depends on labels by
shuffling the labels and measuring performance degradation.

A good model should show >50% performance degradation when labels are shuffled.
Models that don't degrade sufficiently may be memorizing or using leaked information.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


@dataclass
class ShuffleTestResult:
    """Container for shuffle test results."""
    original_score: float
    shuffled_scores: List[float]
    mean_shuffled_score: float
    std_shuffled_score: float
    relative_performance_drop: float
    absolute_performance_drop: float
    degradation_sufficient: bool
    p_value: float
    z_score: float
    test_passed: bool


class ShuffleTestValidator:
    """
    Shuffle test validator for detecting label dependency.
    
    The shuffle test works by:
    1. Training/evaluating model on original data -> baseline performance
    2. Shuffling training labels randomly while keeping features unchanged
    3. Training/evaluating model on shuffled data -> degraded performance
    4. Comparing performance drop - should be >50% for legitimate models
    """
    
    def __init__(self, 
                 n_shuffles: int = 10,
                 degradation_threshold: float = 0.5,
                 significance_level: float = 0.05,
                 random_state: Optional[int] = None):
        """
        Initialize shuffle test validator.
        
        Args:
            n_shuffles: Number of shuffle iterations to run
            degradation_threshold: Minimum required performance drop (0.5 = 50%)
            significance_level: Statistical significance level for tests
            random_state: Random seed for reproducibility
        """
        self.n_shuffles = n_shuffles
        self.degradation_threshold = degradation_threshold
        self.significance_level = significance_level
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def run_shuffle_test(self, 
                        model_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
                        X: np.ndarray,
                        y: np.ndarray,
                        train_idx: np.ndarray,
                        test_idx: np.ndarray) -> Dict[str, Any]:
        """
        Run shuffle test on model function.
        
        Args:
            model_func: Function that trains/evaluates model
                       Signature: model_func(X, y, train_idx, test_idx) -> performance_score
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples] 
            train_idx: Training sample indices
            test_idx: Test sample indices
            
        Returns:
            Dictionary with shuffle test results
        """
        # Validate inputs
        if len(X) != len(y):
            raise ValueError("X and y must have same number of samples")
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("Train and test indices cannot be empty")
        if not set(train_idx).isdisjoint(set(test_idx)):
            raise ValueError("Train and test indices must be disjoint")
            
        # Step 1: Evaluate model on original data
        try:
            original_score = model_func(X, y, train_idx, test_idx)
        except Exception as e:
            return {
                'error': f"Original model evaluation failed: {str(e)}",
                'test_passed': False,
                'degradation_sufficient': False
            }
        
        # Step 2: Evaluate model on shuffled labels multiple times
        shuffled_scores = []
        
        for i in range(self.n_shuffles):
            # Create shuffled copy of labels
            y_shuffled = y.copy()
            
            # Shuffle only training labels, keep test labels unchanged
            train_labels = y_shuffled[train_idx]
            np.random.shuffle(train_labels)
            y_shuffled[train_idx] = train_labels
            
            # Evaluate model with shuffled training labels
            try:
                shuffled_score = model_func(X, y_shuffled, train_idx, test_idx)
                shuffled_scores.append(shuffled_score)
            except Exception as e:
                # Handle individual shuffle failures gracefully
                shuffled_scores.append(0.0)
        
        if not shuffled_scores:
            return {
                'error': 'All shuffle evaluations failed',
                'test_passed': False,
                'degradation_sufficient': False
            }
        
        # Step 3: Compute performance metrics
        shuffled_scores = np.array(shuffled_scores)
        mean_shuffled = np.mean(shuffled_scores)
        std_shuffled = np.std(shuffled_scores)
        
        # Performance drop calculations
        absolute_drop = original_score - mean_shuffled
        
        # Handle division by zero or negative scores
        if abs(original_score) > 1e-10:
            relative_drop = absolute_drop / abs(original_score)
        else:
            relative_drop = 0.0
            
        # Step 4: Statistical significance test
        if len(shuffled_scores) > 1 and std_shuffled > 1e-10:
            z_score = absolute_drop / (std_shuffled / np.sqrt(len(shuffled_scores)))
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        else:
            z_score = 0.0
            p_value = 1.0
        
        # Step 5: Determine if degradation is sufficient
        degradation_sufficient = relative_drop >= self.degradation_threshold
        statistically_significant = p_value < self.significance_level
        test_passed = degradation_sufficient and statistically_significant
        
        return {
            'test_passed': test_passed,
            'degradation_sufficient': degradation_sufficient,
            'statistically_significant': statistically_significant,
            'original_score': float(original_score),
            'shuffled_scores': shuffled_scores.tolist(),
            'mean_shuffled_score': float(mean_shuffled),
            'std_shuffled_score': float(std_shuffled),
            'absolute_performance_drop': float(absolute_drop),
            'relative_performance_drop': float(relative_drop),
            'degradation_threshold': self.degradation_threshold,
            'p_value': float(p_value),
            'z_score': float(z_score),
            'significance_level': self.significance_level,
            'n_shuffles': self.n_shuffles,
            'performance_drop_percentage': f"{relative_drop:.1%}",
            'validation_status': self._get_validation_status(relative_drop, degradation_sufficient, test_passed)
        }
    
    def test_label_shuffling(self,
                           model_func: Callable,
                           X: np.ndarray,
                           y: np.ndarray,
                           train_idx: np.ndarray,
                           test_idx: np.ndarray) -> Dict[str, Any]:
        """Alias for run_shuffle_test for compatibility with existing tests."""
        return self.run_shuffle_test(model_func, X, y, train_idx, test_idx)
    
    def _get_validation_status(self, relative_drop: float, degradation_sufficient: bool, test_passed: bool) -> str:
        """Get human-readable validation status."""
        if test_passed:
            return f"✅ PASS - Model shows {relative_drop:.1%} degradation on shuffled labels"
        elif degradation_sufficient:
            return f"⚠️  MARGINAL - Degradation sufficient ({relative_drop:.1%}) but not statistically significant"
        else:
            return f"❌ FAIL - Insufficient degradation ({relative_drop:.1%}) - possible memorization or leakage"
    
    def validate_multiple_models(self,
                                models: Dict[str, Callable],
                                X: np.ndarray,
                                y: np.ndarray,
                                train_idx: np.ndarray,
                                test_idx: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple models with shuffle test.
        
        Args:
            models: Dictionary mapping model names to model functions
            X, y, train_idx, test_idx: Data and indices
            
        Returns:
            Dictionary mapping model names to their shuffle test results
        """
        results = {}
        
        for model_name, model_func in models.items():
            try:
                model_results = self.run_shuffle_test(model_func, X, y, train_idx, test_idx)
                model_results['model_name'] = model_name
                results[model_name] = model_results
            except Exception as e:
                results[model_name] = {
                    'model_name': model_name,
                    'error': str(e),
                    'test_passed': False,
                    'degradation_sufficient': False
                }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable report from shuffle test results."""
        if 'error' in results:
            return f"❌ SHUFFLE TEST ERROR: {results['error']}"
        
        report_lines = [
            "=" * 60,
            "SHUFFLE TEST VALIDATION REPORT",
            "=" * 60,
            "",
            f"Model Performance on Original Labels: {results['original_score']:.4f}",
            f"Model Performance on Shuffled Labels: {results['mean_shuffled_score']:.4f} ± {results['std_shuffled_score']:.4f}",
            f"Performance Drop: {results['absolute_performance_drop']:.4f} ({results['performance_drop_percentage']})",
            f"Required Degradation: ≥{self.degradation_threshold:.0%}",
            "",
            f"Statistical Significance: p = {results['p_value']:.6f}",
            f"Z-score: {results['z_score']:.3f}",
            f"Number of Shuffles: {results['n_shuffles']}",
            "",
            f"Status: {results['validation_status']}",
            "",
        ]
        
        if results['test_passed']:
            report_lines.extend([
                "✅ INTERPRETATION:",
                "   Model legitimately depends on label information",
                "   No evidence of memorization or information leakage",
                "   Performance degrades appropriately when labels are shuffled"
            ])
        else:
            report_lines.extend([
                "❌ INTERPRETATION:",
                "   Model may be memorizing training data or using leaked information",
                "   Performance does not degrade sufficiently on shuffled labels",
                "   RECOMMENDATION: Investigate data leakage and model architecture"
            ])
        
        return "\n".join(report_lines)


class ShuffleTest:
    """Compatibility class for existing test infrastructure."""
    
    def __init__(self, n_shuffles: int = 10):
        """Initialize with number of shuffles."""
        self.validator = ShuffleTestValidator(n_shuffles=n_shuffles)
        self.n_shuffles = n_shuffles
    
    def test_label_shuffling(self,
                           model_func: Callable,
                           X: np.ndarray,
                           y: np.ndarray,
                           train_idx: np.ndarray,
                           test_idx: np.ndarray) -> Dict[str, Any]:
        """Run shuffle test - compatibility method."""
        return self.validator.run_shuffle_test(model_func, X, y, train_idx, test_idx)


# Example model functions for testing
def example_correlation_model(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    """Example model that uses feature-target correlations."""
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]
    
    # Calculate correlations between features and target
    correlations = []
    for i in range(X_train.shape[1]):
        corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
        correlations.append(0 if np.isnan(corr) else corr)
    
    # Predict using weighted sum of features
    predictions = np.sum(X_test * correlations, axis=1)
    actual = y[test_idx]
    
    # Return correlation as performance metric
    if len(actual) > 1:
        performance = np.corrcoef(predictions, actual)[0, 1]
        return 0.0 if np.isnan(performance) else performance
    else:
        return 0.0


def example_memorization_model(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    """Example model that memorizes training examples (should fail shuffle test)."""
    # This model just returns a constant - demonstrates failure case
    return 0.5  # Constant performance regardless of label shuffling