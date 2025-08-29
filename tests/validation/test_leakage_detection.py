"""
test_leakage_detection.py
=========================

Comprehensive tests for DRQ-105: Initial Leakage Validation

This test suite validates all leakage detection functionality:
- Mutual Information computation using KSG estimator
- Feature leakage detection
- Conditioning independence validation  
- Shuffle test for label dependency
- Statistical significance testing
- Integration with existing validation framework
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.validation.leakage_detector import (
    MutualInformationTester, 
    LeakageValidator,
    KSGMutualInformation,
    compute_mutual_information
)
from src.validation.shuffle_test import (
    ShuffleTestValidator,
    ShuffleTest,
    example_correlation_model,
    example_memorization_model
)


class TestKSGMutualInformation:
    """Test the KSG mutual information estimator."""
    
    def test_ksg_perfect_correlation(self):
        """Test KSG estimator on perfectly correlated data."""
        n_samples = 1000
        x = np.random.randint(0, 10, n_samples)
        y = x.copy()  # Perfect correlation
        
        ksg = KSGMutualInformation(k=3)
        mi = ksg.estimate_mi(x.reshape(-1, 1), y)
        
        # Perfect correlation should have high MI
        assert mi > 1.0, f"Perfect correlation should have high MI, got {mi:.3f}"
    
    def test_ksg_independent_variables(self):
        """Test KSG estimator on independent variables."""
        n_samples = 1000
        np.random.seed(42)
        
        x = np.random.randn(n_samples)
        y = np.random.randint(0, 10, n_samples)  # Independent
        
        ksg = KSGMutualInformation(k=3)
        mi = ksg.estimate_mi(x.reshape(-1, 1), y)
        
        # Independent variables should have low MI
        assert mi < 0.2, f"Independent variables should have low MI, got {mi:.3f}"
    
    def test_ksg_partial_correlation(self):
        """Test KSG estimator on partially correlated data."""
        n_samples = 1000
        np.random.seed(123)
        
        # Create partial correlation
        base = np.random.randn(n_samples)
        x = base + np.random.randn(n_samples) * 0.5  # Some correlation
        y_continuous = base + np.random.randn(n_samples) * 0.5
        y = (y_continuous > 0).astype(int)  # Convert to discrete
        
        ksg = KSGMutualInformation(k=3)
        mi = ksg.estimate_mi(x.reshape(-1, 1), y)
        
        # Partial correlation should have intermediate MI
        assert 0.1 < mi < 1.0, f"Partial correlation should have intermediate MI, got {mi:.3f}"
    
    def test_ksg_edge_cases(self):
        """Test KSG estimator edge cases."""
        ksg = KSGMutualInformation(k=3)
        
        # Test constant Y
        x = np.random.randn(100)
        y_constant = np.ones(100, dtype=int)
        mi = ksg.estimate_mi(x.reshape(-1, 1), y_constant)
        assert mi == 0.0, "Constant Y should give MI = 0"
        
        # Test small sample size
        x_small = np.random.randn(5)
        y_small = np.random.randint(0, 2, 5)
        mi_small = ksg.estimate_mi(x_small.reshape(-1, 1), y_small)
        assert mi_small >= 0.0, "MI should be non-negative"
        
        # Test single sample
        x_single = np.array([1.0])
        y_single = np.array([0])
        mi_single = ksg.estimate_mi(x_single.reshape(-1, 1), y_single)
        assert mi_single == 0.0, "Single sample should give MI = 0"


class TestMutualInformationTester:
    """Test the MutualInformationTester class."""
    
    def test_feature_leakage_detection_clean(self):
        """Test feature leakage detection on clean data."""
        n_samples = 500
        n_features = 10
        
        # Generate clean features
        np.random.seed(42)
        features = np.random.randn(n_samples, n_features)
        puzzle_ids = np.random.randint(0, 20, n_samples)
        
        mi_tester = MutualInformationTester(mi_threshold=0.2)  # Use higher threshold for testing
        results = mi_tester.test_feature_leakage(features, puzzle_ids)
        
        assert not results['leakage_detected'], "Clean features should not show leakage"
        assert results['max_mi_score'] < 0.25, f"Max MI should be reasonable for random data, got {results['max_mi_score']:.3f}"
        assert results['features_above_threshold'] == 0, "No features should exceed threshold"
        assert len(results['mi_scores']) == n_features, "Should have MI score for each feature"
    
    def test_feature_leakage_detection_corrupted(self):
        """Test feature leakage detection on corrupted data."""
        n_samples = 500
        n_features = 8
        
        # Generate features with one leaky feature
        np.random.seed(123)
        features = np.random.randn(n_samples, n_features)
        puzzle_ids = np.random.randint(0, 15, n_samples)
        
        # Corrupt first feature with puzzle_id information (strong correlation)
        features[:, 0] = puzzle_ids * 2.0 + np.random.randn(n_samples) * 0.05  # Very strong signal
        
        mi_tester = MutualInformationTester(mi_threshold=0.2)
        results = mi_tester.test_feature_leakage(features, puzzle_ids)
        
        assert results['leakage_detected'], "Should detect leakage in corrupted features"
        assert results['max_mi_score'] > 0.2, f"Max MI should exceed threshold, got {results['max_mi_score']:.3f}"
        assert results['features_above_threshold'] > 0, "Should identify leaky features"
        assert 0 in results['leaky_feature_indices'], "Should identify first feature as leaky"
    
    def test_prediction_leakage_detection(self):
        """Test prediction leakage detection."""
        n_samples = 300
        
        # Clean predictions
        np.random.seed(456)
        clean_predictions = np.random.randn(n_samples, 1)
        puzzle_ids = np.random.randint(0, 12, n_samples)
        
        mi_tester = MutualInformationTester(mi_threshold=0.2)
        
        # Test clean predictions
        clean_results = mi_tester.test_prediction_leakage(clean_predictions, puzzle_ids)
        assert not clean_results['leakage_detected'], "Clean predictions should not leak"
        
        # Test leaky predictions (strong correlation)
        leaky_predictions = puzzle_ids.reshape(-1, 1) * 3.0 + np.random.randn(n_samples, 1) * 0.05
        leaky_results = mi_tester.test_prediction_leakage(leaky_predictions, puzzle_ids)
        assert leaky_results['leakage_detected'], "Leaky predictions should be detected"
    
    def test_conditioning_independence(self):
        """Test conditioning system independence test."""
        n_samples = 400
        conditioning_dim = 64
        
        # Generate conditioning output independent of puzzle_id
        np.random.seed(789)
        conditioning_output = np.random.randn(n_samples, conditioning_dim)
        puzzle_ids = np.random.randint(0, 25, n_samples)
        
        mi_tester = MutualInformationTester(mi_threshold=0.3)  # Higher threshold for conditioning test
        results = mi_tester.test_conditioning_independence(conditioning_output, puzzle_ids)
        
        assert results['test_passed'], "Independent conditioning should pass test"
        assert results['mi_score'] < 0.3, f"MI should be below threshold, got {results['mi_score']:.3f}"
        assert not results['test_failed'], "Test should not fail for independent data"
        
        # Test that results contain expected keys
        expected_keys = ['test_passed', 'mi_score', 'normalized_mi', 'max_possible_mi', 
                        'n_unique_puzzles', 'sample_size', 'violation_severity']
        for key in expected_keys:
            assert key in results, f"Missing key in results: {key}"
    
    def test_mi_threshold_sensitivity(self):
        """Test different MI thresholds."""
        n_samples = 300
        features = np.random.randn(n_samples, 5)
        puzzle_ids = np.random.randint(0, 10, n_samples)
        
        # Add weak correlation
        features[:, 0] += puzzle_ids * 0.02
        
        # Test with strict threshold
        strict_tester = MutualInformationTester(mi_threshold=0.05)
        strict_results = strict_tester.test_feature_leakage(features, puzzle_ids)
        
        # Test with loose threshold
        loose_tester = MutualInformationTester(mi_threshold=0.3)
        loose_results = loose_tester.test_feature_leakage(features, puzzle_ids)
        
        # Strict threshold should be more sensitive
        assert (strict_results['features_above_threshold'] >= 
                loose_results['features_above_threshold'])


class TestLeakageValidator:
    """Test the LeakageValidator orchestrator."""
    
    def test_system_integrity_validation_clean(self):
        """Test complete system validation on clean data."""
        n_samples = 300
        n_features = 8
        
        # Generate clean system data
        np.random.seed(999)
        features = np.random.randn(n_samples, n_features)
        conditioning_output = np.random.randn(n_samples, 32)
        predictions = np.random.randn(n_samples, 1)
        puzzle_ids = np.random.randint(0, 15, n_samples)
        
        validator = LeakageValidator(mi_threshold=0.1)
        results = validator.validate_system_integrity(
            features, conditioning_output, predictions, puzzle_ids
        )
        
        assert results['system_integrity_passed'], "Clean system should pass integrity check"
        assert not results['system_integrity_failed'], "Clean system should not fail"
        assert not results['critical_violations'], "Should have no critical violations"
        assert len(results['recommendations']) > 0, "Should provide recommendations"
        
        # Check that all sub-tests are included
        assert 'feature_leakage_results' in results
        assert 'conditioning_independence_results' in results
        assert 'prediction_leakage_results' in results
    
    def test_system_integrity_validation_corrupted(self):
        """Test complete system validation on corrupted data."""
        n_samples = 300
        
        # Generate system data with leakage
        np.random.seed(111)
        features = np.random.randn(n_samples, 5)
        puzzle_ids = np.random.randint(0, 20, n_samples)
        
        # Inject leakage into conditioning output
        conditioning_output = np.random.randn(n_samples, 16)
        conditioning_output[:, 0] = puzzle_ids  # Direct leakage
        
        predictions = np.random.randn(n_samples, 1)
        
        validator = LeakageValidator(mi_threshold=0.1)
        results = validator.validate_system_integrity(
            features, conditioning_output, predictions, puzzle_ids
        )
        
        assert not results['system_integrity_passed'], "Corrupted system should fail integrity check"
        assert results['system_integrity_failed'], "Corrupted system should fail"
        assert results['max_mi_score_found'] > 0.1, "Should detect high MI score"


class TestShuffleTestValidator:
    """Test the shuffle test validator."""
    
    def test_shuffle_test_legitimate_model(self):
        """Test shuffle test on legitimate model that depends on labels."""
        n_samples = 500
        n_features = 6
        
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.2  # Legitimate dependency
        
        train_idx = np.arange(350)
        test_idx = np.arange(350, 500)
        
        validator = ShuffleTestValidator(n_shuffles=8, degradation_threshold=0.5)
        results = validator.run_shuffle_test(example_correlation_model, X, y, train_idx, test_idx)
        
        assert results['test_passed'], "Legitimate model should pass shuffle test"
        assert results['degradation_sufficient'], "Should show sufficient performance degradation"
        assert results['relative_performance_drop'] > 0.5, f"Should drop >50%, got {results['relative_performance_drop']:.1%}"
        assert results['original_score'] > results['mean_shuffled_score'], "Original should outperform shuffled"
    
    def test_shuffle_test_memorization_model(self):
        """Test shuffle test detects memorization."""
        n_samples = 200
        X = np.random.randn(n_samples, 4)
        y = np.random.randn(n_samples)
        
        train_idx = np.arange(150)
        test_idx = np.arange(150, 200)
        
        validator = ShuffleTestValidator(n_shuffles=5, degradation_threshold=0.5)
        results = validator.run_shuffle_test(example_memorization_model, X, y, train_idx, test_idx)
        
        # Memorization model should not show sufficient degradation
        assert not results['degradation_sufficient'], "Memorization model should not show sufficient degradation"
        assert results['relative_performance_drop'] < 0.3, f"Memorization should show <30% drop, got {results['relative_performance_drop']:.1%}"
    
    def test_shuffle_test_statistical_significance(self):
        """Test statistical significance calculation in shuffle test."""
        n_samples = 400
        np.random.seed(456)
        
        X = np.random.randn(n_samples, 5)
        y = np.sum(X, axis=1) + np.random.randn(n_samples) * 0.1  # Strong signal
        
        train_idx = np.arange(300)
        test_idx = np.arange(300, 400)
        
        validator = ShuffleTestValidator(n_shuffles=10, significance_level=0.05)
        results = validator.run_shuffle_test(example_correlation_model, X, y, train_idx, test_idx)
        
        assert results['statistically_significant'], "Strong model should be statistically significant"
        assert results['p_value'] < 0.05, f"P-value should be <0.05, got {results['p_value']:.6f}"
        assert abs(results['z_score']) > 1.96, f"Z-score should be >1.96 for significance, got {results['z_score']:.3f}"
    
    def test_shuffle_test_edge_cases(self):
        """Test shuffle test edge cases."""
        validator = ShuffleTestValidator(n_shuffles=3)
        
        # Test tiny dataset
        X_small = np.random.randn(10, 2)
        y_small = np.random.randn(10)
        train_idx_small = np.arange(7)
        test_idx_small = np.arange(7, 10)
        
        results = validator.run_shuffle_test(example_correlation_model, X_small, y_small, 
                                           train_idx_small, test_idx_small)
        
        assert 'original_score' in results, "Should handle small dataset"
        assert 'mean_shuffled_score' in results, "Should return shuffled results"
        assert not ('error' in results), "Should not error on small dataset"
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            validator.run_shuffle_test(example_correlation_model, X_small, y_small,
                                     np.array([]), test_idx_small)  # Empty train_idx
    
    def test_multiple_models_validation(self):
        """Test validating multiple models simultaneously."""
        n_samples = 300
        X = np.random.randn(n_samples, 4)
        y = np.sum(X, axis=1) + np.random.randn(n_samples) * 0.3
        
        train_idx = np.arange(200)
        test_idx = np.arange(200, 300)
        
        models = {
            'legitimate_model': example_correlation_model,
            'memorization_model': example_memorization_model
        }
        
        validator = ShuffleTestValidator(n_shuffles=5)
        results = validator.validate_multiple_models(models, X, y, train_idx, test_idx)
        
        assert 'legitimate_model' in results
        assert 'memorization_model' in results
        assert results['legitimate_model']['degradation_sufficient']
        assert not results['memorization_model']['degradation_sufficient']
    
    def test_shuffle_test_report_generation(self):
        """Test report generation."""
        n_samples = 200
        X = np.random.randn(n_samples, 3)
        y = np.sum(X, axis=1) + np.random.randn(n_samples) * 0.2
        
        train_idx = np.arange(150)
        test_idx = np.arange(150, 200)
        
        validator = ShuffleTestValidator(n_shuffles=5)
        results = validator.run_shuffle_test(example_correlation_model, X, y, train_idx, test_idx)
        
        report = validator.generate_report(results)
        
        assert "SHUFFLE TEST VALIDATION REPORT" in report
        assert "Performance Drop" in report
        assert "Statistical Significance" in report
        assert ("✅ PASS" in report or "❌ FAIL" in report)


class TestShuffleTestCompatibility:
    """Test compatibility with existing test infrastructure."""
    
    def test_shuffle_test_class_compatibility(self):
        """Test ShuffleTest class for compatibility."""
        shuffle_test = ShuffleTest(n_shuffles=5)
        
        # Generate test data
        n_samples = 200
        X = np.random.randn(n_samples, 4)
        y = np.sum(X[:, :2], axis=1) + np.random.randn(n_samples) * 0.3
        
        train_idx = np.arange(150)
        test_idx = np.arange(150, 200)
        
        results = shuffle_test.test_label_shuffling(example_correlation_model, X, y, train_idx, test_idx)
        
        # Should return expected keys for compatibility
        expected_keys = ['degradation_sufficient', 'relative_performance_drop', 
                        'original_score', 'mean_shuffled_score', 'shuffled_scores']
        for key in expected_keys:
            assert key in results, f"Missing compatibility key: {key}"


class TestIntegrationWithExistingTests:
    """Test integration with existing validation framework."""
    
    def test_compute_mutual_information_function(self):
        """Test compatibility function for existing tests."""
        n_samples = 200
        x = np.random.randn(n_samples)
        y = np.random.randint(0, 5, n_samples)
        
        mi = compute_mutual_information(x, y)
        
        assert isinstance(mi, float)
        assert mi >= 0.0, "MI should be non-negative"
    
    def test_acceptance_criteria_validation(self):
        """Test all acceptance criteria from DRQ-105."""
        
        # Acceptance Criterion 1: MI(conditioning_output, puzzle_id) < 0.1 bits
        n_samples = 400
        conditioning_output = np.random.randn(n_samples, 32)
        puzzle_ids = np.random.randint(0, 20, n_samples)
        
        mi_tester = MutualInformationTester(mi_threshold=0.2)
        conditioning_results = mi_tester.test_conditioning_independence(conditioning_output, puzzle_ids)
        
        assert conditioning_results['test_passed'], "✅ Conditioning independence test should pass"
        assert conditioning_results['mi_score'] < 0.1, "✅ MI should be < 0.1 bits"
        
        # Acceptance Criterion 2: Shuffle test shows >50% performance degradation
        X = np.random.randn(n_samples, 6)
        y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.2
        train_idx = np.arange(300)
        test_idx = np.arange(300, 400)
        
        shuffle_validator = ShuffleTestValidator(n_shuffles=8, degradation_threshold=0.5)
        shuffle_results = shuffle_validator.run_shuffle_test(example_correlation_model, X, y, train_idx, test_idx)
        
        assert shuffle_results['degradation_sufficient'], "✅ Shuffle test should show >50% degradation"
        assert shuffle_results['relative_performance_drop'] > 0.5, "✅ Performance should drop >50%"
        
        # Acceptance Criterion 3: Feature leakage tests pass
        features = np.random.randn(n_samples, 10)
        feature_results = mi_tester.test_feature_leakage(features, puzzle_ids)
        
        assert not feature_results['leakage_detected'], "✅ Feature leakage tests should pass"
        
        # Acceptance Criterion 4: All tests run and pass
        validator = LeakageValidator(mi_threshold=0.1)
        predictions = np.random.randn(n_samples, 1)
        
        system_results = validator.validate_system_integrity(
            features, conditioning_output, predictions, puzzle_ids
        )
        
        assert system_results['system_integrity_passed'], "✅ All tests should run and pass"
        
        print("✅ ALL ACCEPTANCE CRITERIA VALIDATED")


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    def test_performance_large_dataset(self):
        """Test performance on larger datasets."""
        n_samples = 2000
        n_features = 20
        
        features = np.random.randn(n_samples, n_features)
        puzzle_ids = np.random.randint(0, 50, n_samples)
        
        mi_tester = MutualInformationTester(mi_threshold=0.2)
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        results = mi_tester.test_feature_leakage(features, puzzle_ids)
        elapsed = time.time() - start_time
        
        assert elapsed < 30.0, f"Large dataset test should complete quickly, took {elapsed:.2f}s"
        assert len(results['mi_scores']) == n_features, "Should process all features"
    
    def test_memory_efficiency(self):
        """Test memory efficiency on reasonably sized datasets."""
        n_samples = 1000
        n_features = 15
        
        features = np.random.randn(n_samples, n_features)
        puzzle_ids = np.random.randint(0, 30, n_samples)
        
        validator = LeakageValidator(mi_threshold=0.1)
        
        # Should not cause memory issues
        conditioning_output = np.random.randn(n_samples, 64)
        predictions = np.random.randn(n_samples, 1)
        
        results = validator.validate_system_integrity(
            features, conditioning_output, predictions, puzzle_ids
        )
        
        # Basic sanity checks
        assert isinstance(results, dict)
        assert 'system_integrity_passed' in results


if __name__ == "__main__":
    # Run specific test groups
    pytest.main([
        __file__ + "::TestKSGMutualInformation",
        __file__ + "::TestMutualInformationTester", 
        __file__ + "::TestShuffleTestValidator",
        __file__ + "::TestIntegrationWithExistingTests",
        "-v"
    ])