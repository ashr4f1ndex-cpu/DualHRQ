"""
test_leakage_mi.py
==================

TDD tests for DRQ-002: Mutual Information Leakage Detection
These tests MUST be written first and will initially FAIL.
Implementation should make these tests pass.

CRITICAL: This is a P0 blocker - no ID leakage allowed.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from tools.check_no_static_ids import (
        scan_for_static_ids, 
        compute_mutual_information,
        MutualInformationTester
    )
except ImportError:
    # These will fail initially - that's expected in TDD
    pass


class TestMutualInformationDetection:
    """Tests for mutual information leakage detection."""
    
    def test_no_puzzle_id_in_features(self):
        """Test that no features are correlated with puzzle_id."""
        # Generate synthetic data that should pass
        n_samples = 1000
        n_features = 20
        
        # Features should be independent of puzzle_id
        features = np.random.randn(n_samples, n_features)
        puzzle_ids = np.random.randint(0, 100, n_samples)  # 100 different puzzles
        
        # This will fail initially until MutualInformationTester is implemented
        mi_tester = MutualInformationTester(mi_threshold=0.1)
        results = mi_tester.test_feature_leakage(features, puzzle_ids)
        
        # No features should leak puzzle_id information
        assert not results['leakage_detected'], \
            f"No leakage should be detected in random features"
        assert results['max_mi_score'] < 0.1, \
            f"Max MI score should be <0.1, got {results['max_mi_score']}"
        assert results['features_above_threshold'] == 0, \
            f"No features should be above threshold, got {results['features_above_threshold']}"
    
    def test_detects_puzzle_id_leakage(self):
        """Test that the system detects actual puzzle_id leakage."""
        # Generate data with intentional leakage
        n_samples = 1000
        n_features = 10
        
        # Most features are clean
        features = np.random.randn(n_samples, n_features)
        puzzle_ids = np.random.randint(0, 50, n_samples)
        
        # But one feature leaks puzzle_id information
        features[:, 0] = puzzle_ids + np.random.randn(n_samples) * 0.1  # Strong correlation
        
        mi_tester = MutualInformationTester(mi_threshold=0.1)
        results = mi_tester.test_feature_leakage(features, puzzle_ids)
        
        # Should detect the leakage
        assert results['leakage_detected'], \
            "Should detect puzzle_id leakage in corrupted features"
        assert results['max_mi_score'] > 0.1, \
            f"Max MI score should be >0.1 for leaky feature, got {results['max_mi_score']}"
        assert results['features_above_threshold'] > 0, \
            "Should identify leaky features"
    
    def test_no_puzzle_id_in_conditioning_output(self):
        """Test that conditioning system output doesn't leak puzzle_id."""
        # Simulate conditioning system output
        n_samples = 500
        conditioning_dim = 256
        
        # Conditioning output should be based on market features, not puzzle_id
        conditioning_output = np.random.randn(n_samples, conditioning_dim)
        puzzle_ids = np.random.randint(0, 75, n_samples)
        
        mi_tester = MutualInformationTester(mi_threshold=0.1)
        results = mi_tester.test_prediction_leakage(conditioning_output, puzzle_ids)
        
        # Conditioning should not leak puzzle_id
        assert not results['leakage_detected'], \
            "Conditioning output should not leak puzzle_id"
        assert results['max_mi_score'] < 0.1, \
            f"Conditioning MI should be <0.1, got {results['max_mi_score']}"
    
    def test_regime_features_independent(self):
        """Test that regime features are independent of puzzle_id."""
        # Simulate regime features (TSRV, BPV, Amihud, SSR/LULD state)
        n_samples = 800
        
        regime_features = {
            'tsrv_5m': np.random.exponential(0.2, n_samples),      # Time-scaled realized vol
            'tsrv_15m': np.random.exponential(0.18, n_samples),
            'tsrv_30m': np.random.exponential(0.15, n_samples),
            'tsrv_60m': np.random.exponential(0.12, n_samples),
            'bpv': np.random.exponential(0.1, n_samples),          # Bipower variation
            'amihud': np.random.exponential(0.05, n_samples),      # Illiquidity measure
            'ssr_active': np.random.binomial(1, 0.1, n_samples),   # SSR state
            'luld_active': np.random.binomial(1, 0.05, n_samples), # LULD state
        }
        
        puzzle_ids = np.random.randint(0, 120, n_samples)
        
        # Convert to array format
        feature_array = np.column_stack([regime_features[key] for key in regime_features.keys()])
        
        mi_tester = MutualInformationTester(mi_threshold=0.01)  # Stricter threshold for regime features
        results = mi_tester.test_feature_leakage(feature_array, puzzle_ids)
        
        # Regime features should be strongly independent of puzzle_id
        assert not results['leakage_detected'], \
            "Regime features should be independent of puzzle_id"
        assert results['max_mi_score'] < 0.01, \
            f"Regime feature MI should be <0.01, got {results['max_mi_score']}"
    
    def test_pattern_retrieval_independent(self):
        """Test that pattern retrieval doesn't depend on puzzle_id."""
        # Simulate pattern retrieval features
        n_samples = 600
        pattern_dim = 128
        
        # Pattern features should be based on market patterns, not puzzle IDs
        pattern_features = np.random.randn(n_samples, pattern_dim)
        puzzle_ids = np.random.randint(0, 80, n_samples)
        
        mi_tester = MutualInformationTester(mi_threshold=0.1)
        results = mi_tester.test_feature_leakage(pattern_features, puzzle_ids)
        
        # Pattern retrieval should be independent
        assert not results['leakage_detected'], \
            "Pattern features should be independent of puzzle_id"
        assert results['features_above_threshold'] == 0, \
            "No pattern features should leak puzzle_id"
    
    def test_mi_calculation_accuracy(self):
        """Test that mutual information calculation is accurate."""
        # Test with known relationships
        n_samples = 10000
        
        # Create perfectly correlated features
        x = np.random.randint(0, 10, n_samples)
        y = x.copy()  # Perfect correlation
        
        mi_score = compute_mutual_information(x, y)
        
        # Perfect correlation should have high MI
        assert mi_score > 1.0, \
            f"Perfect correlation should have high MI, got {mi_score}"
        
        # Create independent features  
        x_indep = np.random.randint(0, 10, n_samples)
        y_indep = np.random.randint(0, 10, n_samples)
        
        mi_score_indep = compute_mutual_information(x_indep, y_indep)
        
        # Independent features should have low MI
        assert mi_score_indep < 0.1, \
            f"Independent features should have low MI, got {mi_score_indep}"
    
    def test_mi_robust_to_data_types(self):
        """Test MI calculation works with different data types."""
        n_samples = 500
        
        # Test with continuous features
        continuous_features = np.random.randn(n_samples, 5)
        discrete_ids = np.random.randint(0, 20, n_samples)
        
        mi_tester = MutualInformationTester()
        results_continuous = mi_tester.test_feature_leakage(continuous_features, discrete_ids)
        
        # Should handle continuous features
        assert 'mi_scores' in results_continuous, \
            "Should return MI scores for continuous features"
        
        # Test with discrete features
        discrete_features = np.random.randint(0, 5, (n_samples, 3))
        
        results_discrete = mi_tester.test_feature_leakage(discrete_features, discrete_ids)
        
        # Should handle discrete features
        assert 'mi_scores' in results_discrete, \
            "Should return MI scores for discrete features"


class TestMIThresholdCalibration:
    """Tests for MI threshold calibration and sensitivity."""
    
    def test_threshold_sensitivity(self):
        """Test that different thresholds detect different levels of leakage."""
        n_samples = 1000
        n_features = 5
        
        # Create features with varying levels of puzzle_id correlation
        features = np.random.randn(n_samples, n_features)
        puzzle_ids = np.random.randint(0, 30, n_samples)
        
        # Add weak correlation to first feature
        features[:, 0] += puzzle_ids * 0.01
        # Add strong correlation to second feature  
        features[:, 1] += puzzle_ids * 0.1
        
        # Test with strict threshold
        mi_tester_strict = MutualInformationTester(mi_threshold=0.05)
        results_strict = mi_tester_strict.test_feature_leakage(features, puzzle_ids)
        
        # Test with loose threshold
        mi_tester_loose = MutualInformationTester(mi_threshold=0.2)
        results_loose = mi_tester_loose.test_feature_leakage(features, puzzle_ids)
        
        # Strict threshold should catch more leakage
        assert results_strict['features_above_threshold'] >= results_loose['features_above_threshold'], \
            "Strict threshold should detect more features"
    
    def test_false_positive_rate(self):
        """Test false positive rate on truly random data."""
        n_trials = 20
        false_positives = 0
        
        for _ in range(n_trials):
            # Generate truly independent data
            features = np.random.randn(500, 10)
            puzzle_ids = np.random.randint(0, 40, 500)
            
            mi_tester = MutualInformationTester(mi_threshold=0.1)
            results = mi_tester.test_feature_leakage(features, puzzle_ids)
            
            if results['leakage_detected']:
                false_positives += 1
        
        false_positive_rate = false_positives / n_trials
        
        # False positive rate should be low (<10%)
        assert false_positive_rate < 0.1, \
            f"False positive rate too high: {false_positive_rate:.2%}"


class TestStaticIDScanning:
    """Tests for static ID detection in codebase."""
    
    def test_scan_for_puzzle_id_usage(self):
        """Test scanning codebase for puzzle_id usage."""
        # This will fail initially until scan_for_static_ids is implemented
        codebase_path = str(Path(__file__).parent.parent.parent)
        allowlist = [
            "tests/",  # Tests can reference puzzle_id
            "docs/",   # Documentation can reference puzzle_id
            "*.md",    # Markdown files can reference puzzle_id
        ]
        
        violations = scan_for_static_ids(codebase_path, allowlist)
        
        # Should find no violations in production code
        production_violations = [v for v in violations if not any(
            allowed in v for allowed in allowlist
        )]
        
        assert len(production_violations) == 0, \
            f"Found puzzle_id usage in production code: {production_violations}"
    
    def test_allowlist_functionality(self):
        """Test that allowlist correctly excludes legitimate usage."""
        # Test that allowlist works correctly
        test_violations = [
            "src/models/hrm_net.py:123",
            "tests/test_model.py:45", 
            "docs/README.md:12",
            "src/features/regime.py:67"
        ]
        
        allowlist = ["tests/", "docs/", "*.md"]
        
        # Filter violations using allowlist logic
        filtered_violations = []
        for violation in test_violations:
            is_allowed = any(
                allowed in violation or violation.endswith(allowed.replace("*", ""))
                for allowed in allowlist
            )
            if not is_allowed:
                filtered_violations.append(violation)
        
        # Should only keep production code violations
        expected_violations = ["src/models/hrm_net.py:123", "src/features/regime.py:67"]
        assert len(filtered_violations) == 2, \
            f"Should filter to production violations only: {filtered_violations}"


# This will run when pytest is called and should initially FAIL
# Implementation should make these tests pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])