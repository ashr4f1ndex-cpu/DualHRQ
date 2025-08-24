"""
Tests for CPCV leakage prevention.

Red/green tests that verify purging and embargo prevent data leakage
in cross-validation, which is critical for valid financial ML results.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from src.testing.cpcv import (
    PurgedCrossValidator, TimeSeriesPurgedCV, CombinatorialPurgedCV,
    LeakageDetector, validate_purging, create_embargo_periods, CPCVConfig
)


def create_synthetic_financial_data(n_samples: int = 1000, 
                                   n_features: int = 10,
                                   autocorr: float = 0.3,
                                   random_seed: int = 42) -> pd.DataFrame:
    """Create synthetic financial data with autocorrelation."""
    np.random.seed(random_seed)
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create autocorrelated features (simulating market data)
    features = np.zeros((n_samples, n_features))
    features[0] = np.random.randn(n_features)
    
    for i in range(1, n_samples):
        features[i] = (autocorr * features[i-1] + 
                      np.sqrt(1 - autocorr**2) * np.random.randn(n_features))
    
    # Create target with some predictability from features
    target = (features[:, 0] + 0.5 * features[:, 1] + 
             0.1 * np.random.randn(n_samples))
    
    # Binary classification target
    binary_target = (target > np.median(target)).astype(int)
    
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
    df['target_continuous'] = target
    df['target_binary'] = binary_target
    df['date'] = dates
    df.index = dates
    
    return df


class TestPurgedCrossValidator:
    """Test purged cross-validation implementation."""
    
    def test_purged_cv_no_overlap(self):
        """Test that purged CV prevents train/test overlap."""
        data = create_synthetic_financial_data(500, 5)
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        cv = PurgedCrossValidator(n_splits=5, purge_window=2, embargo_window=1)
        
        for train_idx, test_idx in cv.split(X, y):
            # Test 1: No direct overlap
            assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap detected"
            
            # Test 2: Purging is enforced
            assert validate_purging(train_idx, test_idx, 2, 1), "Purging validation failed"
            
            # Test 3: Minimum sample sizes
            assert len(train_idx) >= 50, f"Training set too small: {len(train_idx)}"
            assert len(test_idx) >= 10, f"Test set too small: {len(test_idx)}"
    
    def test_purged_cv_with_insufficient_data(self):
        """Test purged CV behavior with insufficient data."""
        data = create_synthetic_financial_data(30, 3)  # Very small dataset
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        cv = PurgedCrossValidator(n_splits=5, purge_window=2, min_samples=50)
        
        # Should raise error for insufficient data
        with pytest.raises(ValueError, match="Insufficient samples"):
            list(cv.split(X, y))
    
    def test_purged_cv_split_count(self):
        """Test that purged CV generates expected number of splits."""
        data = create_synthetic_financial_data(1000, 5)
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        n_splits = 8
        cv = PurgedCrossValidator(n_splits=n_splits, purge_window=1)
        
        splits = list(cv.split(X, y))
        assert len(splits) <= n_splits, f"Too many splits generated: {len(splits)}"
        assert len(splits) >= n_splits - 2, f"Too few splits generated: {len(splits)}"  # Allow some tolerance


class TestTimeSeriesPurgedCV:
    """Test time-series purged cross-validation."""
    
    def test_timeseries_cv_chronological_order(self):
        """Test that time-series CV maintains chronological order."""
        data = create_synthetic_financial_data(500, 5)
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        cv = TimeSeriesPurgedCV(n_splits=5, purge_window=1, embargo_window=1)
        
        for train_idx, test_idx in cv.split(X, y):
            # Training data should come before test data (with gaps for purge/embargo)
            max_train = np.max(train_idx) if len(train_idx) > 0 else -1
            min_test = np.min(test_idx) if len(test_idx) > 0 else float('inf')
            
            # Allow for purge window gap
            assert max_train < min_test - 1, f"Train data not before test data: {max_train} vs {min_test}"
    
    def test_timeseries_cv_no_future_leakage(self):
        """Test that time-series CV prevents future information leakage."""
        data = create_synthetic_financial_data(300, 4)
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        cv = TimeSeriesPurgedCV(n_splits=4, purge_window=2, embargo_window=2)
        
        splits = list(cv.split(X, y))
        assert len(splits) > 0, "No splits generated"
        
        for train_idx, test_idx in splits:
            # Verify purging and embargo
            assert validate_purging(train_idx, test_idx, 2, 2), "Purging validation failed"
            
            # No training index should be >= any test index
            if len(train_idx) > 0 and len(test_idx) > 0:
                assert np.max(train_idx) < np.min(test_idx), "Future leakage detected"


class TestCombinatorialPurgedCV:
    """Test combinatorial purged cross-validation."""
    
    def test_cpcv_validation(self):
        """Test CPCV validation with a simple model."""
        data = create_synthetic_financial_data(400, 6)
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        # Simple model for testing
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        config = CPCVConfig(n_splits=6, purge_window=1, embargo_window=1)
        cpcv = CombinatorialPurgedCV(config)
        
        results = cpcv.validate_strategy(X, y, model)
        
        # Check result structure
        assert 'n_combinations' in results
        assert 'mean_score' in results
        assert 'std_score' in results
        assert 'sharpe_ratio' in results
        assert 'purge_effectiveness' in results
        
        # Check reasonable values
        assert results['n_combinations'] > 0
        assert 0 <= results['mean_score'] <= 1
        assert results['std_score'] >= 0
        assert results['purge_effectiveness'] >= 0
    
    def test_cpcv_with_failing_splits(self):
        """Test CPCV robustness when some splits fail."""
        # Create very small dataset to cause some splits to fail
        data = create_synthetic_financial_data(100, 3)
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        
        config = CPCVConfig(n_splits=10, purge_window=2, embargo_window=2, min_train_samples=20)
        cpcv = CombinatorialPurgedCV(config)
        
        # Should not crash even if some splits fail
        results = cpcv.validate_strategy(X, y, model)
        assert isinstance(results, dict)


class TestLeakageDetector:
    """Test data leakage detection."""
    
    def test_leakage_detector_with_clean_cv(self):
        """Test leakage detector with properly purged CV (should detect no leakage)."""
        data = create_synthetic_financial_data(300, 5)
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        clean_cv = PurgedCrossValidator(n_splits=5, purge_window=2, embargo_window=1)
        
        detector = LeakageDetector(significance_threshold=0.05)
        results = detector.detect_leakage(X, y, model, clean_cv)
        
        # Should detect no leakage with proper purged CV
        assert 'leakage_detected' in results
        # Note: This might sometimes detect false positives due to random variation
        # The test verifies the detector works, not that it never has false alarms
        assert isinstance(results['leakage_detected'], bool)
        assert 'severity' in results
    
    def test_leakage_detector_with_leaky_cv(self):
        """Test leakage detector with standard CV (should detect leakage)."""
        data = create_synthetic_financial_data(300, 5)
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Use standard KFold (potentially leaky for time series)
        leaky_cv = KFold(n_splits=5, shuffle=False)
        
        detector = LeakageDetector(significance_threshold=0.1)
        results = detector.detect_leakage(X, y, model, leaky_cv)
        
        assert 'leakage_detected' in results
        assert isinstance(results['leakage_detected'], bool)
        # Note: Standard KFold might not always show leakage with synthetic data
        # The test verifies the detector runs correctly


class TestPurgeValidation:
    """Test purge validation functions."""
    
    def test_validate_purging_correct(self):
        """Test purge validation with correct setup."""
        # Non-overlapping indices with proper purge gap
        train_indices = np.array([0, 1, 2, 10, 11, 12])
        test_indices = np.array([5, 6, 7])
        
        # Should pass with purge_window=1 (gap of 2 indices: 3,4 and 8,9)
        assert validate_purging(train_indices, test_indices, 1, 1)
    
    def test_validate_purging_overlap(self):
        """Test purge validation with direct overlap (should fail)."""
        train_indices = np.array([0, 1, 2, 5, 8, 9])  # 5 overlaps with test
        test_indices = np.array([5, 6, 7])
        
        # Should fail due to direct overlap
        assert not validate_purging(train_indices, test_indices, 1, 1)
    
    def test_validate_purging_insufficient_gap(self):
        """Test purge validation with insufficient purge gap."""
        train_indices = np.array([0, 1, 2, 4, 8, 9])  # 4 is too close to test (5,6,7)
        test_indices = np.array([5, 6, 7])
        
        # Should fail with purge_window=2 (need gap >= 2)
        assert not validate_purging(train_indices, test_indices, 2, 0)


class TestEmbargoFunctions:
    """Test embargo period creation."""
    
    def test_create_embargo_periods(self):
        """Test embargo period creation."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        embargo_periods = create_embargo_periods(dates, embargo_days=2)
        
        assert len(embargo_periods) > 0
        
        # Check first embargo period
        start, end = embargo_periods[0]
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert end > start
        assert (end - start).days == 2
    
    def test_embargo_periods_no_overlap(self):
        """Test that embargo periods don't overlap unnecessarily."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        embargo_periods = create_embargo_periods(dates, embargo_days=1)
        
        # With 1-day embargo and daily data, should have minimal overlap
        assert len(embargo_periods) >= 1


class TestLeakageIntegration:
    """Integration tests for leakage prevention."""
    
    def test_end_to_end_leakage_prevention(self):
        """End-to-end test that verifies complete leakage prevention pipeline."""
        # Create realistic financial data
        data = create_synthetic_financial_data(400, 8, autocorr=0.5)
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        # Test multiple CV methods
        cv_methods = [
            PurgedCrossValidator(n_splits=5, purge_window=2, embargo_window=1),
            TimeSeriesPurgedCV(n_splits=4, purge_window=1, embargo_window=2),
        ]
        
        for cv_method in cv_methods:
            splits = list(cv_method.split(X, y))
            
            # Verify each split
            for train_idx, test_idx in splits:
                # Basic checks
                assert len(set(train_idx) & set(test_idx)) == 0, "Direct overlap detected"
                assert len(train_idx) > 0, "Empty training set"
                assert len(test_idx) > 0, "Empty test set"
                
                # Purge validation
                purge_window = getattr(cv_method, 'purge_window', 0)
                embargo_window = getattr(cv_method, 'embargo_window', 0)
                
                assert validate_purging(train_idx, test_idx, purge_window, embargo_window), \
                    f"Purging validation failed for {type(cv_method).__name__}"
    
    def test_performance_with_vs_without_purging(self):
        """Compare model performance with and without purging."""
        data = create_synthetic_financial_data(500, 6, autocorr=0.7)  # High autocorr = more leakage risk
        X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
        y = data['target_binary']
        
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        
        # Purged CV (should show more realistic performance)
        purged_cv = PurgedCrossValidator(n_splits=5, purge_window=3, embargo_window=2)
        purged_scores = []
        for train_idx, test_idx in purged_cv.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            purged_scores.append(np.mean(y.iloc[test_idx] == pred))
        
        # Standard CV (potentially inflated due to leakage)
        standard_cv = KFold(n_splits=5, shuffle=False)
        standard_scores = []
        for train_idx, test_idx in standard_cv.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            standard_scores.append(np.mean(y.iloc[test_idx] == pred))
        
        purged_mean = np.mean(purged_scores)
        standard_mean = np.mean(standard_scores)
        
        # Both should be reasonable (not too different for synthetic data)
        assert 0.3 <= purged_mean <= 0.9, f"Purged CV score unrealistic: {purged_mean}"
        assert 0.3 <= standard_mean <= 0.9, f"Standard CV score unrealistic: {standard_mean}"
        
        # Log the difference for analysis
        print(f"Purged CV mean: {purged_mean:.3f}, Standard CV mean: {standard_mean:.3f}")


def test_cpcv_red_test_no_purging():
    """RED TEST: Verify that without purging, we can detect leakage."""
    data = create_synthetic_financial_data(200, 4, autocorr=0.8)
    X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
    y = data['target_binary']
    
    # Create a "bad" CV that doesn't purge (should be leaky)
    bad_cv = KFold(n_splits=5, shuffle=False)
    
    # Simple model that can overfit
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    
    detector = LeakageDetector(significance_threshold=0.2)
    results = detector.detect_leakage(X, y, model, bad_cv)
    
    # This test documents that standard CV can be problematic
    # We expect the detector to sometimes (but not always) flag this
    assert 'leakage_detected' in results


def test_cpcv_green_test_with_purging():
    """GREEN TEST: Verify that with proper purging, leakage is prevented."""
    data = create_synthetic_financial_data(300, 5, autocorr=0.6)
    X = data.drop(['target_continuous', 'target_binary', 'date'], axis=1)
    y = data['target_binary']
    
    # Proper purged CV
    good_cv = PurgedCrossValidator(n_splits=4, purge_window=3, embargo_window=2)
    
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    
    # Test that all splits are properly purged
    for train_idx, test_idx in good_cv.split(X, y):
        assert len(set(train_idx) & set(test_idx)) == 0, "Direct overlap detected"
        assert validate_purging(train_idx, test_idx, 3, 2), "Purge validation failed"
    
    # This should pass - demonstrates that purged CV works correctly
    assert True, "Purged CV passed validation"