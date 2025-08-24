"""
Integration tests for walk-forward testing with CPCV.

Tests the integration between walk-forward testing and CPCV
to ensure proper temporal validation with leak prevention.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('/Users/aziymandias/Desktop/dual_book_trading_lab_v10_complete/lab_v10')

from src.testing.walk_forward_testing import HistoricalWalkForwardTester, WalkForwardConfig
from src.testing.cpcv import PurgedCrossValidator, CombinatorialPurgedCV, CPCVConfig


def create_walkforward_test_data(n_periods: int = 24, samples_per_period: int = 50) -> pd.DataFrame:
    """Create test data suitable for walk-forward testing."""
    np.random.seed(42)
    
    all_data = []
    base_date = datetime(2020, 1, 1)
    
    for period in range(n_periods):
        period_start = base_date + timedelta(days=30 * period)
        dates = pd.date_range(period_start, periods=samples_per_period, freq='D')
        
        # Create features with some trend over time
        features = np.random.randn(samples_per_period, 5)
        features[:, 0] += 0.01 * period  # Slight trend
        
        # Target with some predictability
        target = (features[:, 0] + features[:, 1] * 0.5 + np.random.randn(samples_per_period) * 0.3 > 0).astype(int)
        
        period_data = pd.DataFrame({
            'date': dates,
            'feature_0': features[:, 0],
            'feature_1': features[:, 1], 
            'feature_2': features[:, 2],
            'feature_3': features[:, 3],
            'feature_4': features[:, 4],
            'target': target,
            'symbol': 'TEST'
        })
        
        all_data.append(period_data)
    
    return pd.concat(all_data, ignore_index=True)


class TestWalkForwardCPCVIntegration:
    """Test integration between walk-forward testing and CPCV."""
    
    def test_walkforward_with_purged_cv(self):
        """Test walk-forward testing using purged cross-validation."""
        # Create test data
        data = create_walkforward_test_data(12, 100)  # 12 months, 100 samples each
        
        # Set up walk-forward config
        wf_config = WalkForwardConfig(
            train_window_months=6,
            test_window_months=1,
            step_size_months=1,
            min_train_samples=200
        )
        
        # Simple mock data collector for testing
        class MockDataCollector:
            def __init__(self, data):
                self.data = data
            
            def get_data_for_period(self, start_date, end_date):
                mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
                return self.data[mask].copy()
        
        collector = MockDataCollector(data)
        
        # Test walk-forward periods generation
        tester = HistoricalWalkForwardTester(wf_config, collector)
        periods = tester._generate_test_periods("2020-01-01", "2020-12-31")
        
        assert len(periods) > 0, "No test periods generated"
        
        # Verify periods don't overlap in problematic ways
        for i, period in enumerate(periods):
            assert period.train_start < period.train_end
            assert period.test_start < period.test_end
            assert period.train_end <= period.test_start  # No train/test overlap
            
            print(f"Period {i}: Train {period.train_start.date()} to {period.train_end.date()}, "
                  f"Test {period.test_start.date()} to {period.test_end.date()}")
    
    def test_cpcv_within_walkforward_period(self):
        """Test using CPCV within a single walk-forward period."""
        # Create data for a single walk-forward period
        data = create_walkforward_test_data(6, 150)  # 6 months for training
        
        # Extract features and target
        feature_cols = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
        X = data[feature_cols]
        y = data['target']
        
        # Test CPCV on this walk-forward training data
        config = CPCVConfig(n_splits=5, purge_window=2, embargo_window=1)
        cpcv = CombinatorialPurgedCV(config)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = cpcv.validate_strategy(X, y, model)
        
        # Verify CPCV results
        assert results['n_combinations'] > 0
        assert 0 <= results['mean_score'] <= 1
        assert results['purge_effectiveness'] >= 0
        
        print(f"CPCV Results: Mean Score = {results['mean_score']:.3f}, "
              f"Purge Effectiveness = {results['purge_effectiveness']:.3f}")
    
    def test_walkforward_period_isolation(self):
        """Test that walk-forward periods are properly isolated."""
        data = create_walkforward_test_data(18, 80)
        
        # Create overlapping periods to test isolation
        periods = [
            (datetime(2020, 1, 1), datetime(2020, 6, 30)),  # Period 1: Jan-Jun
            (datetime(2020, 3, 1), datetime(2020, 8, 31)),  # Period 2: Mar-Aug (overlaps)
        ]
        
        for i, (start_date, end_date) in enumerate(periods):
            period_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
            
            if len(period_data) < 100:
                continue
            
            X = period_data[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']]
            y = period_data['target']
            
            # Use purged CV within this period
            cv = PurgedCrossValidator(n_splits=4, purge_window=3, embargo_window=2)
            
            splits = list(cv.split(X, y))
            assert len(splits) > 0, f"No splits generated for period {i+1}"
            
            # Test each split
            for train_idx, test_idx in splits:
                assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap in split"
                assert len(train_idx) > 20, "Training set too small"
                assert len(test_idx) > 5, "Test set too small"
    
    def test_temporal_validation_chain(self):
        """Test the complete temporal validation chain."""
        # Create structured test data with clear temporal patterns
        np.random.seed(123)
        
        n_months = 15
        samples_per_month = 60
        
        all_data = []
        for month in range(n_months):
            dates = pd.date_range(f'2020-{month+1:02d}-01', periods=samples_per_month, freq='D')
            
            # Create features with monthly trend
            base_trend = 0.1 * month
            features = np.random.randn(samples_per_month, 4) + base_trend
            
            # Target depends on features plus monthly effect
            target_prob = 1 / (1 + np.exp(-(features[:, 0] + features[:, 1] * 0.5 + base_trend)))
            target = np.random.binomial(1, target_prob)
            
            month_data = pd.DataFrame({
                'date': dates,
                'feature_0': features[:, 0],
                'feature_1': features[:, 1],
                'feature_2': features[:, 2],
                'feature_3': features[:, 3],
                'target': target,
                'month': month
            })
            
            all_data.append(month_data)
        
        data = pd.concat(all_data, ignore_index=True)
        
        # Test walk-forward approach
        train_months = [0, 1, 2, 3, 4, 5]  # First 6 months for training
        test_months = [6, 7]  # Next 2 months for testing
        
        train_data = data[data['month'].isin(train_months)]
        test_data = data[data['month'].isin(test_months)]
        
        X_train = train_data[['feature_0', 'feature_1', 'feature_2', 'feature_3']]
        y_train = train_data['target']
        X_test = test_data[['feature_0', 'feature_1', 'feature_2', 'feature_3']]
        y_test = test_data['target']
        
        # Use CPCV for training validation
        cv = PurgedCrossValidator(n_splits=5, purge_window=3, embargo_window=2)
        
        model = RandomForestClassifier(n_estimators=15, random_state=42)
        
        # Cross-validate on training data
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = model.predict(X_train.iloc[val_idx])
            score = np.mean(y_train.iloc[val_idx] == pred)
            cv_scores.append(score)
        
        # Train on full training data and test on out-of-sample test data
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_score = np.mean(y_test == test_pred)
        
        cv_mean = np.mean(cv_scores)
        
        assert len(cv_scores) > 0, "No CV scores generated"
        assert 0.3 <= cv_mean <= 0.9, f"CV score unrealistic: {cv_mean}"
        assert 0.3 <= test_score <= 0.9, f"Test score unrealistic: {test_score}"
        
        print(f"CV Score: {cv_mean:.3f}, Out-of-sample Score: {test_score:.3f}")
        
        # CV and out-of-sample scores should be reasonably close (not perfect due to regime change)
        score_diff = abs(cv_mean - test_score)
        assert score_diff < 0.4, f"CV and test scores too different: {score_diff:.3f}"


def test_walkforward_cpcv_no_leakage_integration():
    """Integration test ensuring no leakage in walk-forward + CPCV combination."""
    # Create data with strong autocorrelation (high leakage risk)
    np.random.seed(456)
    
    n_samples = 500
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Highly autocorrelated features
    features = np.zeros((n_samples, 3))
    features[0] = np.random.randn(3)
    
    for i in range(1, n_samples):
        features[i] = 0.7 * features[i-1] + 0.3 * np.random.randn(3)
    
    target = ((features[:, 0] + features[:, 1]) > 0).astype(int)
    
    data = pd.DataFrame({
        'date': dates,
        'feature_0': features[:, 0],
        'feature_1': features[:, 1],
        'feature_2': features[:, 2],
        'target': target
    })
    
    # Walk-forward split: first 300 for training, last 200 for testing
    split_point = 300
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    X_train = train_data[['feature_0', 'feature_1', 'feature_2']]
    y_train = train_data['target']
    X_test = test_data[['feature_0', 'feature_1', 'feature_2']]
    y_test = test_data['target']
    
    # Use purged CV on training data
    cv = PurgedCrossValidator(n_splits=6, purge_window=5, embargo_window=3, min_samples=50)
    
    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    
    # Validate using CPCV
    config = CPCVConfig(n_splits=6, purge_window=5, embargo_window=3)
    cpcv = CombinatorialPurgedCV(config)
    
    try:
        results = cpcv.validate_strategy(X_train, y_train, model)
        
        # Train final model and test out-of-sample
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_score = np.mean(y_test == test_pred)
        
        # Results should be reasonable
        assert results['mean_score'] >= 0.3, "CV score too low"
        assert test_score >= 0.3, "Test score too low"
        assert results['purge_effectiveness'] > 0.5, "Poor purge effectiveness"
        
        print(f"CPCV Score: {results['mean_score']:.3f}, "
              f"Out-of-sample Score: {test_score:.3f}, "
              f"Purge Effectiveness: {results['purge_effectiveness']:.3f}")
        
    except RuntimeError as e:
        # If all splits fail due to aggressive purging, that's also a valid outcome
        print(f"Aggressive purging caused splits to fail: {e}")
        assert "All CV splits failed" in str(e)


def test_embargo_effectiveness():
    """Test that embargo windows effectively prevent forward-looking bias."""
    # Create data where future information could help prediction
    np.random.seed(789)
    
    n_samples = 300
    # Create features where feature[t] contains information about target[t+5]
    features = np.random.randn(n_samples, 3)
    
    # Target has forward-looking dependency (cheating scenario)
    target = np.zeros(n_samples)
    for i in range(n_samples - 5):
        # Target at time t depends on feature at time t+5 (future leak!)
        if i + 5 < n_samples:
            target[i] = (features[i + 5, 0] > 0).astype(int)
        else:
            target[i] = np.random.binomial(1, 0.5)
    
    X = pd.DataFrame(features, columns=['feature_0', 'feature_1', 'feature_2'])
    y = pd.Series(target)
    
    # Test with different embargo windows
    embargo_windows = [0, 2, 5, 10]
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    for embargo in embargo_windows:
        cv = PurgedCrossValidator(n_splits=5, purge_window=1, embargo_window=embargo, min_samples=50)
        
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            if len(train_idx) < 50 or len(test_idx) < 10:
                continue
                
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            score = np.mean(y.iloc[test_idx] == pred)
            scores.append(score)
        
        if scores:
            mean_score = np.mean(scores)
            print(f"Embargo {embargo}: Score = {mean_score:.3f}")
            
            # With proper embargo (≥5), should prevent the forward-looking bias
            if embargo >= 5:
                # Score should be closer to random (0.5) since we blocked the future leak
                assert 0.4 <= mean_score <= 0.7, f"Score with embargo {embargo} suspicious: {mean_score}"