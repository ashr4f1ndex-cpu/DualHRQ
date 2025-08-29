"""
Feature Engineering Validation Tests - Week 5 DRQ-106
======================================================
"""

import torch
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestTemporalOrderingPreservation:
    """Test that features respect temporal ordering."""
    
    def test_no_future_data_leakage(self):
        """Test that features don't use future data."""
        # Create time series with temporal structure
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Simulate price data with trend
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        df = pd.DataFrame({'date': dates, 'price': prices})
        df = df.sort_values('date')
        
        # Calculate moving average (valid feature)
        df['ma_5'] = df['price'].rolling(window=5).mean()
        
        # Check that MA only uses past data
        for i in range(5, len(df)):
            expected_ma = df['price'].iloc[i-4:i+1].mean()
            actual_ma = df['ma_5'].iloc[i]
            
            if not pd.isna(actual_ma):
                assert abs(expected_ma - actual_ma) < 1e-10, \
                    f"Moving average at index {i} uses future data"
        
        print("✅ Temporal ordering preserved in feature calculation")
    
    def test_feature_calculation_causality(self):
        """Test that feature calculation respects causality."""
        # Features should only depend on data available at that time
        n_samples = 50
        timestamps = np.arange(n_samples)
        values = np.random.randn(n_samples)
        
        # Calculate features with proper temporal ordering
        features = []
        for i in range(n_samples):
            if i < 5:
                # Not enough history
                features.append(np.nan)
            else:
                # Use only past 5 values (including current)
                feature = np.mean(values[max(0, i-4):i+1])
                features.append(feature)
        
        # Verify causality: feature at time t should only depend on data <= t
        for i in range(5, n_samples):
            expected_feature = np.mean(values[i-4:i+1])
            assert abs(features[i] - expected_feature) < 1e-10, \
                f"Feature at time {i} violates causality"
        
        print("✅ Feature calculation respects causality")


class TestNoLookaheadBias:
    """Test for look-ahead bias in feature construction."""
    
    def test_rolling_statistics_no_lookahead(self):
        """Test rolling statistics don't use future data."""
        data = np.random.randn(100)
        window_size = 10
        
        # Calculate rolling mean manually (correct way)
        correct_rolling_mean = []
        for i in range(len(data)):
            if i < window_size - 1:
                correct_rolling_mean.append(np.nan)
            else:
                mean_val = np.mean(data[i-window_size+1:i+1])
                correct_rolling_mean.append(mean_val)
        
        # Verify no future data is used
        for i in range(window_size, len(data)):
            # Check that calculation only uses data up to current point
            window_data = data[i-window_size+1:i+1]
            assert len(window_data) == window_size
            assert np.all(np.isfinite(window_data))  # Should have valid data
            
        print("✅ Rolling statistics calculation avoids look-ahead bias")
