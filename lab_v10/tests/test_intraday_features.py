"""
Test suite for intraday feature engineering module.

Comprehensive tests for:
- VWAP calculation with daily resets
- ATR calculation with multiple methods
- Stretch metrics and volatility adjustment
- SSR Gate logic and uptick rules
- LULD mechanics and band violations
- Momentum and mean reversion indicators
- Market hours and timezone handling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from lab_v10.src.common.intraday_features import (
    IntradayFeatureEngine,
    is_market_hours,
    calculate_time_to_close
)


class TestIntradayFeatureEngine:
    """Test cases for IntradayFeatureEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create test engine instance."""
        return IntradayFeatureEngine(
            atr_periods=[14, 20],
            timezone="America/New_York"
        )
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        # Create intraday data for 2 trading days
        start_time = pd.Timestamp('2024-01-02 09:30', tz='America/New_York')
        periods = 2 * 390  # 2 days * 6.5 hours * 60 minutes
        
        timestamps = []
        current_time = start_time
        
        for i in range(periods):
            timestamps.append(current_time)
            current_time += pd.Timedelta(minutes=1)
            
            # Skip to next day if after market close
            if current_time.time() >= time(16, 0):
                next_day = current_time.date() + timedelta(days=1)
                # Skip weekends
                while next_day.weekday() >= 5:
                    next_day += timedelta(days=1)
                current_time = pd.Timestamp.combine(next_day, time(9, 30)).tz_localize('America/New_York')
        
        timestamps = timestamps[:periods]
        
        # Generate realistic OHLCV data
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0, 0.001, len(timestamps))
        closes = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from closes
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.002, len(timestamps))))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.002, len(timestamps))))
        
        volumes = np.random.lognormal(10, 0.5, len(timestamps))
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=pd.DatetimeIndex(timestamps))
    
    def test_calculate_vwap(self, engine, sample_ohlcv_data):
        """Test VWAP calculation with daily resets."""
        vwap = engine.calculate_vwap(sample_ohlcv_data)
        
        # Basic validation
        assert len(vwap) == len(sample_ohlcv_data)
        assert not vwap.isnull().all()
        assert all(vwap > 0)  # VWAP should be positive
        
        # Test daily reset behavior
        dates = vwap.index.date
        unique_dates = np.unique(dates)
        
        if len(unique_dates) > 1:
            # Check that VWAP resets each day (first value of day should be close to typical price)
            for date in unique_dates[1:]:  # Skip first date
                day_mask = dates == date
                day_data = sample_ohlcv_data[day_mask]
                day_vwap = vwap[day_mask]
                
                if len(day_data) > 0:
                    first_typical = (day_data.iloc[0]['high'] + day_data.iloc[0]['low'] + day_data.iloc[0]['close']) / 3
                    first_vwap = day_vwap.iloc[0]
                    
                    # Should be close since VWAP resets daily
                    assert abs(first_vwap - first_typical) / first_typical < 0.01
    
    def test_vwap_with_zero_volume(self, engine):
        """Test VWAP handling of zero volume periods."""
        # Create data with some zero volume periods
        data = pd.DataFrame({
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 0, 2000]  # Zero volume in middle
        }, index=pd.date_range('2024-01-01 10:00', periods=3, freq='1min'))
        
        vwap = engine.calculate_vwap(data)
        
        # Should handle zero volume gracefully
        assert not vwap.isnull().all()
        assert len(vwap) == len(data)
    
    def test_calculate_atr_wilder(self, engine, sample_ohlcv_data):
        """Test ATR calculation using Wilder's method."""
        atr = engine.calculate_atr(sample_ohlcv_data, period=14, method='wilder')
        
        # Basic validation
        assert len(atr) == len(sample_ohlcv_data)
        assert atr.iloc[13:].notna().all()  # Should have values after period
        assert (atr.iloc[13:] > 0).all()  # ATR should be positive
        
        # Check that it's roughly reasonable vs price level
        avg_atr = atr.iloc[13:].mean()
        avg_price = sample_ohlcv_data['close'].mean()
        assert 0.001 < avg_atr / avg_price < 0.1  # ATR should be reasonable fraction of price
    
    def test_calculate_atr_sma(self, engine, sample_ohlcv_data):
        """Test ATR calculation using simple moving average."""
        atr = engine.calculate_atr(sample_ohlcv_data, period=14, method='sma')
        
        # Should produce different results than Wilder's method
        atr_wilder = engine.calculate_atr(sample_ohlcv_data, period=14, method='wilder')
        
        # Results should be similar but not identical
        correlation = atr.iloc[20:].corr(atr_wilder.iloc[20:])
        assert correlation > 0.95  # Should be highly correlated
        assert not atr.iloc[20:].equals(atr_wilder.iloc[20:])  # But not identical
    
    def test_calculate_stretch_metrics(self, engine, sample_ohlcv_data):
        """Test stretch metrics calculation."""
        stretch_metrics = engine.calculate_stretch_metrics(sample_ohlcv_data)
        
        required_metrics = [
            'stretch_pct', 'vol_adjusted_stretch', 'high_stretch', 
            'low_stretch', 'stretch_momentum', 'cumulative_stretch'
        ]
        
        # Check all required metrics are present
        assert all(metric in stretch_metrics for metric in required_metrics)
        
        # Check data types and lengths
        for metric_name, metric_series in stretch_metrics.items():
            assert isinstance(metric_series, pd.Series)
            assert len(metric_series) == len(sample_ohlcv_data)
        
        # Check that stretch percentage is reasonable
        stretch_pct = stretch_metrics['stretch_pct']
        assert abs(stretch_pct.mean()) < 10  # Average stretch shouldn't be too extreme
    
    def test_calculate_ssr_gate(self, engine, sample_ohlcv_data):
        """Test SSR Gate calculation."""
        # Create data with a significant decline to trigger SSR
        data = sample_ohlcv_data.copy()
        
        # Simulate 10% decline on second day
        second_day_mask = data.index.date == data.index.date[400]  # Second day
        data.loc[second_day_mask, 'close'] *= 0.9
        data.loc[second_day_mask, 'low'] *= 0.89
        data.loc[second_day_mask, 'high'] *= 0.91
        
        ssr_features = engine.calculate_ssr_gate(data)
        
        required_features = [
            'ssr_triggered', 'ssr_active', 'ssr_compliant',
            'uptick_distance', 'decline_from_prev_close'
        ]
        
        assert all(feature in ssr_features for feature in required_features)
        
        # Check that SSR was triggered
        assert ssr_features['ssr_triggered'].sum() > 0
        
        # Check that SSR active periods include triggered periods
        triggered_indices = ssr_features['ssr_triggered'] == 1
        active_indices = ssr_features['ssr_active'] == 1
        assert (triggered_indices <= active_indices).all()
    
    def test_calculate_luld_mechanics(self, engine, sample_ohlcv_data):
        """Test LULD mechanics calculation."""
        luld_features = engine.calculate_luld_mechanics(sample_ohlcv_data)
        
        required_features = [
            'luld_upper_band', 'luld_lower_band', 'limit_up_violation',
            'limit_down_violation', 'distance_to_upper_band', 'distance_to_lower_band',
            'luld_band_pressure', 'in_limit_state', 'luld_band_width'
        ]
        
        assert all(feature in luld_features for feature in required_features)
        
        # Check band relationships
        upper_band = luld_features['luld_upper_band']
        lower_band = luld_features['luld_lower_band']
        close_prices = sample_ohlcv_data['close']
        
        # Upper band should be above lower band
        assert (upper_band > lower_band).all()
        
        # Bands should be around current prices
        assert (upper_band > close_prices).all()
        assert (lower_band < close_prices).all()
        
        # Band width should be positive
        assert (luld_features['luld_band_width'] > 0).all()
    
    def test_calculate_momentum_indicators(self, engine, sample_ohlcv_data):
        """Test momentum indicators calculation."""
        momentum_indicators = engine.calculate_momentum_indicators(sample_ohlcv_data)
        
        # Check for expected momentum features
        expected_features = ['momentum_5', 'momentum_10', 'rsi', 'stoch_k', 'stoch_d']
        assert all(feature in momentum_indicators for feature in expected_features)
        
        # Check RSI bounds
        rsi = momentum_indicators['rsi']
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()
        
        # Check Stochastic bounds
        stoch_k = momentum_indicators['stoch_k']
        valid_stoch = stoch_k.dropna()
        if len(valid_stoch) > 0:
            assert (valid_stoch >= 0).all()
            assert (valid_stoch <= 100).all()
    
    def test_calculate_mean_reversion_indicators(self, engine, sample_ohlcv_data):
        """Test mean reversion indicators calculation."""
        mr_indicators = engine.calculate_mean_reversion_indicators(sample_ohlcv_data)
        
        expected_features = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width']
        assert all(feature in mr_indicators for feature in expected_features)
        
        # Check Bollinger Band relationships
        bb_upper = mr_indicators['bb_upper']
        bb_middle = mr_indicators['bb_middle']
        bb_lower = mr_indicators['bb_lower']
        
        valid_mask = bb_upper.notna() & bb_middle.notna() & bb_lower.notna()
        if valid_mask.any():
            assert (bb_upper[valid_mask] >= bb_middle[valid_mask]).all()
            assert (bb_middle[valid_mask] >= bb_lower[valid_mask]).all()
        
        # Check BB position bounds
        bb_position = mr_indicators['bb_position']
        valid_position = bb_position.dropna()
        if len(valid_position) > 0:
            # Should mostly be between 0 and 1, but can exceed in extreme cases
            assert valid_position.quantile(0.05) >= -0.5
            assert valid_position.quantile(0.95) <= 1.5
    
    def test_extract_all_features(self, engine, sample_ohlcv_data):
        """Test comprehensive feature extraction."""
        all_features_df = engine.extract_all_features(sample_ohlcv_data)
        
        # Should include original columns plus new features
        assert len(all_features_df.columns) > len(sample_ohlcv_data.columns)
        
        # Should include core features
        core_features = ['vwap', 'atr_14', 'stretch_pct', 'ssr_active']
        assert all(feature in all_features_df.columns for feature in core_features)
        
        # Check data integrity
        assert len(all_features_df) == len(sample_ohlcv_data)
        assert all_features_df.index.equals(sample_ohlcv_data.index)
    
    def test_leakage_prevention(self, engine, sample_ohlcv_data):
        """Test that features don't use future information."""
        # Test VWAP calculation doesn't use future data
        vwap = engine.calculate_vwap(sample_ohlcv_data)
        
        # For each point, VWAP should only depend on current and past data
        for i in range(10, len(sample_ohlcv_data)):
            partial_data = sample_ohlcv_data.iloc[:i+1]
            partial_vwap = engine.calculate_vwap(partial_data)
            
            # VWAP at position i should be same whether calculated on partial or full data
            assert abs(vwap.iloc[i] - partial_vwap.iloc[i]) < 1e-10
    
    def test_edge_cases(self, engine):
        """Test edge cases and error handling."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000]
        }, index=[pd.Timestamp('2024-01-01 10:00')])
        
        vwap = engine.calculate_vwap(minimal_data)
        assert len(vwap) == 1
        assert not pd.isna(vwap.iloc[0])
        
        # Test with NaN values
        data_with_nans = sample_ohlcv_data.copy()
        data_with_nans.iloc[5:10, :] = np.nan
        
        result = engine.extract_all_features(data_with_nans)
        assert len(result) == len(data_with_nans)


class TestMarketHoursUtilities:
    """Test market hours and timezone utilities."""
    
    def test_is_market_hours(self):
        """Test market hours detection."""
        # Market hours timestamp
        market_time = pd.Timestamp('2024-01-02 14:30', tz='America/New_York')  # 2:30 PM ET
        assert is_market_hours(market_time)
        
        # Before market hours
        early_time = pd.Timestamp('2024-01-02 08:00', tz='America/New_York')  # 8:00 AM ET
        assert not is_market_hours(early_time)
        
        # After market hours
        late_time = pd.Timestamp('2024-01-02 18:00', tz='America/New_York')  # 6:00 PM ET
        assert not is_market_hours(late_time)
    
    def test_calculate_time_to_close(self):
        """Test time to close calculation."""
        # 2 hours before close
        timestamp = pd.Timestamp('2024-01-02 14:00', tz='America/New_York')
        time_to_close = calculate_time_to_close(timestamp)
        
        assert abs(time_to_close - 2.0) < 0.1  # Should be close to 2 hours
        
        # 30 minutes before close
        timestamp = pd.Timestamp('2024-01-02 15:30', tz='America/New_York')
        time_to_close = calculate_time_to_close(timestamp)
        
        assert abs(time_to_close - 0.5) < 0.1  # Should be close to 0.5 hours


class TestRobustness:
    """Test robustness and performance."""
    
    def test_different_price_levels(self):
        """Test features work across different price levels."""
        engine = IntradayFeatureEngine()
        
        # Test with penny stock
        penny_data = pd.DataFrame({
            'open': [1.0, 1.01, 0.99],
            'high': [1.02, 1.03, 1.01],
            'low': [0.98, 0.99, 0.97],
            'close': [1.01, 0.99, 1.00],
            'volume': [10000, 15000, 12000]
        }, index=pd.date_range('2024-01-01 10:00', periods=3, freq='1min'))
        
        features_penny = engine.extract_all_features(penny_data)
        
        # Test with high-priced stock
        expensive_data = penny_data.copy()
        price_cols = ['open', 'high', 'low', 'close']
        expensive_data[price_cols] *= 1000
        
        features_expensive = engine.extract_all_features(expensive_data)
        
        # Relative features should be similar
        penny_stretch = features_penny['stretch_pct']
        expensive_stretch = features_expensive['stretch_pct']
        
        # Should be similar since it's a percentage
        assert abs(penny_stretch.iloc[-1] - expensive_stretch.iloc[-1]) < 1.0
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        engine = IntradayFeatureEngine()
        
        # Create data with gaps
        data = pd.DataFrame({
            'open': [100, np.nan, 102, 101],
            'high': [101, np.nan, 103, 102],
            'low': [99, np.nan, 101, 100],
            'close': [100.5, np.nan, 102.5, 101.5],
            'volume': [1000, np.nan, 1200, 1100]
        }, index=pd.date_range('2024-01-01 10:00', periods=4, freq='1min'))
        
        result = engine.extract_all_features(data)
        
        # Should handle missing data gracefully
        assert len(result) == len(data)
        assert not result.isnull().all().any()  # No column should be all NaN
    
    def test_timezone_handling(self):
        """Test timezone handling."""
        # Create engine with different timezone
        engine = IntradayFeatureEngine(timezone="America/Chicago")
        
        # Create data in different timezone
        utc_data = pd.DataFrame({
            'open': [100, 101], 'high': [101, 102], 'low': [99, 100],
            'close': [100.5, 101.5], 'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01 15:30', periods=2, freq='1min', tz='UTC'))
        
        result = engine.extract_all_features(utc_data)
        
        # Should handle timezone conversion gracefully
        assert len(result) == len(utc_data)


@pytest.mark.integration
class TestIntradayFeatureIntegration:
    """Integration tests for intraday features."""
    
    def test_realistic_trading_day(self):
        """Test with realistic full trading day data."""
        engine = IntradayFeatureEngine()
        
        # Create full trading day (390 minutes)
        start_time = pd.Timestamp('2024-01-02 09:30', tz='America/New_York')
        end_time = pd.Timestamp('2024-01-02 16:00', tz='America/New_York')
        
        timestamps = pd.date_range(start_time, end_time, freq='1min')
        
        # Realistic intraday pattern
        np.random.seed(42)
        base_price = 150.0
        
        # Simulate intraday drift and volatility patterns
        minutes_from_open = np.arange(len(timestamps))
        
        # U-shaped volatility (higher at open/close)
        vol_pattern = 0.02 * (1 + 0.5 * np.exp(-minutes_from_open/60) + 0.3 * np.exp(-(390-minutes_from_open)/60))
        
        returns = np.random.normal(0, vol_pattern)
        prices = base_price * np.exp(np.cumsum(returns * np.sqrt(1/390)))
        
        # Generate volume with realistic patterns
        volume_pattern = 1000000 * (1 + 0.8 * np.exp(-minutes_from_open/30) + 0.5 * np.exp(-(390-minutes_from_open)/30))
        volumes = volume_pattern * (1 + 0.3 * np.random.random(len(timestamps)))
        
        data = pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, len(prices)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(prices)))),
            'close': prices,
            'volume': volumes
        }, index=timestamps)
        
        data.iloc[0, 0] = base_price  # Set first open
        
        # Extract all features
        result = engine.extract_all_features(data)
        
        # Validate results
        assert len(result) == len(data)
        assert 'vwap' in result.columns
        assert 'atr_14' in result.columns
        assert 'stretch_pct' in result.columns
        
        # VWAP should be reasonable relative to prices
        vwap_end = result['vwap'].iloc[-1]
        price_range = data['close'].max() - data['close'].min()
        assert data['close'].min() <= vwap_end <= data['close'].max()
        
        # Stretch should show mean reversion patterns
        stretch = result['stretch_pct']
        assert stretch.std() > 0  # Should have some variation
        assert abs(stretch.mean()) < 5  # Should not be systematically biased


if __name__ == '__main__':
    pytest.main([__file__])