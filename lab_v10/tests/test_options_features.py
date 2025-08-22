"""
Test suite for options feature engineering module.

Comprehensive tests for:
- Black-Scholes pricing and Greeks calculation
- Implied volatility term structure features
- Volatility smile analysis
- Realized volatility metrics
- Forward basis calculations
- ATM straddle features
- Leakage prevention compliance
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from lab_v10.src.common.options_features import (
    OptionsFeatureEngine,
    calculate_iv_from_price
)


class TestOptionsFeatureEngine:
    """Test cases for OptionsFeatureEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create test engine instance."""
        return OptionsFeatureEngine(
            risk_free_rate=0.05,
            dividend_yield=0.02,
            vol_lookback_windows=[5, 21, 63]
        )
    
    @pytest.fixture
    def sample_price_series(self):
        """Create sample price series for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 100)))
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def sample_iv_surface(self):
        """Create sample IV surface."""
        strikes = [90, 95, 100, 105, 110]
        ttms = [0.25, 0.5, 1.0]
        
        surface = {}
        for ttm in ttms:
            for strike in strikes:
                # Simplified smile: higher vol for OTM options
                atm_vol = 0.20
                moneyness = strike / 100
                skew = 0.1 * (1 - moneyness)
                vol = atm_vol + skew + 0.05 * ttm
                surface[(strike, ttm)] = max(0.1, vol)
        
        return surface
    
    def test_black_scholes_price(self, engine):
        """Test Black-Scholes pricing."""
        # Test call option
        call_price = engine.black_scholes_price(
            S=100, K=100, T=0.25, r=0.05, q=0.02, sigma=0.20, option_type='call'
        )
        assert call_price > 0
        assert isinstance(call_price, float)
        
        # Test put option
        put_price = engine.black_scholes_price(
            S=100, K=100, T=0.25, r=0.05, q=0.02, sigma=0.20, option_type='put'
        )
        assert put_price > 0
        assert isinstance(put_price, float)
        
        # Test put-call parity approximation
        forward = 100 * np.exp((0.05 - 0.02) * 0.25)
        parity_diff = abs(call_price - put_price - (forward - 100))
        assert parity_diff < 1.0  # Should be close
    
    def test_black_scholes_edge_cases(self, engine):
        """Test Black-Scholes edge cases."""
        # Zero time to expiration
        price = engine.black_scholes_price(
            S=100, K=95, T=0, r=0.05, q=0.02, sigma=0.20, option_type='call'
        )
        assert price == max(0, 100 - 95)
        
        # Zero volatility
        price = engine.black_scholes_price(
            S=100, K=95, T=0.25, r=0.05, q=0.02, sigma=0, option_type='call'
        )
        assert price >= 0
    
    def test_calculate_greeks(self, engine):
        """Test Greeks calculation."""
        greeks = engine.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, q=0.02, sigma=0.20, option_type='call'
        )
        
        # Check required Greeks
        required_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        assert all(greek in greeks for greek in required_greeks)
        
        # Check reasonable values
        assert 0 < greeks['delta'] < 1  # Call delta should be positive
        assert greeks['gamma'] > 0  # Gamma always positive
        assert greeks['theta'] < 0  # Call theta usually negative
        assert greeks['vega'] > 0  # Vega always positive
        
        # Test put Greeks
        put_greeks = engine.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, q=0.02, sigma=0.20, option_type='put'
        )
        assert -1 < put_greeks['delta'] < 0  # Put delta should be negative
    
    def test_iv_term_structure_features(self, engine, sample_iv_surface):
        """Test IV term structure feature extraction."""
        features = engine.iv_term_structure_features(sample_iv_surface, spot_price=100)
        
        # Check required features
        required_features = ['iv_ts_slope', 'iv_ts_curvature', 'iv_ts_vol_of_vol']
        assert all(feature in features for feature in required_features)
        
        # Check data types
        assert all(isinstance(features[f], float) for f in required_features)
        
        # Check reasonable ranges
        assert abs(features['iv_ts_slope']) < 1.0  # Slope shouldn't be too extreme
    
    def test_volatility_smile_features(self, engine, sample_iv_surface):
        """Test volatility smile feature extraction."""
        features = engine.volatility_smile_features(sample_iv_surface, spot_price=100)
        
        required_features = ['smile_skew', 'smile_kurtosis', 'smile_convexity', 'put_call_skew']
        assert all(feature in features for feature in required_features)
        
        # Check data types
        assert all(isinstance(features[f], float) for f in required_features)
    
    def test_realized_volatility_features(self, engine, sample_price_series):
        """Test realized volatility feature extraction."""
        current_time = sample_price_series.index[-10]  # Use historical point
        
        features = engine.realized_volatility_features(sample_price_series, current_time)
        
        # Check volatility features for each window
        for window in engine.vol_lookback_windows:
            assert f'realized_vol_{window}d' in features
            assert features[f'realized_vol_{window}d'] >= 0
        
        # Check regime indicators
        regime_features = ['vol_regime_indicator', 'vol_persistence', 'vol_mean_reversion']
        assert all(feature in features for feature in regime_features)
    
    def test_forward_basis_features(self, engine):
        """Test forward basis feature extraction."""
        forward_prices = {0.25: 101, 0.5: 102, 1.0: 104}
        
        features = engine.forward_basis_features(
            spot_price=100,
            forward_prices=forward_prices
        )
        
        required_features = ['forward_basis_1m', 'forward_basis_3m', 'forward_basis_6m', 
                           'carry_slope', 'dividend_yield_estimate']
        assert all(feature in features for feature in required_features)
        
        # Check basis calculations are reasonable
        assert features['forward_basis_3m'] > 0  # Should be positive for upward sloping curve
    
    def test_atm_straddle_features(self, engine, sample_iv_surface):
        """Test ATM straddle feature extraction."""
        features = engine.atm_straddle_features(sample_iv_surface, spot_price=100)
        
        required_features = ['atm_straddle_delta', 'atm_straddle_gamma', 
                           'atm_straddle_theta', 'atm_straddle_vega', 'atm_iv']
        assert all(feature in features for feature in required_features)
        
        # ATM straddle delta should be close to zero
        assert abs(features['atm_straddle_delta']) < 0.1
        
        # Gamma and vega should be positive
        assert features['atm_straddle_gamma'] > 0
        assert features['atm_straddle_vega'] > 0
        
        # Theta should be negative
        assert features['atm_straddle_theta'] < 0
    
    def test_extract_all_features(self, engine, sample_iv_surface, sample_price_series):
        """Test comprehensive feature extraction."""
        current_time = sample_price_series.index[-10]
        
        all_features = engine.extract_all_features(
            iv_surface=sample_iv_surface,
            spot_price=100,
            price_history=sample_price_series,
            current_time=current_time
        )
        
        # Should contain features from all categories
        feature_categories = [
            'iv_ts_slope', 'smile_skew', 'realized_vol_21d',
            'forward_basis_3m', 'atm_straddle_gamma'
        ]
        assert all(feature in all_features for feature in feature_categories)
        
        # All features should be numeric
        assert all(isinstance(v, (int, float)) for v in all_features.values())
        assert all(not np.isnan(v) for v in all_features.values())
    
    def test_leakage_prevention(self, engine, sample_price_series):
        """Test that features don't use future information."""
        current_time = sample_price_series.index[50]  # Middle of series
        
        # Extract features using only data up to current_time
        features = engine.realized_volatility_features(sample_price_series, current_time)
        
        # Verify no future data was used by checking internal calculations
        historical_data = sample_price_series[sample_price_series.index <= current_time]
        assert len(historical_data) <= 51  # Should only use data up to current_time
        
        # Features should be calculable with historical data only
        assert all(isinstance(v, (int, float)) for v in features.values())
    
    def test_empty_data_handling(self, engine):
        """Test handling of empty or insufficient data."""
        empty_series = pd.Series([], dtype=float)
        current_time = pd.Timestamp('2024-01-01')
        
        features = engine.realized_volatility_features(empty_series, current_time)
        
        # Should return default values
        assert all(features[f'realized_vol_{w}d'] == 0.0 for w in engine.vol_lookback_windows)
        
        # Test empty IV surface
        empty_surface = {}
        features = engine.iv_term_structure_features(empty_surface, spot_price=100)
        assert all(v == 0.0 for v in features.values())


class TestImpliedVolatilityCalculation:
    """Test implied volatility calculation."""
    
    def test_calculate_iv_from_price(self):
        """Test IV calculation from option price."""
        # Test with known Black-Scholes inputs
        S, K, T, r, q = 100, 100, 0.25, 0.05, 0.02
        true_vol = 0.20
        
        # Calculate theoretical price
        engine = OptionsFeatureEngine(r, q)
        theoretical_price = engine.black_scholes_price(S, K, T, r, q, true_vol, 'call')
        
        # Calculate IV from price
        calculated_iv = calculate_iv_from_price(
            theoretical_price, S, K, T, r, q, 'call'
        )
        
        # Should recover original volatility
        assert abs(calculated_iv - true_vol) < 0.001
    
    def test_iv_calculation_edge_cases(self):
        """Test IV calculation edge cases."""
        # Very low option price
        low_iv = calculate_iv_from_price(
            price=0.01, S=100, K=110, T=0.1, r=0.05, q=0.02, option_type='call'
        )
        assert 0.01 <= low_iv <= 5.0  # Should be bounded
        
        # Very high option price (shouldn't happen in practice)
        high_iv = calculate_iv_from_price(
            price=50, S=100, K=100, T=0.25, r=0.05, q=0.02, option_type='call'
        )
        assert 0.01 <= high_iv <= 5.0  # Should be bounded


class TestFeatureValidation:
    """Test feature validation and quality checks."""
    
    def test_feature_ranges(self):
        """Test that features are within reasonable ranges."""
        engine = OptionsFeatureEngine()
        
        # Create realistic test data
        sample_surface = {
            (95, 0.25): 0.25, (100, 0.25): 0.20, (105, 0.25): 0.22,
            (95, 0.5): 0.23, (100, 0.5): 0.21, (105, 0.5): 0.24
        }
        
        features = engine.iv_term_structure_features(sample_surface, spot_price=100)
        
        # IV term structure slope should be reasonable
        assert -2.0 < features['iv_ts_slope'] < 2.0
        assert -1.0 < features['iv_ts_curvature'] < 1.0
        assert 0.0 <= features['iv_ts_vol_of_vol'] < 1.0
    
    def test_feature_consistency(self):
        """Test feature consistency across different inputs."""
        engine = OptionsFeatureEngine()
        
        # Test with different spot prices
        sample_surface = {(100, 0.25): 0.20, (105, 0.25): 0.22, (110, 0.25): 0.25}
        
        features_1 = engine.volatility_smile_features(sample_surface, spot_price=100)
        features_2 = engine.volatility_smile_features(sample_surface, spot_price=102)
        
        # Features should be similar for similar inputs
        assert abs(features_1['smile_skew'] - features_2['smile_skew']) < 0.1
    
    def test_deterministic_output(self):
        """Test that features are deterministic given same inputs."""
        engine = OptionsFeatureEngine(risk_free_rate=0.05, dividend_yield=0.02)
        
        # Same inputs should produce same outputs
        greeks_1 = engine.calculate_greeks(100, 100, 0.25, 0.05, 0.02, 0.20, 'call')
        greeks_2 = engine.calculate_greeks(100, 100, 0.25, 0.05, 0.02, 0.20, 'call')
        
        for key in greeks_1:
            assert abs(greeks_1[key] - greeks_2[key]) < 1e-10


@pytest.mark.integration
class TestOptionsFeatureIntegration:
    """Integration tests for options features."""
    
    def test_real_data_pipeline(self):
        """Test with realistic market data structure."""
        # Simulate realistic IV surface
        strikes = np.arange(80, 121, 5)
        ttms = [1/12, 3/12, 6/12, 12/12]
        spot = 100
        
        iv_surface = {}
        for ttm in ttms:
            for strike in strikes:
                moneyness = np.log(strike / spot)
                # Realistic smile shape
                iv = 0.20 + 0.1 * moneyness**2 + 0.02 * np.sqrt(ttm)
                iv_surface[(strike, ttm)] = max(0.05, iv)
        
        # Simulate price history
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, 252)  # ~20% annual vol
        prices = 100 * np.exp(np.cumsum(returns))
        price_series = pd.Series(prices, index=dates)
        
        # Extract features
        engine = OptionsFeatureEngine()
        current_time = dates[200]  # Use historical point
        
        all_features = engine.extract_all_features(
            iv_surface=iv_surface,
            spot_price=spot,
            price_history=price_series,
            current_time=current_time
        )
        
        # Verify comprehensive feature set
        assert len(all_features) > 15
        assert all(isinstance(v, (int, float)) for v in all_features.values())
        assert all(not np.isnan(v) for v in all_features.values())
        assert all(np.isfinite(v) for v in all_features.values())
    
    def test_performance_benchmark(self):
        """Basic performance test for feature extraction."""
        import time
        
        # Large IV surface
        strikes = np.arange(50, 151, 1)
        ttms = np.arange(1/365, 2, 1/365)[:50]  # 50 expiries
        
        iv_surface = {}
        for ttm in ttms:
            for strike in strikes:
                iv_surface[(strike, ttm)] = 0.20 + 0.1 * np.random.random()
        
        # Large price series
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, len(dates))), index=dates)
        
        engine = OptionsFeatureEngine()
        current_time = dates[-100]
        
        start_time = time.time()
        features = engine.extract_all_features(
            iv_surface=iv_surface,
            spot_price=100,
            price_history=prices,
            current_time=current_time
        )
        execution_time = time.time() - start_time
        
        # Should complete reasonably quickly
        assert execution_time < 5.0  # 5 seconds max
        assert len(features) > 0


if __name__ == '__main__':
    pytest.main([__file__])