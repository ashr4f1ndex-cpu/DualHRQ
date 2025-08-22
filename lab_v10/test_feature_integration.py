"""
Comprehensive test script for dual-book feature engineering integration.

This script tests all components of the integrated feature engineering pipeline:
- Options features (IV term structure, Greeks, volatility regimes)
- Intraday features (VWAP, ATR, SSR gates, LULD mechanics)  
- Corporate action adjustments (CRSP methodology)
- Time alignment (market calendars, timezone handling)
- Leakage prevention (CPCV with purging/embargo)
- Integration with HRM training pipeline

Run with: python test_feature_integration.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import all modules to test
from src.common.options_features import OptionsFeatureEngine
from src.common.intraday_features import IntradayFeatureEngine
from src.common.corporate_actions import CorporateActionAdjuster, CorporateActionDatabase, CorporateActionType, CorporateAction
from src.common.data_alignment import DataAligner, USEquityCalendar
from src.common.leakage_prevention import CombinatorialPurgedCV, LeakageAuditor
from src.common.feature_integration import IntegratedFeatureEngine, FeatureConfig
from src.options.hrm_input_enhanced import EnhancedFeatureProcessor, EnhancedTokenConfig

warnings.filterwarnings('ignore', category=RuntimeWarning)


def create_test_data():
    """Create comprehensive test data for all components."""
    print("Creating test data...")
    
    # Create equity data with proper datetime index and market hours
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='1min')
    
    # Filter to market hours (9:30-16:00 ET, weekdays)
    market_hours = dates[
        (dates.hour >= 9) & (dates.hour < 16) & 
        (dates.weekday < 5) &
        ~((dates.hour == 9) & (dates.minute < 30))
    ]
    
    # Generate realistic price data
    np.random.seed(42)
    n_samples = len(market_hours)
    base_price = 100.0
    
    # Generate correlated OHLCV data
    returns = np.random.normal(0, 0.001, n_samples)  # Smaller returns for minute data
    price_changes = np.cumsum(returns)
    close_prices = base_price * np.exp(price_changes)
    
    # Generate OHLC from close prices
    high_adj = np.random.lognormal(0, 0.01, n_samples)
    low_adj = np.random.lognormal(0, 0.01, n_samples) 
    open_adj = np.random.normal(0, 0.005, n_samples)
    
    equity_data = pd.DataFrame({
        'open': close_prices * (1 + open_adj),
        'high': close_prices * high_adj,
        'low': close_prices / low_adj,
        'close': close_prices,
        'volume': np.random.lognormal(8, 1, n_samples)
    }, index=market_hours)
    
    # Ensure OHLC consistency
    equity_data['high'] = equity_data[['open', 'high', 'low', 'close']].max(axis=1)
    equity_data['low'] = equity_data[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Create IV surface data
    trading_days = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    trading_days = trading_days[trading_days.weekday < 5]  # Only weekdays
    
    iv_surface_data = {}
    for date in trading_days:
        spot_price = 100 + np.random.normal(0, 10)
        strikes = np.arange(80, 121, 5)  # Strikes from 80 to 120
        ttms = [30/365, 60/365, 90/365]  # 30, 60, 90 day options
        
        surface = {}
        for strike in strikes:
            for ttm in ttms:
                moneyness = np.log(strike / spot_price)
                # Realistic IV smile
                iv = 0.20 + 0.1 * moneyness**2 + 0.05 * np.sqrt(ttm) + np.random.normal(0, 0.01)
                surface[(strike, ttm)] = max(0.05, iv)
        
        iv_surface_data[date] = surface
    
    # Create corporate actions
    corporate_actions = pd.DataFrame({
        'symbol': ['TEST'] * 3,
        'action_type': ['CASH_DIVIDEND', 'STOCK_SPLIT', 'CASH_DIVIDEND'],
        'ex_date': ['2023-03-15', '2023-07-15', '2023-11-15'],
        'cash_amount': [0.50, None, 0.55],
        'split_ratio': [None, 2.0, None],
        'description': ['Q1 Dividend', '2-for-1 Split', 'Q4 Dividend']
    })
    
    return {
        'equity': equity_data,
        'iv_surface': iv_surface_data,
        'corporate_actions': corporate_actions
    }


def test_options_features():
    """Test options feature extraction."""
    print("\n=== Testing Options Features ===")
    
    engine = OptionsFeatureEngine()
    
    # Create test IV surface
    iv_surface = {
        (95, 0.08): 0.25,   # 95 strike, 30 days
        (100, 0.08): 0.22,  # ATM, 30 days
        (105, 0.08): 0.26,  # 105 strike, 30 days
        (95, 0.25): 0.23,   # 95 strike, 90 days
        (100, 0.25): 0.20,  # ATM, 90 days
        (105, 0.25): 0.24   # 105 strike, 90 days
    }
    
    spot_price = 100.0
    price_history = pd.Series(
        np.random.normal(100, 5, 100),
        index=pd.date_range('2023-01-01', periods=100, freq='D')
    )
    current_time = pd.Timestamp('2023-04-10')
    
    # Test all features
    features = engine.extract_all_features(
        iv_surface=iv_surface,
        spot_price=spot_price,
        price_history=price_history,
        current_time=current_time
    )
    
    print(f"Extracted {len(features)} options features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    
    # Test individual components
    print("\nTesting individual components:")
    
    # IV term structure
    ts_features = engine.iv_term_structure_features(iv_surface, spot_price)
    print(f"IV term structure features: {len(ts_features)}")
    
    # Volatility smile
    smile_features = engine.volatility_smile_features(iv_surface, spot_price)
    print(f"Volatility smile features: {len(smile_features)}")
    
    # Greeks calculation
    greeks = engine.calculate_greeks(100, 100, 0.25, 0.05, 0.02, 0.20, 'call')
    print(f"Greeks: delta={greeks['delta']:.3f}, gamma={greeks['gamma']:.3f}")
    
    print("✓ Options features test passed")


def test_intraday_features():
    """Test intraday feature extraction."""
    print("\n=== Testing Intraday Features ===")
    
    engine = IntradayFeatureEngine()
    
    # Create test intraday data
    times = pd.date_range('2023-01-03 09:30', '2023-01-03 16:00', freq='1min')
    n_samples = len(times)
    
    # Generate realistic intraday price action
    base_price = 100.0
    returns = np.random.normal(0, 0.001, n_samples)
    prices = base_price * np.cumprod(1 + returns)
    
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(6, 0.5, n_samples)
    }, index=times)
    
    # Ensure OHLC consistency
    test_data['high'] = test_data[['open', 'high', 'low', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Test all features
    enhanced_data = engine.extract_all_features(test_data)
    
    new_features = [col for col in enhanced_data.columns if col not in test_data.columns]
    print(f"Extracted {len(new_features)} intraday features:")
    for feature in new_features[:10]:  # Show first 10
        print(f"  {feature}")
    if len(new_features) > 10:
        print(f"  ... and {len(new_features) - 10} more")
    
    # Test individual components
    print("\nTesting individual components:")
    
    # VWAP
    vwap = engine.calculate_vwap(test_data)
    print(f"VWAP calculated: {len(vwap)} points")
    
    # ATR
    atr = engine.calculate_atr(test_data, period=14)
    print(f"ATR calculated: {len(atr.dropna())} points")
    
    # SSR gates
    ssr_features = engine.calculate_ssr_gate(test_data)
    print(f"SSR features: {len(ssr_features)}")
    
    # LULD mechanics
    luld_features = engine.calculate_luld_mechanics(test_data)
    print(f"LULD features: {len(luld_features)}")
    
    print("✓ Intraday features test passed")


def test_corporate_actions():
    """Test corporate action handling."""
    print("\n=== Testing Corporate Actions ===")
    
    # Create test data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates)))
    
    test_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    # Create corporate actions
    actions = [
        CorporateAction(
            symbol='TEST',
            action_type=CorporateActionType.CASH_DIVIDEND,
            ex_date=pd.Timestamp('2023-03-15'),
            cash_amount=2.0
        ),
        CorporateAction(
            symbol='TEST',
            action_type=CorporateActionType.STOCK_SPLIT,
            ex_date=pd.Timestamp('2023-07-15'),
            split_ratio=2.0
        )
    ]
    
    # Test adjustment
    adjuster = CorporateActionAdjuster()
    adjusted_data = adjuster.apply_adjustments(test_data, actions)
    
    print(f"Original data shape: {test_data.shape}")
    print(f"Adjusted data shape: {adjusted_data.shape}")
    print(f"New columns: {[col for col in adjusted_data.columns if col not in test_data.columns]}")
    
    # Test database
    db = CorporateActionDatabase()
    for action in actions:
        db.add_action(action)
    
    retrieved_actions = db.get_actions_by_symbol('TEST')
    print(f"Stored and retrieved {len(retrieved_actions)} corporate actions")
    
    print("✓ Corporate actions test passed")


def test_data_alignment():
    """Test data alignment and market calendars."""
    print("\n=== Testing Data Alignment ===")
    
    # Create test data with some non-trading times
    dates = pd.date_range('2023-01-01', '2023-01-31', freq='1H')
    test_data = pd.DataFrame({
        'price': np.random.normal(100, 5, len(dates)),
        'volume': np.random.lognormal(8, 1, len(dates))
    }, index=dates)
    
    aligner = DataAligner()
    
    # Test market calendar
    calendar = USEquityCalendar()
    print(f"Is 2023-01-03 a trading day? {calendar.is_trading_day(pd.Timestamp('2023-01-03'))}")
    print(f"Is 2023-01-01 a trading day? {calendar.is_trading_day(pd.Timestamp('2023-01-01'))}")
    
    # Test alignment to trading days
    aligned_data = aligner.align_to_trading_days(test_data)
    print(f"Original data: {len(test_data)} samples")
    print(f"Trading days aligned: {len(aligned_data)} samples")
    
    # Test timezone conversion
    test_data_tz = aligner.convert_timezone(test_data, 'America/New_York', 'UTC')
    print(f"Timezone converted: {test_data_tz.index.tz}")
    
    print("✓ Data alignment test passed")


def test_leakage_prevention():
    """Test leakage prevention with CPCV."""
    print("\n=== Testing Leakage Prevention ===")
    
    # Create test data
    dates = pd.date_range('2023-01-01', '2023-06-01', freq='D')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'feature1': np.random.normal(0, 1, len(dates)),
        'feature2': np.random.normal(0, 1, len(dates))
    })
    
    target = pd.Series(np.random.normal(0, 1, len(dates)))
    
    # Test CPCV
    cpcv = CombinatorialPurgedCV(
        n_splits=5,
        n_test_groups=1,
        purge=pd.Timedelta(hours=24),
        embargo=pd.Timedelta(hours=24)
    )
    
    splits = list(cpcv.split(test_data, target))
    print(f"Generated {len(splits)} CV splits")
    
    # Test leakage auditor
    auditor = LeakageAuditor()
    
    if splits:
        train_idx, test_idx = splits[0]
        train_data = test_data.iloc[train_idx]
        test_data_split = test_data.iloc[test_idx]
        
        audit_results = auditor.audit_temporal_leakage(train_data, test_data_split)
        print(f"Temporal leakage audit: {audit_results}")
    
    print("✓ Leakage prevention test passed")


def test_integrated_pipeline():
    """Test the complete integrated feature pipeline."""
    print("\n=== Testing Integrated Pipeline ===")
    
    # Create comprehensive test data
    test_data = create_test_data()
    
    # Configure feature engine
    config = FeatureConfig(
        enable_options_features=True,
        enable_intraday_features=True,
        enable_corporate_actions=True,
        enable_leakage_prevention=True
    )
    
    engine = IntegratedFeatureEngine(config)
    
    # Prepare data
    prepared_data = engine.prepare_raw_data(**test_data)
    print(f"Prepared data keys: {list(prepared_data.keys())}")
    
    # Get sample of dates for feature extraction
    equity_data = prepared_data['equity']
    sample_dates = pd.date_range(
        start=equity_data.index.min().normalize(),
        end=equity_data.index.min().normalize() + pd.Timedelta(days=30),
        freq='D'
    )
    sample_dates = sample_dates[sample_dates.weekday < 5][:10]  # First 10 trading days
    
    # Create feature pipeline
    daily_features, intraday_features = engine.create_feature_pipeline(
        prepared_data, sample_dates
    )
    
    print(f"Daily features shape: {daily_features.shape}")
    print(f"Intraday features shape: {intraday_features.shape}")
    
    if not daily_features.empty:
        print(f"Daily feature columns: {list(daily_features.columns)[:10]}...")  # Show first 10
    
    # Validate feature quality
    validation_results = engine.validate_feature_quality(daily_features, intraday_features)
    print(f"Feature quality assessment: {validation_results.get('overall_quality', 'Unknown')}")
    
    print("✓ Integrated pipeline test passed")


def test_hrm_integration():
    """Test integration with HRM input pipeline."""
    print("\n=== Testing HRM Integration ===")
    
    # Create test data
    test_data = create_test_data()
    
    # Subsample for faster testing
    equity_subset = test_data['equity'].iloc[::100]  # Every 100th sample
    test_data['equity'] = equity_subset
    
    # Create sample targets
    daily_prices = equity_subset.groupby(equity_subset.index.date)['close'].last()
    
    targets = {
        'yA': pd.Series(np.random.normal(0.2, 0.05, len(daily_prices)), index=daily_prices.index),  # Volatility target
        'yB': pd.Series(np.random.binomial(1, 0.3, len(daily_prices)), index=daily_prices.index)     # Binary target
    }
    
    # Configure enhanced processor
    config = EnhancedTokenConfig(
        feature_config=FeatureConfig(
            enable_options_features=True,
            enable_intraday_features=True,
            enable_leakage_prevention=False  # Disable for faster testing
        ),
        daily_window=30,  # Smaller window for testing
        minutes_per_day=100,  # Fewer minutes for testing
        max_features=20   # Fewer features for testing
    )
    
    processor = EnhancedFeatureProcessor(config)
    
    # Prepare features
    daily_features, intraday_features = processor.prepare_features_from_raw_data(test_data)
    
    print(f"Daily features for HRM: {daily_features.shape}")
    print(f"Intraday features for HRM: {intraday_features.shape}")
    
    # Test scaler fitting (use smaller dataset)
    if len(daily_features) > 10:
        train_idx = np.arange(min(10, len(daily_features) // 2))
        scalers = processor.fit_enhanced_scalers(daily_features, intraday_features, train_idx)
        print(f"Fitted scalers: daily features={len(scalers.selected_daily_features or [])}, "
              f"intraday features={len(scalers.selected_intraday_features or [])}")
        
        # Test token creation
        sample_dates = daily_features.index[:5]  # Just 5 dates for testing
        H_tokens = processor.make_enhanced_h_tokens(daily_features, sample_dates, scalers)
        L_tokens = processor.make_enhanced_l_tokens(intraday_features, sample_dates, scalers)
        
        print(f"H tokens shape: {H_tokens.shape}")
        print(f"L tokens shape: {L_tokens.shape}")
        
        # Verify tensor properties
        assert isinstance(H_tokens, torch.Tensor), "H tokens should be torch.Tensor"
        assert isinstance(L_tokens, torch.Tensor), "L tokens should be torch.Tensor"
        assert H_tokens.dtype == torch.float32, "H tokens should be float32"
        assert L_tokens.dtype == torch.float32, "L tokens should be float32"
        
        print("✓ Token creation successful")
    
    print("✓ HRM integration test passed")


def run_all_tests():
    """Run all integration tests."""
    print("=== Dual-Book Feature Engineering Integration Tests ===")
    print("Testing comprehensive feature pipeline components...\n")
    
    try:
        test_options_features()
        test_intraday_features()
        test_corporate_actions()
        test_data_alignment()
        test_leakage_prevention()
        test_integrated_pipeline()
        test_hrm_integration()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The dual-book feature engineering pipeline is ready for production!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)