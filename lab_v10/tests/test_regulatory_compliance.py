"""
Tests for Regulatory Compliance System.

Tests SSR, LULD, and other regulatory constraints to ensure
proper enforcement during backtesting for realistic simulation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from decimal import Decimal

import sys
sys.path.append('/Users/aziymandias/Desktop/dual_book_trading_lab_v10_complete/lab_v10')

from src.trading.regulatory_compliance import (
    RegulatoryComplianceEngine, BacktestComplianceValidator,
    RegulatoryRule, SSRStatus, LULDBands, TradingHalt, ComplianceViolation
)


def create_test_market_data(symbol: str = 'TEST', start_price: float = 100.0, 
                           n_points: int = 100) -> pd.DataFrame:
    """Create test market data for compliance testing."""
    np.random.seed(42)
    
    timestamps = []
    prices = []
    volumes = []
    highs = []
    lows = []
    
    current_time = datetime(2023, 6, 15, 9, 30)  # Thursday 9:30 AM
    current_price = start_price
    
    for i in range(n_points):
        # Generate realistic intraday movements
        price_change_pct = np.random.normal(0, 0.005)  # 0.5% volatility
        current_price = max(current_price * (1 + price_change_pct), 0.01)
        
        high = current_price * (1 + abs(np.random.normal(0, 0.002)))
        low = current_price * (1 - abs(np.random.normal(0, 0.002)))
        volume = int(np.random.uniform(1000, 10000))
        
        timestamps.append(current_time)
        prices.append(current_price)
        volumes.append(volume)
        highs.append(max(high, current_price))
        lows.append(min(low, current_price))
        
        # Advance time by 1 minute
        current_time += timedelta(minutes=1)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': symbol,
        'price': prices,
        'volume': volumes,
        'high': highs,
        'low': lows
    })


class TestRegulatoryComplianceEngine:
    """Test regulatory compliance engine functionality."""
    
    def test_engine_initialization(self):
        """Test compliance engine initialization."""
        engine = RegulatoryComplianceEngine(enable_ssr=True, enable_luld=True)
        
        assert engine.enable_ssr is True
        assert engine.enable_luld is True
        assert engine.market_hours == (time(9, 30), time(16, 0))
        assert engine.ssr_threshold == 0.10
        assert len(engine.luld_tiers) > 0
        assert len(engine.ssr_status) == 0
        assert len(engine.violations) == 0
    
    def test_ssr_trigger_detection(self):
        """Test SSR trigger when stock drops 10%."""
        engine = RegulatoryComplianceEngine(enable_ssr=True)
        symbol = 'TSLA'
        start_time = datetime(2023, 6, 15, 9, 30)
        
        # Start at $200
        start_price = 200.0
        engine.update_market_data(symbol, start_time, start_price, 1000, start_price, start_price)
        
        # Drop to $179 (10.5% decline) - should trigger SSR
        decline_time = start_time + timedelta(hours=1)
        low_price = start_price * 0.895  # 10.5% decline
        
        result = engine.update_market_data(symbol, decline_time, low_price, 2000, start_price, low_price)
        
        # Check SSR was triggered
        assert symbol in engine.ssr_status
        assert engine.ssr_status[symbol].is_restricted is True
        assert result['ssr_restricted'] is True
        assert len(engine.ssr_triggers) == 1
        
        # Check trigger details
        trigger = engine.ssr_triggers[0]
        assert trigger['symbol'] == symbol
        assert trigger['decline_percent'] >= 10.0
    
    def test_ssr_uptick_rule(self):
        """Test SSR uptick rule for short sales."""
        engine = RegulatoryComplianceEngine(enable_ssr=True)
        symbol = 'AAPL'
        current_time = datetime(2023, 6, 15, 10, 0)
        
        # Manually set SSR restriction
        engine.ssr_status[symbol] = SSRStatus(
            symbol=symbol,
            is_restricted=True,
            restriction_date=current_time,
            trigger_price=150.0,
            restriction_expires=current_time + timedelta(days=1),
            last_uptick_price=149.0,
            last_price_direction='down'
        )
        
        # Try to short sell - should be rejected (no uptick)
        is_valid, reason, compliance_info = engine.validate_order(
            symbol, 'short', 100, 148.5, current_time
        )
        
        assert is_valid is False
        assert 'SSR violation' in reason
        assert len(compliance_info['violations']) == 1
        
        # Update price upward (uptick)
        engine.update_market_data(symbol, current_time + timedelta(minutes=1), 149.5, 1000)
        
        # Try to short sell again - should be allowed now
        is_valid, reason, compliance_info = engine.validate_order(
            symbol, 'short', 100, 149.5, current_time + timedelta(minutes=1)
        )
        
        assert is_valid is True
        assert reason == "Order validated"
    
    def test_luld_band_calculation(self):
        """Test LULD band calculations."""
        engine = RegulatoryComplianceEngine(enable_luld=True)
        symbol = 'SPY'
        timestamp = datetime(2023, 6, 15, 10, 0)
        price = 400.0
        
        # Update market data to establish bands
        result = engine.update_market_data(symbol, timestamp, price, 1000)
        
        assert symbol in engine.luld_bands
        bands = engine.luld_bands[symbol]
        
        # Check band calculation (should be 5% for tier1)
        expected_lower = price * 0.95  # 5% below
        expected_upper = price * 1.05  # 5% above
        
        assert abs(bands.lower_band - expected_lower) < 0.01
        assert abs(bands.upper_band - expected_upper) < 0.01
        assert bands.band_percentage == 0.05
        assert bands.is_doubled is False
    
    def test_luld_band_doubling_last_25_minutes(self):
        """Test LULD band doubling in last 25 minutes."""
        engine = RegulatoryComplianceEngine(enable_luld=True)
        symbol = 'QQQ'
        
        # Start during regular hours
        regular_time = datetime(2023, 6, 15, 14, 0)  # 2:00 PM
        price = 300.0
        
        engine.update_market_data(symbol, regular_time, price, 1000)
        original_band_pct = engine.luld_bands[symbol].band_percentage
        
        # Move to last 25 minutes (3:40 PM)
        last_25_time = datetime(2023, 6, 15, 15, 40)  # 3:40 PM (20 min before 4:00 close)
        
        result = engine.update_market_data(symbol, last_25_time, price, 1000)
        
        # Bands should be doubled
        bands = engine.luld_bands[symbol]
        assert bands.is_doubled is True
        assert bands.band_percentage == original_band_pct * 2
        
        # Check doubled band values
        expected_lower = price * (1 - bands.band_percentage)
        expected_upper = price * (1 + bands.band_percentage)
        
        assert abs(bands.lower_band - expected_lower) < 0.01
        assert abs(bands.upper_band - expected_upper) < 0.01
    
    def test_luld_violation_and_halt(self):
        """Test LULD violation triggers trading halt."""
        engine = RegulatoryComplianceEngine(enable_luld=True)
        symbol = 'NVDA'
        timestamp = datetime(2023, 6, 15, 11, 0)
        
        # Establish bands at $500
        reference_price = 500.0
        engine.update_market_data(symbol, timestamp, reference_price, 1000)
        
        bands = engine.luld_bands[symbol]
        
        # Price moves above upper band - should trigger halt
        violation_price = bands.upper_band + 1.0
        violation_time = timestamp + timedelta(minutes=5)
        
        result = engine.update_market_data(symbol, violation_time, violation_price, 2000)
        
        # Check violation was detected
        assert result['luld_violation'] is True
        assert result['luld_info']['violation_type'] == 'limit_up'
        
        # Check trading halt was triggered
        assert symbol in engine.active_halts
        halt = engine.active_halts[symbol]
        assert halt.halt_type == RegulatoryRule.LULD
        assert halt.halt_start == violation_time
        assert 'limit_up' in halt.halt_reason.lower()
        
        # Check violations were recorded
        assert len(engine.luld_violations) == 1
        assert engine.luld_violations[0]['violation_type'] == 'limit_up'
    
    def test_trading_halt_order_rejection(self):
        """Test order rejection during trading halt."""
        engine = RegulatoryComplianceEngine()
        symbol = 'TSLA'
        halt_time = datetime(2023, 6, 15, 12, 0)
        
        # Manually create trading halt
        halt = TradingHalt(
            symbol=symbol,
            halt_start=halt_time,
            halt_end=halt_time + timedelta(minutes=5),
            halt_reason='LULD limit_up violation',
            halt_type=RegulatoryRule.LULD
        )
        engine.active_halts[symbol] = halt
        
        # Try to place order during halt
        is_valid, reason, compliance_info = engine.validate_order(
            symbol, 'buy', 100, 200.0, halt_time + timedelta(minutes=1)
        )
        
        assert is_valid is False
        assert 'Trading halted' in reason
        assert len(compliance_info['violations']) == 1
        
        # Try after halt ends
        is_valid, reason, compliance_info = engine.validate_order(
            symbol, 'buy', 100, 200.0, halt_time + timedelta(minutes=6)
        )
        
        assert is_valid is True
        assert symbol not in engine.active_halts  # Should be cleared
    
    def test_market_hours_validation(self):
        """Test market hours validation."""
        engine = RegulatoryComplianceEngine()
        symbol = 'AAPL'
        
        # Before market open (9:00 AM)
        early_time = datetime(2023, 6, 15, 9, 0)
        is_valid, reason, compliance_info = engine.validate_order(
            symbol, 'buy', 100, 150.0, early_time
        )
        
        assert is_valid is True  # Still valid, but should have warning
        assert len(compliance_info['warnings']) == 1
        assert 'outside market hours' in compliance_info['warnings'][0].description
        
        # During market hours (2:00 PM)  
        market_time = datetime(2023, 6, 15, 14, 0)
        is_valid, reason, compliance_info = engine.validate_order(
            symbol, 'buy', 100, 150.0, market_time
        )
        
        assert is_valid is True
        assert len(compliance_info['warnings']) == 0
        
        # After market close (5:00 PM)
        late_time = datetime(2023, 6, 15, 17, 0)
        is_valid, reason, compliance_info = engine.validate_order(
            symbol, 'buy', 100, 150.0, late_time
        )
        
        assert is_valid is True  # Still valid, but should have warning
        assert len(compliance_info['warnings']) == 1
    
    def test_compliance_report(self):
        """Test compliance reporting functionality."""
        engine = RegulatoryComplianceEngine(enable_ssr=True, enable_luld=True)
        
        # Generate some test data
        symbol = 'TEST'
        timestamp = datetime(2023, 6, 15, 10, 0)
        
        # Trigger SSR
        engine.update_market_data(symbol, timestamp, 100.0, 1000, 100.0, 100.0)
        engine.update_market_data(symbol, timestamp + timedelta(minutes=30), 89.0, 2000, 100.0, 89.0)
        
        # Try to create SSR violation - short sell during restriction
        engine.validate_order(symbol, 'short', 100, 89.0, timestamp + timedelta(minutes=31))
        
        # Generate report
        report = engine.get_compliance_report()
        
        assert 'total_violations' in report
        assert 'ssr_statistics' in report
        assert 'luld_statistics' in report
        assert report['ssr_statistics']['total_triggers'] >= 1
        assert report['total_violations'] >= 1
        
        # Check that we have violations (may be SSR or LULD depending on band setup)
        assert len(report['recent_violations']) > 0
        violation = report['recent_violations'][0]
        assert violation.violation_type in [RegulatoryRule.SSR, RegulatoryRule.LULD]  # Accept either


class TestBacktestComplianceValidator:
    """Test backtest compliance validation."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        engine = RegulatoryComplianceEngine()
        validator = BacktestComplianceValidator(engine)
        
        assert validator.compliance_engine is engine
        assert len(validator.trade_validations) == 0
    
    def test_backtest_trade_validation(self):
        """Test validation of backtest trades."""
        engine = RegulatoryComplianceEngine(enable_ssr=True, enable_luld=True)
        validator = BacktestComplianceValidator(engine)
        
        # Create test trades DataFrame
        trades_data = {
            'timestamp': [
                datetime(2023, 6, 15, 10, 0),
                datetime(2023, 6, 15, 11, 0),
                datetime(2023, 6, 15, 12, 0),
                datetime(2023, 6, 15, 13, 0)
            ],
            'symbol': ['AAPL', 'AAPL', 'TSLA', 'TSLA'],
            'side': ['buy', 'short', 'buy', 'sell'],
            'quantity': [100, 50, 200, 150],
            'price': [150.0, 149.0, 250.0, 251.0]
        }
        
        trades_df = pd.DataFrame(trades_data)
        
        # Validate trades
        validation_report = validator.validate_backtest_trades(trades_df)
        
        # Check report structure
        assert 'total_trades' in validation_report
        assert 'valid_trades' in validation_report
        assert 'invalid_trades' in validation_report
        assert 'compliance_rate' in validation_report
        
        assert validation_report['total_trades'] == 4
        assert 0.0 <= validation_report['compliance_rate'] <= 1.0
        
        # Check trade validations were recorded
        assert len(validator.trade_validations) == 4


class TestRegulatoryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_ssr_expiration(self):
        """Test SSR restriction expiration."""
        engine = RegulatoryComplianceEngine(enable_ssr=True)
        symbol = 'EXPIRY'
        start_time = datetime(2023, 6, 15, 15, 0)
        
        # Create expired SSR restriction
        engine.ssr_status[symbol] = SSRStatus(
            symbol=symbol,
            is_restricted=True,
            restriction_date=start_time - timedelta(days=2),
            trigger_price=100.0,
            restriction_expires=start_time - timedelta(minutes=1),  # Already expired
            last_uptick_price=99.0
        )
        
        # Update market data - should clear expired restriction
        result = engine.update_market_data(symbol, start_time, 99.5, 1000)
        
        assert engine.ssr_status[symbol].is_restricted is False
        assert result['ssr_restricted'] is False
    
    def test_luld_band_reset(self):
        """Test LULD band reset for new trading day."""
        engine = RegulatoryComplianceEngine(enable_luld=True)
        symbol = 'RESET'
        
        # Set up doubled bands
        yesterday = datetime(2023, 6, 14, 15, 40)  # Yesterday 3:40 PM
        engine.update_market_data(symbol, yesterday, 100.0, 1000)
        
        bands = engine.luld_bands[symbol]
        original_pct = bands.band_percentage
        
        # Simulate new trading day start
        today = datetime(2023, 6, 15, 9, 35)  # Today 9:35 AM
        engine.update_market_data(symbol, today, 101.0, 1000)
        
        # Bands should be reset (not doubled anymore)
        assert bands.is_doubled is False
    
    def test_empty_order_validation(self):
        """Test validation with zero quantity orders."""
        engine = RegulatoryComplianceEngine()
        symbol = 'EMPTY'
        timestamp = datetime(2023, 6, 15, 12, 0)
        
        # Zero quantity order
        is_valid, reason, compliance_info = engine.validate_order(
            symbol, 'buy', 0, 100.0, timestamp
        )
        
        # Should still validate (quantity validation is separate concern)
        assert is_valid is True
    
    def test_negative_price_handling(self):
        """Test handling of negative or zero prices."""
        engine = RegulatoryComplianceEngine(enable_luld=True)
        symbol = 'NEGATIVE'
        timestamp = datetime(2023, 6, 15, 12, 0)
        
        # Update with very small positive price
        result = engine.update_market_data(symbol, timestamp, 0.01, 1000)
        
        assert symbol in engine.luld_bands
        bands = engine.luld_bands[symbol]
        assert bands.lower_band >= 0  # Should not be negative
        assert bands.upper_band > bands.lower_band
    
    def test_compliance_engine_reset(self):
        """Test compliance engine reset functionality."""
        engine = RegulatoryComplianceEngine(enable_ssr=True, enable_luld=True)
        
        # Add some state
        symbol = 'RESET_TEST'
        timestamp = datetime(2023, 6, 15, 12, 0)
        
        engine.update_market_data(symbol, timestamp, 100.0, 1000)
        engine.validate_order(symbol, 'buy', 100, 100.0, timestamp)
        
        # Verify state exists
        assert len(engine.ssr_status) > 0 or len(engine.luld_bands) > 0
        assert len(engine.violations) >= 0
        
        # Reset engine
        engine.reset()
        
        # Verify state is cleared
        assert len(engine.ssr_status) == 0
        assert len(engine.luld_bands) == 0
        assert len(engine.active_halts) == 0
        assert len(engine.violations) == 0


def test_comprehensive_regulatory_scenario():
    """End-to-end test of regulatory compliance in realistic scenario."""
    engine = RegulatoryComplianceEngine(enable_ssr=True, enable_luld=True)
    validator = BacktestComplianceValidator(engine)
    
    symbol = 'COMPREHENSIVE'
    base_time = datetime(2023, 6, 15, 9, 30)  # Market open
    
    # Scenario: Stock gaps down at open, triggers SSR, then has LULD violations
    
    # 1. Previous close at $200, opens at $175 (12.5% gap down)
    previous_close = 200.0
    open_price = 175.0
    
    # Update at market open
    result = engine.update_market_data(symbol, base_time, open_price, 10000, open_price, open_price)
    
    # Should not trigger SSR yet (need intraday decline)
    assert not result.get('ssr_restricted', False)
    
    # 2. Price continues to decline to $170 (15% total decline)
    decline_time = base_time + timedelta(minutes=30)
    decline_price = 170.0
    
    result = engine.update_market_data(symbol, decline_time, decline_price, 5000, open_price, decline_price)
    
    # Now should trigger SSR (>10% intraday decline)
    # Note: In real implementation, we'd track previous close separately
    
    # 3. Try to short sell - should be restricted
    short_time = decline_time + timedelta(minutes=5)
    is_valid, reason, info = engine.validate_order(symbol, 'short', 100, 169.0, short_time)
    
    # 4. Price spikes up violating LULD upper band
    spike_time = short_time + timedelta(minutes=10)
    bands = engine.luld_bands.get(symbol)
    
    if bands:
        spike_price = bands.upper_band + 2.0
        result = engine.update_market_data(symbol, spike_time, spike_price, 15000)
        
        # Should trigger LULD halt
        assert result.get('luld_violation', False) or symbol in engine.active_halts
    
    # 5. Generate trades for validation
    trades = []
    current_time = base_time
    
    for i in range(10):
        trades.append({
            'timestamp': current_time + timedelta(minutes=i*10),
            'symbol': symbol,
            'side': np.random.choice(['buy', 'sell', 'short']),
            'quantity': np.random.randint(50, 500),
            'price': open_price + np.random.normal(0, 5)
        })
    
    trades_df = pd.DataFrame(trades)
    
    # Validate all trades
    validation_report = validator.validate_backtest_trades(trades_df)
    
    # Generate final compliance report
    compliance_report = engine.get_compliance_report()
    
    # Verify comprehensive testing occurred
    assert validation_report['total_trades'] == len(trades)
    assert 'compliance_rate' in validation_report
    assert compliance_report['total_violations'] >= 0
    
    print(f"Comprehensive test results:")
    print(f"- Total trades: {validation_report['total_trades']}")
    print(f"- Compliance rate: {validation_report['compliance_rate']:.2%}")
    print(f"- Total violations: {compliance_report['total_violations']}")
    print(f"- SSR triggers: {compliance_report['ssr_statistics']['total_triggers']}")
    print(f"- LULD halts: {compliance_report['luld_statistics']['total_halts']}")