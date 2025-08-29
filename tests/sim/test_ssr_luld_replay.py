"""
test_ssr_luld_replay.py
=======================

TDD tests for DRQ-004: SSR/LULD Compliance and Historical Replay
These tests MUST be written first and will initially FAIL.
Implementation should make these tests pass.

CRITICAL: Zero regulatory violations required.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.compliance.ssr_luld_engine import (
        SSRComplianceEngine,
        LULDComplianceEngine, 
        HistoricalReplayEngine,
        ComplianceReport
    )
except ImportError:
    # These will fail initially - that's expected in TDD
    pass


class TestSSRCompliance:
    """Tests for Short Sale Restriction (SSR) compliance."""
    
    def test_ssr_trigger_detection(self):
        """Test that 10% decline correctly triggers SSR."""
        # This will fail initially until SSRComplianceEngine is implemented
        ssr_engine = SSRComplianceEngine()
        
        # Create price history with 10% decline
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = [100.0, 99.0, 98.0, 95.0, 90.0, 89.0, 88.0, 87.0, 86.0, 85.0]
        price_history = pd.Series(prices, index=dates)
        
        # Check SSR trigger on day when decline reaches 10%
        trigger_date = dates[4]  # Day when price hits 90.0 (10% decline from 100.0)
        ssr_triggered = ssr_engine.check_ssr_trigger(price_history.loc[:trigger_date])
        
        assert ssr_triggered, \
            f"SSR should trigger at 10% decline: {price_history.loc[:trigger_date].iloc[-1]/price_history.iloc[0] - 1:.1%}"
        
        # Should not trigger before 10% decline
        before_trigger = dates[3]  # Price at 95.0 (5% decline)
        ssr_not_triggered = ssr_engine.check_ssr_trigger(price_history.loc[:before_trigger])
        
        assert not ssr_not_triggered, \
            f"SSR should not trigger before 10% decline: {price_history.loc[:before_trigger].iloc[-1]/price_history.iloc[0] - 1:.1%}"
    
    def test_ssr_uptick_rule_enforcement(self):
        """Test that SSR enforces uptick-only execution."""
        ssr_engine = SSRComplianceEngine()
        
        # Simulate trading scenario with SSR active
        current_price = 90.0
        last_price = 89.5
        
        # Short sale order
        short_sale_order = {
            'side': 'sell',
            'quantity': 100,
            'order_type': 'market',
            'is_short_sale': True
        }
        
        # With SSR active, short sale should only be allowed on uptick
        ssr_active = True
        
        # Uptick scenario (current > last)
        uptick_allowed = ssr_engine.check_short_sale_allowed(
            short_sale_order, current_price, last_price, ssr_active
        )
        assert uptick_allowed, \
            "Short sale should be allowed on uptick when SSR active"
        
        # Downtick scenario (current < last) 
        downtick_allowed = ssr_engine.check_short_sale_allowed(
            short_sale_order, 89.0, 89.5, ssr_active
        )
        assert not downtick_allowed, \
            "Short sale should NOT be allowed on downtick when SSR active"
        
        # Same price (zero tick) - typically not allowed
        same_price_allowed = ssr_engine.check_short_sale_allowed(
            short_sale_order, 89.5, 89.5, ssr_active
        )
        assert not same_price_allowed, \
            "Short sale should NOT be allowed on zero tick when SSR active"
    
    def test_ssr_duration_and_expiry(self):
        """Test SSR duration and expiry rules."""
        ssr_engine = SSRComplianceEngine()
        
        # SSR typically remains in effect until end of next trading day
        trigger_date = datetime(2024, 1, 15, 14, 30)  # Monday 2:30 PM
        
        # Should be active for remainder of trigger day
        same_day_check = datetime(2024, 1, 15, 15, 45)
        assert ssr_engine.is_ssr_active(trigger_date, same_day_check), \
            "SSR should be active for remainder of trigger day"
        
        # Should be active for entire next trading day
        next_day_check = datetime(2024, 1, 16, 10, 30)  # Tuesday 10:30 AM
        assert ssr_engine.is_ssr_active(trigger_date, next_day_check), \
            "SSR should be active for entire next trading day"
        
        # Should expire after next trading day
        day_after_check = datetime(2024, 1, 17, 10, 30)  # Wednesday 10:30 AM
        assert not ssr_engine.is_ssr_active(trigger_date, day_after_check), \
            "SSR should expire after next trading day"
    
    def test_ssr_calculation_reference_price(self):
        """Test SSR calculation uses correct reference price."""
        ssr_engine = SSRComplianceEngine()
        
        # SSR calculation should use prior day's closing price as reference
        prior_close = 100.0
        current_prices = [99.0, 95.0, 90.0, 85.0]  # 1%, 5%, 10%, 15% declines
        
        for i, current_price in enumerate(current_prices):
            decline_pct = (current_price - prior_close) / prior_close
            expected_trigger = decline_pct <= -0.10  # 10% or more decline
            
            actual_trigger = ssr_engine.calculate_ssr_trigger(prior_close, current_price)
            
            assert actual_trigger == expected_trigger, \
                f"SSR trigger incorrect for {decline_pct:.1%} decline: expected {expected_trigger}, got {actual_trigger}"


class TestLULDCompliance:
    """Tests for Limit Up/Limit Down (LULD) compliance."""
    
    def test_luld_band_calculation(self):
        """Test LULD bands calculated correctly for all market conditions."""
        # This will fail initially until LULDComplianceEngine is implemented
        luld_engine = LULDComplianceEngine()
        
        reference_price = 50.0
        
        # Standard market hours - 5% bands
        standard_time = datetime(2024, 1, 15, 11, 30)  # 11:30 AM
        bands = luld_engine.calculate_luld_bands(reference_price, standard_time)
        
        expected_upper = reference_price * 1.05  # 5% above
        expected_lower = reference_price * 0.95  # 5% below
        
        assert abs(bands['upper'] - expected_upper) < 0.001, \
            f"LULD upper band incorrect: expected {expected_upper}, got {bands['upper']}"
        assert abs(bands['lower'] - expected_lower) < 0.001, \
            f"LULD lower band incorrect: expected {expected_lower}, got {bands['lower']}"
        
        # Test different reference prices
        for ref_price in [10.0, 100.0, 500.0]:
            bands = luld_engine.calculate_luld_bands(ref_price, standard_time)
            assert abs(bands['upper'] - ref_price * 1.05) < 0.001
            assert abs(bands['lower'] - ref_price * 0.95) < 0.001
    
    def test_luld_last_25_minutes_doubling(self):
        """Test that LULD bands double in last 25 minutes of trading."""
        luld_engine = LULDComplianceEngine()
        
        reference_price = 40.0
        
        # Normal market hours (before last 25 minutes) - standard 5% bands
        normal_time = datetime(2024, 1, 15, 15, 30)  # 3:30 PM (30 min before close)
        normal_bands = luld_engine.calculate_luld_bands(reference_price, normal_time)
        
        expected_normal_upper = reference_price * 1.05
        expected_normal_lower = reference_price * 0.95
        
        # Last 25 minutes (3:35 PM to 4:00 PM) - doubled to 10% bands
        last_25_time = datetime(2024, 1, 15, 15, 45)  # 3:45 PM (15 min before close)
        last_25_bands = luld_engine.calculate_luld_bands(reference_price, last_25_time)
        
        expected_last_25_upper = reference_price * 1.10  # 10% above
        expected_last_25_lower = reference_price * 0.90  # 10% below
        
        assert abs(last_25_bands['upper'] - expected_last_25_upper) < 0.001, \
            f"Last 25 min LULD upper should be 10%: expected {expected_last_25_upper}, got {last_25_bands['upper']}"
        assert abs(last_25_bands['lower'] - expected_last_25_lower) < 0.001, \
            f"Last 25 min LULD lower should be 10%: expected {expected_last_25_lower}, got {last_25_bands['lower']}"
        
        # Bands should be exactly double
        assert abs(last_25_bands['upper'] - normal_bands['upper']) > 0.01, \
            "Last 25 minutes bands should be different from normal bands"
    
    def test_luld_halt_trigger_conditions(self):
        """Test conditions that trigger LULD trading halts."""
        luld_engine = LULDComplianceEngine()
        
        reference_price = 60.0
        timestamp = datetime(2024, 1, 15, 12, 0)  # Noon - normal hours
        
        bands = luld_engine.calculate_luld_bands(reference_price, timestamp)
        
        # Trade within bands - should not halt
        within_band_price = 61.0  # Within 5% band
        halt_within = luld_engine.check_halt_condition(within_band_price, bands)
        assert not halt_within, \
            "Trade within LULD bands should not trigger halt"
        
        # Trade at band boundary - should not halt
        at_upper_band = bands['upper']
        halt_at_band = luld_engine.check_halt_condition(at_upper_band, bands)
        assert not halt_at_band, \
            "Trade at LULD band should not trigger halt"
        
        # Trade outside upper band - should halt
        above_upper_band = bands['upper'] + 0.01
        halt_above = luld_engine.check_halt_condition(above_upper_band, bands)
        assert halt_above, \
            "Trade above LULD upper band should trigger halt"
        
        # Trade outside lower band - should halt  
        below_lower_band = bands['lower'] - 0.01
        halt_below = luld_engine.check_halt_condition(below_lower_band, bands)
        assert halt_below, \
            "Trade below LULD lower band should trigger halt"
    
    def test_luld_reference_price_updates(self):
        """Test LULD reference price update mechanisms."""
        luld_engine = LULDComplianceEngine()
        
        initial_reference = 80.0
        
        # Reference price should update based on recent trading activity
        recent_trades = [
            {'price': 79.0, 'timestamp': datetime(2024, 1, 15, 10, 0)},
            {'price': 78.5, 'timestamp': datetime(2024, 1, 15, 10, 1)},
            {'price': 79.2, 'timestamp': datetime(2024, 1, 15, 10, 2)},
        ]
        
        updated_reference = luld_engine.update_reference_price(initial_reference, recent_trades)
        
        # Updated reference should reflect recent trading (specific algorithm TBD)
        assert updated_reference != initial_reference, \
            "Reference price should update based on recent trades"
        assert 75.0 <= updated_reference <= 85.0, \
            f"Updated reference price should be reasonable: {updated_reference}"


class TestHistoricalReplay:
    """Tests for historical compliance replay validation."""
    
    def test_historical_replay_zero_violations(self):
        """Test zero violations in comprehensive historical replay."""
        # This will fail initially until HistoricalReplayEngine is implemented
        replay_engine = HistoricalReplayEngine()
        
        # Create synthetic historical data that should be compliant
        start_date = '2023-01-01'
        end_date = '2023-01-31'  # One month of data
        
        # Generate compliant synthetic data
        synthetic_data = self._generate_compliant_historical_data(start_date, end_date)
        
        replay_results = replay_engine.replay_period(
            start_date=start_date,
            end_date=end_date,
            data=synthetic_data
        )
        
        # Should find zero violations in compliant synthetic data
        assert replay_results['ssr_violations'] == 0, \
            f"Should have zero SSR violations in compliant data, got {replay_results['ssr_violations']}"
        assert replay_results['luld_violations'] == 0, \
            f"Should have zero LULD violations in compliant data, got {replay_results['luld_violations']}"
        assert replay_results['total_trades_checked'] > 0, \
            "Should have checked some trades"
        
    def test_replay_detects_violations(self):
        """Test that replay engine detects actual violations."""
        replay_engine = HistoricalReplayEngine()
        
        # Create synthetic data with intentional violations
        start_date = '2024-01-01'
        end_date = '2024-01-07'  # One week
        
        # Generate data with known violations
        violating_data = self._generate_violating_historical_data(start_date, end_date)
        
        replay_results = replay_engine.replay_period(
            start_date=start_date,
            end_date=end_date,
            data=violating_data
        )
        
        # Should detect the intentional violations
        total_violations = replay_results['ssr_violations'] + replay_results['luld_violations']
        assert total_violations > 0, \
            f"Should detect violations in violating data, got {total_violations}"
        
        # Should provide details about violations
        assert 'violation_details' in replay_results, \
            "Should provide details about detected violations"
    
    def test_edge_case_handling(self):
        """Test edge case handling in compliance engine."""
        replay_engine = HistoricalReplayEngine()
        
        edge_cases = [
            # Market halt scenarios
            {'scenario': 'market_halt', 'expected_behavior': 'no_violations'},
            # Circuit breaker scenarios
            {'scenario': 'circuit_breaker', 'expected_behavior': 'halt_trading'},
            # Holiday trading scenarios
            {'scenario': 'holiday_trading', 'expected_behavior': 'modified_hours'},
            # After-hours trading
            {'scenario': 'after_hours', 'expected_behavior': 'different_rules'},
        ]
        
        for case in edge_cases:
            test_data = self._generate_edge_case_data(case['scenario'])
            
            try:
                results = replay_engine.test_edge_case_scenario(case['scenario'], test_data)
                assert results['handled_correctly'], \
                    f"Edge case {case['scenario']} should be handled correctly"
                assert results['no_violations'], \
                    f"Edge case {case['scenario']} should not generate violations"
            except NotImplementedError:
                # Edge case handling may not be fully implemented initially
                pytest.skip(f"Edge case handling for {case['scenario']} not yet implemented")
    
    def test_performance_large_dataset(self):
        """Test replay performance on large datasets."""
        replay_engine = HistoricalReplayEngine()
        
        # Test with larger dataset (but still reasonable for CI)
        start_date = '2023-01-01'
        end_date = '2023-02-28'  # Two months
        
        large_dataset = self._generate_compliant_historical_data(start_date, end_date, 
                                                               trades_per_day=1000)
        
        import time
        start_time = time.time()
        
        replay_results = replay_engine.replay_period(
            start_date=start_date,
            end_date=end_date,
            data=large_dataset
        )
        
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert elapsed_time < 30, \
            f"Historical replay should complete quickly, took {elapsed_time:.2f}s"
        
        # Should check substantial number of trades
        assert replay_results['total_trades_checked'] > 10000, \
            f"Should check many trades: {replay_results['total_trades_checked']}"
    
    def test_compliance_report_format(self):
        """Test compliance report format and completeness."""
        replay_engine = HistoricalReplayEngine()
        
        test_data = self._generate_compliant_historical_data('2024-01-01', '2024-01-05')
        
        replay_results = replay_engine.replay_period(
            start_date='2024-01-01',
            end_date='2024-01-05', 
            data=test_data
        )
        
        # Check required fields in compliance report
        required_fields = [
            'ssr_violations',
            'luld_violations', 
            'total_trades_checked',
            'start_date',
            'end_date',
            'compliance_percentage'
        ]
        
        for field in required_fields:
            assert field in replay_results, \
                f"Compliance report should include {field}"
        
        # Compliance percentage should be calculated correctly
        expected_compliance = 100.0 if (replay_results['ssr_violations'] == 0 and 
                                       replay_results['luld_violations'] == 0) else None
        if expected_compliance:
            assert abs(replay_results['compliance_percentage'] - expected_compliance) < 0.01, \
                f"Compliance percentage incorrect: {replay_results['compliance_percentage']}"
    
    # Helper methods for generating test data
    def _generate_compliant_historical_data(self, start_date: str, end_date: str, 
                                           trades_per_day: int = 100) -> Dict[str, Any]:
        """Generate synthetic historical data that should be compliant."""
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Generate realistic but compliant price movements and trades
        data = {
            'trades': [],
            'prices': {},
            'volumes': {}
        }
        
        for date in dates:
            # Generate daily price data with small, compliant movements
            base_price = 100.0
            daily_prices = []
            for hour in range(9, 16):  # 9 AM to 4 PM
                for minute in range(0, 60, 5):  # Every 5 minutes
                    # Small random walk that won't trigger SSR/LULD
                    price_change = np.random.normal(0, 0.001)  # 0.1% std dev
                    new_price = base_price * (1 + price_change)
                    daily_prices.append(new_price)
                    base_price = new_price
            
            data['prices'][date.strftime('%Y-%m-%d')] = daily_prices
            
            # Generate trade data
            for i in range(trades_per_day):
                trade_time = date + timedelta(
                    hours=np.random.randint(9, 16),
                    minutes=np.random.randint(0, 60),
                    seconds=np.random.randint(0, 60)
                )
                
                data['trades'].append({
                    'timestamp': trade_time,
                    'price': np.random.choice(daily_prices),
                    'volume': np.random.randint(100, 1000),
                    'side': np.random.choice(['buy', 'sell'])
                })
        
        return data
    
    def _generate_violating_historical_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Generate synthetic historical data with intentional violations."""
        data = self._generate_compliant_historical_data(start_date, end_date)
        
        # Add some intentional SSR violations
        violation_date = pd.Timestamp(start_date) + timedelta(days=1)
        
        # Create 10% price drop to trigger SSR
        data['trades'].append({
            'timestamp': violation_date.replace(hour=10),
            'price': 90.0,  # 10% below assumed 100.0 base
            'volume': 500,
            'side': 'sell'
        })
        
        # Add short sale on downtick while SSR active (violation)
        data['trades'].append({
            'timestamp': violation_date.replace(hour=11),
            'price': 89.5,  # Downtick
            'volume': 200,
            'side': 'sell',
            'is_short_sale': True  # This should be blocked by SSR
        })
        
        return data
    
    def _generate_edge_case_data(self, scenario: str) -> Dict[str, Any]:
        """Generate data for specific edge case scenarios."""
        base_data = self._generate_compliant_historical_data('2024-01-01', '2024-01-02', 50)
        
        if scenario == 'market_halt':
            # Add market halt event
            base_data['market_events'] = [
                {
                    'type': 'halt',
                    'timestamp': pd.Timestamp('2024-01-01 14:30:00'),
                    'duration_minutes': 15
                }
            ]
        elif scenario == 'circuit_breaker':
            # Add circuit breaker event
            base_data['market_events'] = [
                {
                    'type': 'circuit_breaker',
                    'timestamp': pd.Timestamp('2024-01-01 13:00:00'),
                    'level': 1,  # 7% market decline
                    'duration_minutes': 15
                }
            ]
        # Add other edge case scenarios as needed
        
        return base_data


# This will run when pytest is called and should initially FAIL
# Implementation should make these tests pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])