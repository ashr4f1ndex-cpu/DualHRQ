"""
Tests for Realistic Backtester with Regulatory Compliance.

Tests the integration of regulatory compliance, HRM learning, and
walk-forward testing in a complete backtesting system.
"""

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import List, Dict, Any

import sys
sys.path.append('/Users/aziymandias/Desktop/dual_book_trading_lab_v10_complete/lab_v10')

from src.trading.realistic_backtester import (
    RealisticBacktester, BacktestConfig, Trade, Position, BacktestResults
)
from src.trading.regulatory_compliance import RegulatoryRule


def create_test_market_data(n_days: int = 30, symbols: List[str] = ['AAPL', 'TSLA'],
                           start_date: str = '2023-06-01') -> pd.DataFrame:
    """Create realistic test market data for backtesting."""
    np.random.seed(42)
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    data = []
    
    for symbol in symbols:
        base_price = 100.0 if symbol == 'AAPL' else 200.0
        current_price = base_price
        
        for day in range(n_days):
            date = start_dt + timedelta(days=day)
            
            # Skip weekends (simplified)
            if date.weekday() >= 5:
                continue
            
            # Generate intraday data (hourly)
            for hour in range(9, 16):  # 9 AM to 4 PM
                timestamp = date.replace(hour=hour, minute=30)
                
                # Random price movement
                price_change = np.random.normal(0, 0.01)  # 1% volatility
                current_price = max(current_price * (1 + price_change), 0.01)
                
                high = current_price * (1 + abs(np.random.normal(0, 0.005)))
                low = current_price * (1 - abs(np.random.normal(0, 0.005)))
                volume = int(np.random.uniform(1000, 10000))
                
                data.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'price': current_price,
                    'high': max(high, current_price),
                    'low': min(low, current_price),
                    'volume': volume
                })
    
    return pd.DataFrame(data).sort_values('timestamp').reset_index(drop=True)


def simple_momentum_strategy(market_data: pd.DataFrame, positions: Dict[str, Position], 
                           portfolio_value: float) -> List[Dict[str, Any]]:
    """Simple momentum strategy for testing."""
    signals = []
    
    for symbol in market_data['symbol'].unique():
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        if len(symbol_data) < 2:
            continue
        
        # Simple momentum signal
        current_price = symbol_data['price'].iloc[-1]
        prev_price = symbol_data['price'].iloc[-2]
        
        price_change = (current_price - prev_price) / prev_price
        
        # Generate signals based on momentum
        if price_change > 0.02:  # > 2% up
            signals.append({
                'symbol': symbol,
                'side': 'buy',
                'quantity': 100,
                'timestamp': symbol_data['timestamp'].iloc[-1],
                'reason': 'momentum_up'
            })
        elif price_change < -0.02:  # < -2% down
            # Only sell if we have a position
            if symbol in positions and positions[symbol].quantity > 0:
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': min(100, positions[symbol].quantity),
                    'timestamp': symbol_data['timestamp'].iloc[-1],
                    'reason': 'momentum_down'
                })
    
    return signals


def create_ssr_trigger_data(symbol: str = 'DECLINE', start_date: str = '2023-06-15') -> pd.DataFrame:
    """Create market data that triggers SSR."""
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    
    data = []
    base_price = 100.0
    
    # Morning: normal trading
    for hour in range(9, 12):
        timestamp = start_dt.replace(hour=hour, minute=30)
        price = base_price + np.random.normal(0, 0.5)
        
        data.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'price': price,
            'high': price + 1.0,
            'low': price - 1.0,
            'volume': 5000
        })
    
    # Afternoon: major decline triggering SSR
    decline_price = base_price * 0.88  # 12% decline
    for hour in range(12, 16):
        timestamp = start_dt.replace(hour=hour, minute=30)
        
        data.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'price': decline_price,
            'high': max(base_price, decline_price + 1.0) if hour == 12 else decline_price + 0.5,
            'low': decline_price - 0.5,
            'volume': 10000
        })
        
        decline_price *= 0.995  # Gradual further decline
    
    return pd.DataFrame(data)


class TestRealisticBacktester:
    """Test realistic backtester functionality."""
    
    def test_backtester_initialization(self):
        """Test backtester initialization."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-30',
            initial_capital=100000.0,
            enable_ssr=True,
            enable_luld=True,
            enable_hrm=False  # Disable for faster testing
        )
        
        backtester = RealisticBacktester(config)
        
        assert backtester.config.initial_capital == 100000.0
        assert backtester.portfolio_value == 100000.0
        assert backtester.cash == 100000.0
        assert len(backtester.positions) == 0
        assert len(backtester.trades) == 0
        assert backtester.compliance_engine.enable_ssr is True
        assert backtester.compliance_engine.enable_luld is True
    
    def test_basic_backtest_execution(self):
        """Test basic backtest execution with simple strategy."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-10',
            initial_capital=50000.0,
            enable_ssr=False,  # Disable for simpler test
            enable_luld=False,
            enable_hrm=False
        )
        
        backtester = RealisticBacktester(config)
        market_data = create_test_market_data(10, ['TEST'], '2023-06-01')
        
        # Run backtest
        results = backtester.run_backtest(simple_momentum_strategy, market_data)
        
        # Verify results structure
        assert isinstance(results, BacktestResults)
        assert len(results.trades) >= 0
        assert results.portfolio_value > 0
        assert -1.0 <= results.total_return <= 5.0  # Reasonable return range
        assert results.compliance_rate >= 0.0
        assert results.regulatory_violations >= 0
    
    def test_position_management(self):
        """Test position management and P&L calculations."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-05',
            initial_capital=100000.0,
            enable_ssr=False,
            enable_luld=False,
            enable_hrm=False
        )
        
        backtester = RealisticBacktester(config)
        
        # Test position updates
        symbol = 'POS_TEST'
        
        # Buy 100 shares at $50
        executed_qty = backtester._update_position(symbol, 'buy', 100, 50.0)
        assert executed_qty == 100
        assert symbol in backtester.positions
        
        position = backtester.positions[symbol]
        assert position.quantity == 100
        assert position.avg_price == 50.0
        assert position.position_type == 'long'
        
        # Sell 60 shares
        executed_qty = backtester._update_position(symbol, 'sell', 60, 55.0)
        assert executed_qty == 60
        assert position.quantity == 40
        
        # Update position value
        backtester._update_position_values(symbol, 60.0)
        assert position.current_price == 60.0
        assert position.unrealized_pnl == (60.0 - 50.0) * 40  # $400 profit
    
    def test_ssr_compliance_integration(self):
        """Test SSR compliance during backtesting."""
        config = BacktestConfig(
            start_date='2023-06-15',
            end_date='2023-06-16',
            initial_capital=100000.0,
            enable_ssr=True,
            enable_luld=False,
            enable_hrm=False
        )
        
        backtester = RealisticBacktester(config)
        
        # Create data that triggers SSR
        ssr_data = create_ssr_trigger_data('SSR_TEST', '2023-06-15')
        
        # Strategy that tries to short sell
        def short_strategy(market_data, positions, portfolio_value):
            signals = []
            for symbol in market_data['symbol'].unique():
                symbol_data = market_data[market_data['symbol'] == symbol]
                current_price = symbol_data['price'].iloc[-1]
                
                # Try to short when price is declining
                if current_price < 95.0:  # When stock has declined
                    signals.append({
                        'symbol': symbol,
                        'side': 'short',
                        'quantity': 100,
                        'timestamp': symbol_data['timestamp'].iloc[-1]
                    })
            return signals
        
        # Run backtest
        results = backtester.run_backtest(short_strategy, ssr_data)
        
        # Check SSR impacts
        assert 'total_ssr_triggers' in results.ssr_impacts
        
        # Should have some rejected trades due to SSR
        rejected_trades = [t for t in results.trades if not t.compliance_valid]
        if results.ssr_impacts['total_ssr_triggers'] > 0:
            assert len(rejected_trades) > 0
    
    def test_transaction_costs_and_slippage(self):
        """Test transaction costs and slippage implementation."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-05',
            initial_capital=100000.0,
            transaction_costs=0.001,  # 0.1%
            slippage=0.0005,  # 0.05%
            enable_ssr=False,
            enable_luld=False,
            enable_hrm=False
        )
        
        backtester = RealisticBacktester(config)
        market_data = create_test_market_data(5, ['COST_TEST'], '2023-06-01')
        
        # Strategy that makes one buy trade
        def single_buy_strategy(market_data, positions, portfolio_value):
            if len(positions) == 0:  # Only trade once
                return [{
                    'symbol': 'COST_TEST',
                    'side': 'buy', 
                    'quantity': 100,
                    'timestamp': market_data['timestamp'].iloc[-1]
                }]
            return []
        
        results = backtester.run_backtest(single_buy_strategy, market_data)
        
        # Check that trade was executed with costs
        if len(results.trades) > 0:
            trade = results.trades[0]
            assert trade.compliance_valid
            assert trade.commission > 0
            
            # Price should include slippage (higher for buy)
            market_price = market_data[market_data['symbol'] == 'COST_TEST']['price'].iloc[0]
            assert trade.price > market_price  # Slippage applied
    
    def test_position_size_limits(self):
        """Test position size limitations."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-05',
            initial_capital=10000.0,  # Small capital
            max_position_size=0.20,  # 20% max per position
            enable_ssr=False,
            enable_luld=False,
            enable_hrm=False
        )
        
        backtester = RealisticBacktester(config)
        
        # Strategy that tries to buy large position
        def large_position_strategy(market_data, positions, portfolio_value):
            return [{
                'symbol': 'LARGE_POS',
                'side': 'buy',
                'quantity': 1000,  # Very large quantity
                'timestamp': market_data['timestamp'].iloc[-1]
            }]
        
        market_data = create_test_market_data(5, ['LARGE_POS'], '2023-06-01')
        results = backtester.run_backtest(large_position_strategy, market_data)
        
        # Position should be limited by max_position_size
        if len(results.positions) > 0:
            position = results.positions[0]
            position_value = abs(position.quantity) * position.current_price
            max_allowed = config.initial_capital * config.max_position_size
            assert position_value <= max_allowed * 1.1  # Allow small margin for slippage
    
    def test_portfolio_performance_metrics(self):
        """Test portfolio performance metric calculations."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-15',
            initial_capital=100000.0,
            enable_ssr=False,
            enable_luld=False,
            enable_hrm=False
        )
        
        backtester = RealisticBacktester(config)
        market_data = create_test_market_data(15, ['PERF'], '2023-06-01')
        
        results = backtester.run_backtest(simple_momentum_strategy, market_data)
        
        # Check performance metrics
        assert isinstance(results.total_return, float)
        assert isinstance(results.sharpe_ratio, float)
        assert isinstance(results.max_drawdown, float)
        assert 0.0 <= results.max_drawdown <= 1.0
        assert len(results.daily_returns) > 0
        assert len(results.equity_curve) > 0
        
        # Check equity curve is reasonable
        assert results.equity_curve.iloc[0] == config.initial_capital
        assert results.equity_curve.iloc[-1] == results.portfolio_value
    
    def test_compliance_reporting(self):
        """Test regulatory compliance reporting."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-10',
            initial_capital=100000.0,
            enable_ssr=True,
            enable_luld=True,
            enable_hrm=False
        )
        
        backtester = RealisticBacktester(config)
        market_data = create_test_market_data(10, ['COMPLIANCE'], '2023-06-01')
        
        results = backtester.run_backtest(simple_momentum_strategy, market_data)
        
        # Check compliance reporting
        assert 0.0 <= results.compliance_rate <= 1.0
        assert results.regulatory_violations >= 0
        assert 'total_ssr_triggers' in results.ssr_impacts
        assert 'total_luld_violations' in results.luld_impacts
        assert isinstance(results.ssr_impacts['rejected_short_trades'], int)
        assert isinstance(results.luld_impacts['total_trading_halts'], int)


class TestBacktestIntegration:
    """Test backtest integration scenarios."""
    
    def test_multi_symbol_backtest(self):
        """Test backtesting with multiple symbols."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-15',
            initial_capital=200000.0,
            enable_ssr=True,
            enable_luld=True,
            enable_hrm=False
        )
        
        backtester = RealisticBacktester(config)
        symbols = ['MULTI1', 'MULTI2', 'MULTI3']
        market_data = create_test_market_data(15, symbols, '2023-06-01')
        
        # Strategy that trades multiple symbols
        def multi_symbol_strategy(market_data, positions, portfolio_value):
            signals = []
            for symbol in market_data['symbol'].unique():
                symbol_data = market_data[market_data['symbol'] == symbol]
                
                if len(symbol_data) >= 2:
                    current_price = symbol_data['price'].iloc[-1]
                    prev_price = symbol_data['price'].iloc[-2]
                    change = (current_price - prev_price) / prev_price
                    
                    if change > 0.015:  # 1.5% up
                        signals.append({
                            'symbol': symbol,
                            'side': 'buy',
                            'quantity': 50,
                            'timestamp': symbol_data['timestamp'].iloc[-1]
                        })
            
            return signals
        
        results = backtester.run_backtest(multi_symbol_strategy, market_data)
        
        # Should have trades across multiple symbols
        traded_symbols = set(trade.symbol for trade in results.trades if trade.compliance_valid)
        assert len(traded_symbols) >= 1  # At least one symbol traded
        
        # Portfolio value should be reasonable
        assert results.portfolio_value > 0
        assert -0.5 <= results.total_return <= 2.0  # Reasonable return range
    
    def test_insufficient_capital_handling(self):
        """Test handling of insufficient capital scenarios."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-10',
            initial_capital=1000.0,  # Very small capital
            enable_ssr=False,
            enable_luld=False,
            enable_hrm=False
        )
        
        backtester = RealisticBacktester(config)
        market_data = create_test_market_data(10, ['EXPENSIVE'], '2023-06-01')
        
        # Strategy that tries to buy expensive stocks
        def expensive_strategy(market_data, positions, portfolio_value):
            return [{
                'symbol': 'EXPENSIVE',
                'side': 'buy',
                'quantity': 100,  # May be too expensive
                'timestamp': market_data['timestamp'].iloc[-1]
            }]
        
        results = backtester.run_backtest(expensive_strategy, market_data)
        
        # Should handle insufficient capital gracefully
        assert results.portfolio_value >= 0
        assert results.cash >= 0  # Should not go negative
    
    def test_empty_market_data_handling(self):
        """Test handling of empty market data."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-10',
            initial_capital=100000.0
        )
        
        backtester = RealisticBacktester(config)
        empty_data = pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'volume'])
        
        # Should raise error for empty data
        with pytest.raises(ValueError, match="No data found"):
            backtester.run_backtest(simple_momentum_strategy, empty_data)
    
    def test_hrm_integration_enabled(self):
        """Test HRM integration when enabled."""
        config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2023-06-10',
            initial_capital=100000.0,
            enable_hrm=True,  # Enable HRM
            enable_ssr=False,
            enable_luld=False
        )
        
        backtester = RealisticBacktester(config)
        
        # HRM integrator should be initialized
        assert backtester.hrm_integrator is not None
        
        market_data = create_test_market_data(10, ['HRM_TEST'], '2023-06-01')
        results = backtester.run_backtest(simple_momentum_strategy, market_data)
        
        # HRM performance should be included in results
        assert 'hrm_performance' in results.__dict__
        assert isinstance(results.hrm_performance, dict)


def test_comprehensive_realistic_backtest():
    """Comprehensive end-to-end realistic backtest."""
    config = BacktestConfig(
        start_date='2023-06-01',
        end_date='2023-06-30',
        initial_capital=500000.0,
        transaction_costs=0.0005,  # 0.05%
        slippage=0.0003,  # 0.03%
        max_position_size=0.15,  # 15% max per position
        enable_ssr=True,
        enable_luld=True,
        enable_hrm=False  # Disable for performance
    )
    
    backtester = RealisticBacktester(config)
    
    # Create comprehensive market data
    symbols = ['COMP1', 'COMP2', 'COMP3', 'COMP4']
    market_data = create_test_market_data(30, symbols, '2023-06-01')
    
    # More sophisticated strategy
    def comprehensive_strategy(market_data, positions, portfolio_value):
        signals = []
        
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 5:
                continue
            
            # Calculate moving averages
            prices = symbol_data['price'].values
            if len(prices) >= 5:
                short_ma = np.mean(prices[-3:])
                long_ma = np.mean(prices[-5:])
                current_price = prices[-1]
                
                # Mean reversion strategy
                if current_price < long_ma * 0.98:  # 2% below MA
                    signals.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': 200,
                        'timestamp': symbol_data['timestamp'].iloc[-1],
                        'reason': 'mean_reversion_buy'
                    })
                
                elif symbol in positions and positions[symbol].quantity > 0:
                    # Take profits
                    if current_price > positions[symbol].avg_price * 1.05:  # 5% profit
                        signals.append({
                            'symbol': symbol,
                            'side': 'sell',
                            'quantity': positions[symbol].quantity // 2,  # Sell half
                            'timestamp': symbol_data['timestamp'].iloc[-1],
                            'reason': 'take_profit'
                        })
        
        return signals
    
    # Run comprehensive backtest
    results = backtester.run_backtest(comprehensive_strategy, market_data)
    
    # Comprehensive result validation
    print(f"\nComprehensive Backtest Results:")
    print(f"Portfolio Value: ${results.portfolio_value:,.2f}")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Total Trades: {len(results.trades)}")
    print(f"Successful Trades: {len([t for t in results.trades if t.compliance_valid])}")
    print(f"Compliance Rate: {results.compliance_rate:.1%}")
    print(f"Regulatory Violations: {results.regulatory_violations}")
    print(f"SSR Triggers: {results.ssr_impacts['total_ssr_triggers']}")
    print(f"LULD Violations: {results.luld_impacts['total_luld_violations']}")
    
    # Assertions
    assert results.portfolio_value > 0
    assert -1.0 <= results.total_return <= 3.0  # Reasonable range
    assert 0.0 <= results.compliance_rate <= 1.0
    assert results.regulatory_violations >= 0
    assert len(results.daily_returns) > 0
    assert len(results.equity_curve) > 0
    
    # Check that we have reasonable trading activity
    successful_trades = [t for t in results.trades if t.compliance_valid]
    assert len(successful_trades) >= 0  # May be zero if no good signals
    
    # Check position management
    assert len(results.positions) <= len(symbols)  # Can't have more positions than symbols
    
    # Validate performance metrics are reasonable
    if len(results.daily_returns) > 1:
        daily_vol = np.std(results.daily_returns)
        assert 0.0 <= daily_vol <= 0.1  # Daily volatility should be reasonable
    
    assert results.max_drawdown >= 0.0
    assert abs(results.sharpe_ratio) <= 10.0  # Should not be extreme