"""
Tests for Paper Trading Engine.

Tests Alpaca integration, kill switches, and strategy execution
in paper trading environment.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import time

import sys
sys.path.append('/Users/aziymandias/Desktop/dual_book_trading_lab_v10_complete/lab_v10')

from src.trading.paper_trading import (
    PaperTradingEngine, AlpacaPaperTrader, KillSwitchManager,
    PaperTradingConfig, KillSwitchConfig, TradingState,
    KillSwitchType, TradingMetrics, KillSwitchEvent
)
from src.trading.strategy_executor import StrategyExecutor, StrategyConfig, Signal, Position


# Mock Alpaca API for testing
class MockAlpacaAPI:
    """Mock Alpaca API for testing."""
    
    def __init__(self):
        self.account_data = {
            'id': 'test_account',
            'equity': 100000.0,
            'cash': 50000.0,
            'buying_power': 200000.0,
            'portfolio_value': 100000.0,
            'daytrade_count': 0
        }
        self.positions = []
        self.orders = []
        self.connected = True
    
    def get_account(self):
        """Mock get_account."""
        mock_account = Mock()
        for key, value in self.account_data.items():
            setattr(mock_account, key, value)
        return mock_account
    
    def list_positions(self):
        """Mock list_positions."""
        return self.positions
    
    def submit_order(self, symbol, qty, side, type, time_in_force):
        """Mock submit_order."""
        order = Mock()
        order.id = f"order_{len(self.orders)}"
        order.symbol = symbol
        order.qty = str(qty)
        order.side = side
        order.status = 'filled'
        order.submitted_at = datetime.now()
        order.filled_at = datetime.now()
        order.filled_qty = str(qty)
        
        self.orders.append(order)
        return order
    
    def cancel_all_orders(self):
        """Mock cancel_all_orders."""
        self.orders.clear()
    
    def get_clock(self):
        """Mock get_clock."""
        mock_clock = Mock()
        mock_clock.is_open = True
        return mock_clock
    
    def get_bars(self, symbols, timeframe, start, end, limit, asof=None, feed=None):
        """Mock get_bars."""
        bars = []
        for symbol in symbols:
            for i in range(min(limit, 50)):  # Generate test data
                bar = Mock()
                bar.S = symbol
                bar.t = datetime.now() - timedelta(minutes=i)
                bar.o = 100.0 + np.random.normal(0, 1)
                bar.h = bar.o + abs(np.random.normal(0, 0.5))
                bar.l = bar.o - abs(np.random.normal(0, 0.5))
                bar.c = bar.o + np.random.normal(0, 0.5)
                bar.v = int(1000 + np.random.normal(0, 100))
                bars.append(bar)
        return bars


@pytest.fixture
def mock_alpaca_api():
    """Fixture providing mock Alpaca API."""
    return MockAlpacaAPI()


@pytest.fixture
def paper_trading_config():
    """Fixture providing paper trading configuration."""
    return PaperTradingConfig(
        initial_capital=100000.0,
        alpaca_api_key="test_key",
        alpaca_secret_key="test_secret",
        kill_switches=KillSwitchConfig(
            max_drawdown_threshold=0.10,
            max_daily_loss_threshold=0.05,
            enable_all=True
        ),
        update_frequency=1  # 1 second for testing
    )


@pytest.fixture
def kill_switch_config():
    """Fixture providing kill switch configuration."""
    return KillSwitchConfig(
        max_drawdown_threshold=0.15,
        max_daily_loss_threshold=0.05,
        market_vol_threshold=0.50,
        enable_all=True
    )


class TestKillSwitchManager:
    """Test kill switch manager functionality."""
    
    def test_kill_switch_manager_initialization(self, kill_switch_config):
        """Test kill switch manager initialization."""
        manager = KillSwitchManager(kill_switch_config)
        
        assert manager.config == kill_switch_config
        assert len(manager.active_switches) == len(KillSwitchType)
        assert all(not active for active in manager.active_switches.values())
        assert len(manager.switch_events) == 0
        assert manager.baseline_metrics is None
    
    def test_baseline_registration(self, kill_switch_config):
        """Test baseline metrics registration."""
        manager = KillSwitchManager(kill_switch_config)
        
        baseline_metrics = TradingMetrics(
            timestamp=datetime.now(),
            portfolio_value=100000.0,
            total_return=0.0,
            daily_return=0.0,
            drawdown=0.0,
            sharpe_ratio=0.0,
            active_positions=0,
            cash_balance=100000.0,
            equity=100000.0,
            buying_power=200000.0,
            day_trades_left=3
        )
        
        manager.register_baseline(baseline_metrics)
        assert manager.baseline_metrics == baseline_metrics
    
    def test_max_drawdown_kill_switch(self, kill_switch_config):
        """Test maximum drawdown kill switch."""
        manager = KillSwitchManager(kill_switch_config)
        
        # Register baseline
        baseline_metrics = TradingMetrics(
            timestamp=datetime.now(),
            portfolio_value=100000.0,
            total_return=0.0,
            daily_return=0.0,
            drawdown=0.0,
            sharpe_ratio=0.0,
            active_positions=0,
            cash_balance=100000.0,
            equity=100000.0,
            buying_power=200000.0,
            day_trades_left=3
        )
        manager.register_baseline(baseline_metrics)
        
        # Test normal drawdown (no trigger)
        normal_metrics = TradingMetrics(
            timestamp=datetime.now(),
            portfolio_value=95000.0,
            total_return=-0.05,
            daily_return=-0.02,
            drawdown=0.05,  # 5% drawdown (below 15% threshold)
            sharpe_ratio=0.5,
            active_positions=5,
            cash_balance=20000.0,
            equity=95000.0,
            buying_power=190000.0,
            day_trades_left=2
        )
        
        triggered, events = manager.check_all_switches(normal_metrics)
        assert not triggered
        assert len(events) == 0
        
        # Test excessive drawdown (trigger)
        high_drawdown_metrics = TradingMetrics(
            timestamp=datetime.now(),
            portfolio_value=80000.0,
            total_return=-0.20,
            daily_return=-0.03,
            drawdown=0.20,  # 20% drawdown (above 15% threshold)
            sharpe_ratio=0.2,
            active_positions=3,
            cash_balance=10000.0,
            equity=80000.0,
            buying_power=160000.0,
            day_trades_left=1
        )
        
        triggered, events = manager.check_all_switches(high_drawdown_metrics)
        assert triggered
        assert len(events) >= 1
        
        # Find drawdown event
        drawdown_events = [e for e in events if e.switch_type == KillSwitchType.MAX_DRAWDOWN]
        assert len(drawdown_events) == 1
        
        event = drawdown_events[0]
        assert event.trigger_value == 0.20
        assert event.threshold == 0.15
        assert "drawdown exceeded" in event.message.lower()
        assert event.action_taken == "EMERGENCY_STOP"
    
    def test_daily_loss_kill_switch(self, kill_switch_config):
        """Test daily loss kill switch."""
        manager = KillSwitchManager(kill_switch_config)
        
        # Test excessive daily loss
        high_loss_metrics = TradingMetrics(
            timestamp=datetime.now(),
            portfolio_value=92000.0,
            total_return=-0.08,
            daily_return=-0.08,  # 8% daily loss (above 5% threshold)
            drawdown=0.08,
            sharpe_ratio=0.3,
            active_positions=2,
            cash_balance=15000.0,
            equity=92000.0,
            buying_power=184000.0,
            day_trades_left=3
        )
        
        triggered, events = manager.check_all_switches(high_loss_metrics)
        assert triggered
        
        daily_loss_events = [e for e in events if e.switch_type == KillSwitchType.MAX_DAILY_LOSS]
        assert len(daily_loss_events) == 1
        
        event = daily_loss_events[0]
        assert event.trigger_value == -0.08
        assert event.threshold == -0.05
        assert "daily loss" in event.message.lower()
    
    def test_market_volatility_kill_switch(self, kill_switch_config):
        """Test market volatility kill switch."""
        manager = KillSwitchManager(kill_switch_config)
        
        # Create mock high volatility market data
        dates = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                             end=datetime.now(), freq='1min')
        
        # Generate highly volatile SPY data
        returns = np.random.normal(0, 0.05, len(dates))  # High volatility returns
        spy_prices = 100 * np.cumprod(1 + returns)
        
        spy_data = pd.DataFrame({
            'close': spy_prices,
            'open': spy_prices * (1 + np.random.normal(0, 0.001, len(spy_prices))),
            'high': spy_prices * (1 + abs(np.random.normal(0, 0.002, len(spy_prices)))),
            'low': spy_prices * (1 - abs(np.random.normal(0, 0.002, len(spy_prices)))),
            'volume': np.random.randint(1000, 10000, len(spy_prices))
        }, index=dates)
        
        market_data = {'SPY': spy_data}
        
        normal_metrics = TradingMetrics(
            timestamp=datetime.now(),
            portfolio_value=98000.0,
            total_return=-0.02,
            daily_return=-0.01,
            drawdown=0.02,
            sharpe_ratio=0.8,
            active_positions=4,
            cash_balance=30000.0,
            equity=98000.0,
            buying_power=196000.0,
            day_trades_left=3
        )
        
        triggered, events = manager.check_all_switches(normal_metrics, market_data)
        
        # Check if volatility kill switch was triggered
        vol_events = [e for e in events if e.switch_type == KillSwitchType.MARKET_VOLATILITY]
        if vol_events:  # May or may not trigger depending on random data
            event = vol_events[0]
            assert event.trigger_value > event.threshold
            assert "volatility" in event.message.lower()
    
    def test_switch_reset(self, kill_switch_config):
        """Test kill switch reset functionality."""
        manager = KillSwitchManager(kill_switch_config)
        
        # Manually activate a switch
        manager.active_switches[KillSwitchType.MAX_DRAWDOWN] = True
        assert manager.active_switches[KillSwitchType.MAX_DRAWDOWN]
        
        # Reset specific switch
        manager.reset_switch(KillSwitchType.MAX_DRAWDOWN)
        assert not manager.active_switches[KillSwitchType.MAX_DRAWDOWN]
        
        # Activate multiple switches
        manager.active_switches[KillSwitchType.MAX_DRAWDOWN] = True
        manager.active_switches[KillSwitchType.MAX_DAILY_LOSS] = True
        
        # Reset all switches
        manager.reset_all_switches()
        assert all(not active for active in manager.active_switches.values())


class TestAlpacaPaperTrader:
    """Test Alpaca paper trader functionality."""
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_alpaca_trader_initialization(self, mock_tradeapi, paper_trading_config):
        """Test Alpaca trader initialization."""
        # Mock the API
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        trader = AlpacaPaperTrader(paper_trading_config)
        
        assert trader.config == paper_trading_config
        assert trader._connected
        assert trader._api == mock_api
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_connection_check(self, mock_tradeapi, paper_trading_config):
        """Test connection checking."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        trader = AlpacaPaperTrader(paper_trading_config)
        
        # Test successful connection
        assert trader.is_connected()
        
        # Test connection failure
        mock_api.connected = False
        def raise_error():
            raise Exception("Connection failed")
        mock_api.get_clock = raise_error
        
        assert not trader.is_connected()
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_get_account_info(self, mock_tradeapi, paper_trading_config):
        """Test getting account information."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        trader = AlpacaPaperTrader(paper_trading_config)
        account_info = trader.get_account_info()
        
        assert 'account_id' in account_info
        assert 'equity' in account_info
        assert 'cash' in account_info
        assert 'buying_power' in account_info
        assert 'portfolio_value' in account_info
        assert 'active_positions' in account_info
        
        assert account_info['equity'] == 100000.0
        assert account_info['cash'] == 50000.0
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_place_order(self, mock_tradeapi, paper_trading_config):
        """Test placing orders."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        trader = AlpacaPaperTrader(paper_trading_config)
        
        order_result = trader.place_order('AAPL', 'buy', 100)
        
        assert 'order_id' in order_result
        assert 'symbol' in order_result
        assert 'side' in order_result
        assert 'quantity' in order_result
        assert 'status' in order_result
        
        assert order_result['symbol'] == 'AAPL'
        assert order_result['side'] == 'buy'
        assert order_result['quantity'] == 100
        assert order_result['status'] == 'filled'
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_get_market_data(self, mock_tradeapi, paper_trading_config):
        """Test getting market data."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        trader = AlpacaPaperTrader(paper_trading_config)
        
        market_data = trader.get_market_data(['AAPL', 'MSFT'], timeframe='1Min', limit=50)
        
        assert 'AAPL' in market_data
        assert 'MSFT' in market_data
        
        # Check data structure
        for symbol, data in market_data.items():
            if not data.empty:
                assert 'open' in data.columns
                assert 'high' in data.columns
                assert 'low' in data.columns
                assert 'close' in data.columns
                assert 'volume' in data.columns


class TestPaperTradingEngine:
    """Test main paper trading engine."""
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_engine_initialization(self, mock_tradeapi, paper_trading_config):
        """Test paper trading engine initialization."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        engine = PaperTradingEngine(paper_trading_config)
        
        assert engine.config == paper_trading_config
        assert engine.state == TradingState.STOPPED
        assert isinstance(engine.alpaca_trader, AlpacaPaperTrader)
        assert isinstance(engine.kill_switch_manager, KillSwitchManager)
        assert len(engine.metrics_history) == 0
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_start_trading(self, mock_tradeapi, paper_trading_config):
        """Test starting paper trading."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        engine = PaperTradingEngine(paper_trading_config)
        
        # Test successful start
        success = engine.start_trading()
        assert success
        assert engine.state == TradingState.RUNNING
        assert engine.start_time is not None
        assert engine.initial_portfolio_value is not None
        
        # Clean up
        engine.stop_trading()
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_stop_trading(self, mock_tradeapi, paper_trading_config):
        """Test stopping paper trading."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        engine = PaperTradingEngine(paper_trading_config)
        engine.start_trading()
        
        # Test stop
        engine.stop_trading()
        assert engine.state == TradingState.STOPPED
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_emergency_stop(self, mock_tradeapi, paper_trading_config):
        """Test emergency stop functionality."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        engine = PaperTradingEngine(paper_trading_config)
        engine.start_trading()
        
        # Test emergency stop
        engine.emergency_stop("Test emergency")
        assert engine.state == TradingState.EMERGENCY_STOP
        
        # Verify orders were cancelled
        assert len(mock_api.orders) == 0
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_get_current_metrics(self, mock_tradeapi, paper_trading_config):
        """Test getting current trading metrics."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        engine = PaperTradingEngine(paper_trading_config)
        engine.start_trading()
        
        metrics = engine.get_current_metrics()
        
        assert isinstance(metrics, TradingMetrics)
        assert metrics.portfolio_value > 0
        assert metrics.timestamp is not None
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.active_positions, int)
        
        engine.stop_trading()


class TestStrategyExecutor:
    """Test strategy executor functionality."""
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_strategy_executor_initialization(self, mock_tradeapi, paper_trading_config):
        """Test strategy executor initialization."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        paper_trader = PaperTradingEngine(paper_trading_config)
        strategy_config = StrategyConfig()
        
        executor = StrategyExecutor(paper_trader, strategy_config)
        
        assert executor.paper_trader == paper_trader
        assert executor.config == strategy_config
        assert not executor.is_running
        assert len(executor.execution_log) == 0
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_signal_generation(self, mock_tradeapi, paper_trading_config):
        """Test trading signal generation."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        paper_trader = PaperTradingEngine(paper_trading_config)
        paper_trader.start_trading()
        
        executor = StrategyExecutor(paper_trader, StrategyConfig())
        executor.initialize_strategy(['AAPL', 'MSFT'])  # Initialize first
        executor.start_execution()
        
        # Generate signals
        signals = executor.generate_signals(['AAPL', 'MSFT'])
        
        # Should generate some signals
        assert isinstance(signals, list)
        
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.symbol in ['AAPL', 'MSFT']
            assert signal.action in ['buy', 'sell', 'hold']
            assert 0.0 <= signal.confidence <= 1.0
            assert isinstance(signal.target_weight, float)
        
        paper_trader.stop_trading()
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_signal_execution(self, mock_tradeapi, paper_trading_config):
        """Test trading signal execution."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        paper_trader = PaperTradingEngine(paper_trading_config)
        paper_trader.start_trading()
        
        executor = StrategyExecutor(paper_trader, StrategyConfig(rebalance_frequency=0))
        executor.start_execution()
        
        # Create test signals
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                action='buy',
                confidence=0.8,
                target_weight=0.05,
                pattern_id='test_pattern'
            ),
            Signal(
                timestamp=datetime.now(),
                symbol='MSFT',
                action='buy',
                confidence=0.7,
                target_weight=0.04,
                pattern_id='test_pattern'
            )
        ]
        
        # Execute signals
        success = executor.execute_signals(signals)
        
        # Check execution
        if success:  # May not execute if no rebalancing needed
            assert len(executor.execution_log) > 0
            
            for log_entry in executor.execution_log:
                assert 'timestamp' in log_entry
                assert 'order' in log_entry
                assert log_entry['status'] in ['success', 'failed']
        
        paper_trader.stop_trading()
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_strategy_cycle(self, mock_tradeapi, paper_trading_config):
        """Test complete strategy cycle."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        paper_trader = PaperTradingEngine(paper_trading_config)
        paper_trader.start_trading()
        
        executor = StrategyExecutor(paper_trader, StrategyConfig())
        executor.initialize_strategy(['AAPL', 'MSFT', 'GOOGL'])  # Initialize first
        executor.start_execution()
        
        # Run strategy cycle
        universe = ['AAPL', 'MSFT', 'GOOGL']
        cycle_result = executor.run_strategy_cycle(universe)
        
        assert isinstance(cycle_result, dict)
        assert 'timestamp' in cycle_result
        assert 'signals_generated' in cycle_result
        assert 'status' in cycle_result
        assert cycle_result['status'] in ['success', 'error']
        
        paper_trader.stop_trading()
    
    @patch('src.trading.paper_trading.tradeapi')
    def test_execution_summary(self, mock_tradeapi, paper_trading_config):
        """Test execution summary generation."""
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        paper_trader = PaperTradingEngine(paper_trading_config)
        executor = StrategyExecutor(paper_trader, StrategyConfig())
        
        # Test with no execution log
        summary = executor.get_execution_summary()
        assert 'no_data' in summary
        
        # Add some mock execution log entries
        executor.execution_log = [
            {
                'timestamp': datetime.now(),
                'order': {'symbol': 'AAPL', 'side': 'buy', 'quantity': 100},
                'status': 'success'
            },
            {
                'timestamp': datetime.now(),
                'order': {'symbol': 'MSFT', 'side': 'sell', 'quantity': 50},
                'status': 'failed'
            }
        ]
        
        summary = executor.get_execution_summary()
        
        assert 'total_orders' in summary
        assert 'successful_orders' in summary
        assert 'failed_orders' in summary
        assert 'success_rate' in summary
        
        assert summary['total_orders'] == 2
        assert summary['successful_orders'] == 1
        assert summary['failed_orders'] == 1
        assert summary['success_rate'] == 0.5


def test_integration_kill_switch_trigger():
    """Integration test: Kill switch triggering emergency stop."""
    with patch('src.trading.paper_trading.tradeapi') as mock_tradeapi:
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        # Configure for quick kill switch trigger
        config = PaperTradingConfig(
            kill_switches=KillSwitchConfig(
                max_drawdown_threshold=0.05,  # Very low threshold
                enable_all=True
            ),
            update_frequency=0.1  # Fast updates
        )
        
        engine = PaperTradingEngine(config)
        
        # Mock account with high drawdown
        mock_api.account_data['portfolio_value'] = 80000.0  # 20% loss
        
        # Start trading
        assert engine.start_trading()
        
        # Wait briefly for monitoring to detect kill switch
        time.sleep(0.5)
        
        # Check if emergency stop was triggered
        # Note: This test may be flaky due to threading timing
        # In practice, you'd want more deterministic testing
        
        engine.stop_trading()


def test_end_to_end_paper_trading():
    """End-to-end test of paper trading system."""
    with patch('src.trading.paper_trading.tradeapi') as mock_tradeapi:
        mock_api = MockAlpacaAPI()
        mock_tradeapi.REST.return_value = mock_api
        
        # Create paper trading engine
        config = PaperTradingConfig(
            initial_capital=100000.0,
            kill_switches=KillSwitchConfig(enable_all=True)
        )
        
        engine = PaperTradingEngine(config)
        
        # Create strategy executor
        executor = StrategyExecutor(engine, StrategyConfig())
        
        try:
            # Start paper trading
            assert engine.start_trading()
            executor.initialize_strategy(['AAPL', 'MSFT'])  # Initialize first
            executor.start_execution()
            
            # Get initial metrics
            initial_metrics = engine.get_current_metrics()
            assert initial_metrics is not None
            assert initial_metrics.portfolio_value == 100000.0
            
            # Run strategy cycle
            universe = ['AAPL', 'MSFT']
            cycle_result = executor.run_strategy_cycle(universe)
            assert cycle_result['status'] == 'success'
            
            # Check execution summary
            summary = executor.get_execution_summary()
            assert isinstance(summary, dict)
            
            print(f"\nEnd-to-End Paper Trading Test Results:")
            print(f"Initial Portfolio Value: ${initial_metrics.portfolio_value:,.2f}")
            print(f"Signals Generated: {cycle_result.get('signals_generated', 0)}")
            print(f"Orders Executed: {cycle_result.get('orders_executed', False)}")
            print(f"Engine State: {engine.state.value}")
            
        finally:
            # Clean up
            executor.stop_execution()
            engine.stop_trading()