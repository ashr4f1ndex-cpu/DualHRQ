"""
Paper Trading Engine for DualHRQ.

Implements realistic paper trading with Alpaca Markets integration,
comprehensive kill-switches, and risk management controls.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from alpaca_trade_api.rest import TimeFrame

from ..validation.backtest_validation import ValidationResults

logger = logging.getLogger(__name__)


class TradingState(Enum):
    """Paper trading system states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


class KillSwitchType(Enum):
    """Types of kill switches."""
    MAX_DRAWDOWN = "max_drawdown"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_POSITION_SIZE = "max_position_size"
    STRATEGY_DIVERGENCE = "strategy_divergence"
    MARKET_VOLATILITY = "market_volatility"
    CONNECTION_FAILURE = "connection_failure"
    VALIDATION_FAILURE = "validation_failure"


@dataclass
class KillSwitchConfig:
    """Configuration for kill switches."""
    max_drawdown_threshold: float = 0.15  # 15% max drawdown
    max_daily_loss_threshold: float = 0.05  # 5% max daily loss
    max_position_size: float = 0.10  # 10% of portfolio per position
    strategy_divergence_threshold: float = 0.20  # 20% performance divergence
    market_vol_threshold: float = 0.50  # 50% daily market volatility threshold
    connection_timeout: int = 30  # seconds
    validation_confidence_threshold: float = 0.60  # minimum validation confidence
    enable_all: bool = True


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading engine."""
    initial_capital: float = 100000.0
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    kill_switches: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    update_frequency: int = 60  # seconds between updates
    max_positions: int = 20
    enable_logging: bool = True
    enable_alerts: bool = True
    performance_tracking_window: int = 252  # days


@dataclass
class TradingMetrics:
    """Real-time trading metrics."""
    timestamp: datetime
    portfolio_value: float
    total_return: float
    daily_return: float
    drawdown: float
    sharpe_ratio: float
    active_positions: int
    cash_balance: float
    equity: float
    buying_power: float
    day_trades_left: int


@dataclass
class KillSwitchEvent:
    """Kill switch activation event."""
    timestamp: datetime
    switch_type: KillSwitchType
    trigger_value: float
    threshold: float
    message: str
    action_taken: str


class AlpacaPaperTrader:
    """
    Alpaca Markets paper trading integration.
    
    Provides realistic execution with market data feeds,
    order management, and portfolio tracking.
    """

    def __init__(self, config: PaperTradingConfig):
        self.config = config
        self._api = None
        self._connected = False
        self._last_heartbeat = None

        # Initialize Alpaca API
        self._initialize_api()

    def _initialize_api(self):
        """Initialize Alpaca API connection."""
        try:
            self._api = tradeapi.REST(
                key_id=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
                base_url=self.config.alpaca_base_url,
                api_version='v2'
            )

            # Test connection
            account = self._api.get_account()
            self._connected = True
            self._last_heartbeat = datetime.now()

            logger.info(f"Connected to Alpaca paper trading. Account: {account.id}")
            logger.info(f"Buying power: ${float(account.buying_power):,.2f}")

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API: {e}")
            self._connected = False
            raise

    def is_connected(self) -> bool:
        """Check if API connection is active."""
        if not self._connected or not self._api:
            return False

        try:
            # Heartbeat check
            if self._last_heartbeat and \
               datetime.now() - self._last_heartbeat > timedelta(seconds=self.config.kill_switches.connection_timeout):
                return False

            # Test connection with lightweight call
            self._api.get_clock()
            self._last_heartbeat = datetime.now()
            return True

        except Exception as e:
            logger.warning(f"Connection check failed: {e}")
            self._connected = False
            return False

    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca API")

        try:
            account = self._api.get_account()
            positions = self._api.list_positions()

            return {
                'account_id': account.id,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': int(account.daytrade_count),
                'day_trades_left': 3 - int(account.daytrade_count),  # PDT rule
                'active_positions': len(positions),
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'qty': float(pos.qty),
                        'market_value': float(pos.market_value),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_plpc': float(pos.unrealized_plpc)
                    } for pos in positions
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise

    def place_order(self, symbol: str, side: str, quantity: int,
                   order_type: str = 'market', time_in_force: str = 'day') -> Dict[str, Any]:
        """Place a paper trading order."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca API")

        try:
            order = self._api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )

            logger.info(f"Order placed: {side} {quantity} {symbol} ({order_type})")

            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': int(order.qty),
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_qty': int(order.filled_qty) if order.filled_qty else 0
            }

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def get_market_data(self, symbols: List[str], timeframe: str = '1Min',
                       limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Get market data for symbols."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca API")

        try:
            # Convert timeframe (simplified for compatibility)
            tf_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame.Minute,  # Will handle multiplier in API call
                '15Min': TimeFrame.Minute,
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day
            }

            timeframe_obj = tf_map.get(timeframe, TimeFrame.Minute)

            # Get bars for all symbols
            end_time = datetime.now()
            start_time = end_time - timedelta(days=5)  # 5 days of data

            bars = self._api.get_bars(
                symbols, timeframe_obj, start=start_time, end=end_time,
                limit=limit, asof=None, feed=None
            )

            # Convert to DataFrames
            data = {}
            for symbol in symbols:
                symbol_bars = [bar for bar in bars if bar.S == symbol]

                if symbol_bars:
                    df = pd.DataFrame([{
                        'timestamp': bar.t,
                        'open': bar.o,
                        'high': bar.h,
                        'low': bar.l,
                        'close': bar.c,
                        'volume': bar.v
                    } for bar in symbol_bars])

                    df.set_index('timestamp', inplace=True)
                    data[symbol] = df
                else:
                    data[symbol] = pd.DataFrame()

            return data

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            raise


class KillSwitchManager:
    """
    Comprehensive kill switch system for risk management.
    
    Monitors multiple risk factors and can immediately halt trading
    when thresholds are breached.
    """

    def __init__(self, config: KillSwitchConfig):
        self.config = config
        self.active_switches: Dict[KillSwitchType, bool] = {
            switch_type: False for switch_type in KillSwitchType
        }
        self.switch_events: List[KillSwitchEvent] = []
        self.baseline_metrics: Optional[TradingMetrics] = None

    def register_baseline(self, metrics: TradingMetrics):
        """Register baseline performance metrics."""
        self.baseline_metrics = metrics
        logger.info(f"Kill switch baseline registered: Portfolio ${metrics.portfolio_value:,.2f}")

    def check_all_switches(self, current_metrics: TradingMetrics,
                          market_data: Dict[str, pd.DataFrame] = None) -> Tuple[bool, List[KillSwitchEvent]]:
        """Check all kill switches and return activation status."""
        if not self.config.enable_all:
            return False, []

        triggered_events = []

        # Max drawdown check
        if self._check_max_drawdown(current_metrics):
            event = KillSwitchEvent(
                timestamp=current_metrics.timestamp,
                switch_type=KillSwitchType.MAX_DRAWDOWN,
                trigger_value=current_metrics.drawdown,
                threshold=self.config.max_drawdown_threshold,
                message=f"Maximum drawdown exceeded: {current_metrics.drawdown:.2%} > {self.config.max_drawdown_threshold:.2%}",
                action_taken="EMERGENCY_STOP"
            )
            triggered_events.append(event)

        # Daily loss check
        if self._check_daily_loss(current_metrics):
            event = KillSwitchEvent(
                timestamp=current_metrics.timestamp,
                switch_type=KillSwitchType.MAX_DAILY_LOSS,
                trigger_value=current_metrics.daily_return,
                threshold=-self.config.max_daily_loss_threshold,
                message=f"Daily loss limit exceeded: {current_metrics.daily_return:.2%} < {-self.config.max_daily_loss_threshold:.2%}",
                action_taken="EMERGENCY_STOP"
            )
            triggered_events.append(event)

        # Market volatility check
        if market_data and self._check_market_volatility(market_data):
            market_vol = self._calculate_market_volatility(market_data)
            event = KillSwitchEvent(
                timestamp=current_metrics.timestamp,
                switch_type=KillSwitchType.MARKET_VOLATILITY,
                trigger_value=market_vol,
                threshold=self.config.market_vol_threshold,
                message=f"Market volatility too high: {market_vol:.2%} > {self.config.market_vol_threshold:.2%}",
                action_taken="PAUSE_TRADING"
            )
            triggered_events.append(event)

        # Update switch states
        for event in triggered_events:
            self.active_switches[event.switch_type] = True
            self.switch_events.append(event)

        any_critical = any(event.action_taken == "EMERGENCY_STOP" for event in triggered_events)

        return any_critical or len(triggered_events) > 0, triggered_events

    def _check_max_drawdown(self, metrics: TradingMetrics) -> bool:
        """Check maximum drawdown kill switch."""
        if not self.baseline_metrics:
            return False

        return metrics.drawdown > self.config.max_drawdown_threshold

    def _check_daily_loss(self, metrics: TradingMetrics) -> bool:
        """Check daily loss kill switch."""
        return metrics.daily_return < -self.config.max_daily_loss_threshold

    def _check_market_volatility(self, market_data: Dict[str, pd.DataFrame]) -> bool:
        """Check market volatility kill switch."""
        market_vol = self._calculate_market_volatility(market_data)
        return market_vol > self.config.market_vol_threshold

    def _calculate_market_volatility(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current market volatility (SPY proxy)."""
        if 'SPY' in market_data and not market_data['SPY'].empty:
            spy_data = market_data['SPY']
            if len(spy_data) > 1:
                returns = spy_data['close'].pct_change().dropna()
                if len(returns) > 0:
                    # Daily volatility (annualized)
                    return returns.std() * np.sqrt(252)

        # Default if no SPY data
        return 0.20  # 20% baseline volatility

    def reset_switch(self, switch_type: KillSwitchType):
        """Reset a specific kill switch."""
        self.active_switches[switch_type] = False
        logger.info(f"Kill switch reset: {switch_type.value}")

    def reset_all_switches(self):
        """Reset all kill switches."""
        for switch_type in self.active_switches:
            self.active_switches[switch_type] = False
        logger.info("All kill switches reset")


class PaperTradingEngine:
    """
    Main paper trading engine coordinating all components.
    
    Integrates Alpaca paper trading, kill switches, performance monitoring,
    and strategy execution in a production-ready framework.
    """

    def __init__(self, config: PaperTradingConfig,
                 validation_results: Optional[ValidationResults] = None):
        self.config = config
        self.validation_results = validation_results
        self.state = TradingState.STOPPED

        # Initialize components
        self.alpaca_trader = AlpacaPaperTrader(config)
        self.kill_switch_manager = KillSwitchManager(config.kill_switches)

        # Performance tracking
        self.metrics_history: List[TradingMetrics] = []
        self.start_time: Optional[datetime] = None
        self.initial_portfolio_value: Optional[float] = None

        # Threading for async operations
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        # Callbacks
        self.on_kill_switch_triggered: Optional[Callable] = None
        self.on_performance_update: Optional[Callable] = None

    def start_trading(self) -> bool:
        """Start the paper trading engine."""
        try:
            self.state = TradingState.STARTING

            # Validate connection
            if not self.alpaca_trader.is_connected():
                raise ConnectionError("Alpaca API not connected")

            # Validate strategy if results provided
            if self.validation_results and \
               self.validation_results.confidence_score < self.config.kill_switches.validation_confidence_threshold:
                raise ValueError(f"Strategy validation confidence too low: "
                               f"{self.validation_results.confidence_score:.2%} < "
                               f"{self.config.kill_switches.validation_confidence_threshold:.2%}")

            # Initialize baseline metrics
            account_info = self.alpaca_trader.get_account_info()
            initial_metrics = self._create_metrics_from_account(account_info)
            self.kill_switch_manager.register_baseline(initial_metrics)
            self.initial_portfolio_value = initial_metrics.portfolio_value
            self.start_time = datetime.now()

            # Start monitoring thread
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()

            self.state = TradingState.RUNNING
            logger.info(f"Paper trading started. Initial portfolio: ${initial_metrics.portfolio_value:,.2f}")

            return True

        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            self.state = TradingState.ERROR
            return False

    def stop_trading(self):
        """Stop the paper trading engine."""
        self.state = TradingState.PAUSING

        # Stop monitoring
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10)

        self.state = TradingState.STOPPED
        logger.info("Paper trading stopped")

    def emergency_stop(self, reason: str):
        """Emergency stop with immediate halt."""
        logger.critical(f"EMERGENCY STOP: {reason}")

        # Stop monitoring first
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)

        # Cancel all open orders
        try:
            self.alpaca_trader._api.cancel_all_orders()
            logger.info("All open orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

        # Close all positions if configured
        # (Optional - usually keep positions in paper trading)

        # Set emergency stop state AFTER cleanup
        self.state = TradingState.EMERGENCY_STOP

    def get_current_metrics(self) -> Optional[TradingMetrics]:
        """Get current trading metrics."""
        if self.state not in [TradingState.RUNNING, TradingState.PAUSED]:
            return None

        try:
            account_info = self.alpaca_trader.get_account_info()
            return self._create_metrics_from_account(account_info)
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return None

    def _create_metrics_from_account(self, account_info: Dict[str, Any]) -> TradingMetrics:
        """Create TradingMetrics from account information."""
        portfolio_value = account_info['portfolio_value']

        # Calculate returns
        if self.initial_portfolio_value and self.metrics_history:
            total_return = (portfolio_value / self.initial_portfolio_value) - 1.0

            # Daily return from last metrics
            last_value = self.metrics_history[-1].portfolio_value
            daily_return = (portfolio_value / last_value) - 1.0 if last_value > 0 else 0.0
        else:
            total_return = 0.0
            daily_return = 0.0

        # Calculate drawdown
        if self.metrics_history:
            peak_value = max(m.portfolio_value for m in self.metrics_history[-252:])  # 1 year window
            drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0.0
        else:
            drawdown = 0.0

        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = 0.0
        if len(self.metrics_history) > 30:  # Need at least 30 data points
            returns = np.array([m.daily_return for m in self.metrics_history[-252:]])  # 1 year
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

        return TradingMetrics(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            total_return=total_return,
            daily_return=daily_return,
            drawdown=drawdown,
            sharpe_ratio=sharpe_ratio,
            active_positions=account_info['active_positions'],
            cash_balance=account_info['cash'],
            equity=account_info['equity'],
            buying_power=account_info['buying_power'],
            day_trades_left=account_info['day_trades_left']
        )

    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        logger.info("Monitoring loop started")

        while not self._stop_monitoring.is_set():
            try:
                # Get current metrics
                current_metrics = self.get_current_metrics()
                if not current_metrics:
                    continue

                # Store metrics
                self.metrics_history.append(current_metrics)

                # Keep only recent metrics (memory management)
                if len(self.metrics_history) > self.config.performance_tracking_window * 2:
                    self.metrics_history = self.metrics_history[-self.config.performance_tracking_window:]

                # Get market data for volatility check
                market_data = None
                try:
                    market_data = self.alpaca_trader.get_market_data(['SPY'], timeframe='1Min', limit=50)
                except Exception as e:
                    logger.warning(f"Failed to get market data: {e}")

                # Check kill switches
                kill_switch_triggered, events = self.kill_switch_manager.check_all_switches(
                    current_metrics, market_data
                )

                if kill_switch_triggered:
                    # Handle kill switch activation
                    critical_events = [e for e in events if e.action_taken == "EMERGENCY_STOP"]
                    if critical_events:
                        self.emergency_stop(f"Kill switches triggered: {[e.switch_type.value for e in critical_events]}")
                    else:
                        self.state = TradingState.PAUSED
                        logger.warning(f"Trading paused by kill switches: {[e.switch_type.value for e in events]}")

                    # Trigger callback
                    if self.on_kill_switch_triggered:
                        self.on_kill_switch_triggered(events)

                # Performance update callback
                if self.on_performance_update:
                    self.on_performance_update(current_metrics)

                # Log periodic status
                if len(self.metrics_history) % 60 == 0:  # Every hour with 1-min updates
                    logger.info(f"Portfolio: ${current_metrics.portfolio_value:,.2f}, "
                              f"Return: {current_metrics.total_return:.2%}, "
                              f"Drawdown: {current_metrics.drawdown:.2%}, "
                              f"Positions: {current_metrics.active_positions}")

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                if self.state == TradingState.RUNNING:
                    self.state = TradingState.ERROR

            # Wait for next update
            self._stop_monitoring.wait(self.config.update_frequency)

        logger.info("Monitoring loop stopped")
