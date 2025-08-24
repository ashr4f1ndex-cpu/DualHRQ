"""
Realistic Backtesting Engine with Regulatory Compliance.

Integrates regulatory compliance (SSR/LULD) with walk-forward testing
to provide realistic backtesting that matches live trading constraints.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from .regulatory_compliance import RegulatoryComplianceEngine, BacktestComplianceValidator, RegulatoryRule
# from ..models.hrm_integration import HRMWalkForwardIntegrator
# from ..testing.walk_forward_testing import HistoricalWalkForwardTester, WalkForwardConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for realistic backtesting."""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    transaction_costs: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 0.10  # 10% of portfolio per position
    enable_ssr: bool = True
    enable_luld: bool = True
    enable_hrm: bool = True
    risk_free_rate: float = 0.02  # 2% annual risk-free rate


@dataclass
class Trade:
    """Individual trade record."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy', 'sell', 'short', 'cover'
    quantity: int
    price: float
    value: float
    commission: float
    compliance_valid: bool
    compliance_reason: str
    pattern_id: Optional[str] = None


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    position_type: str  # 'long', 'short'


@dataclass
class BacktestResults:
    """Backtesting results with regulatory compliance analysis."""
    trades: List[Trade]
    positions: List[Position]
    portfolio_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    compliance_rate: float
    regulatory_violations: int
    ssr_impacts: Dict[str, Any]
    luld_impacts: Dict[str, Any]
    hrm_performance: Dict[str, Any]
    daily_returns: pd.Series
    equity_curve: pd.Series


class RealisticBacktester:
    """
    Realistic backtesting engine with full regulatory compliance.
    
    Combines walk-forward testing, HRM-inspired learning, and regulatory
    constraints (SSR/LULD) to provide accurate backtesting results.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize realistic backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        
        # Initialize regulatory compliance engine
        self.compliance_engine = RegulatoryComplianceEngine(
            enable_ssr=config.enable_ssr,
            enable_luld=config.enable_luld
        )
        
        # Initialize HRM integration if enabled (simplified for Phase F)
        self.hrm_integrator = None
        # if config.enable_hrm:
        #     self.hrm_integrator = HRMWalkForwardIntegrator()
        
        # Walk-forward config (simplified)
        self.walk_forward_months = 6
        
        # Portfolio tracking
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.peak_value = config.initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"Realistic backtester initialized with ${config.initial_capital:,.0f} capital")
    
    def run_backtest(self, strategy_func: Callable, market_data: pd.DataFrame,
                    model: Optional[nn.Module] = None) -> BacktestResults:
        """
        Run complete realistic backtest with regulatory compliance.
        
        Args:
            strategy_func: Trading strategy function
            market_data: Historical market data
            model: Optional ML model for predictions
            
        Returns:
            Complete backtesting results with compliance analysis
        """
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Filter data to backtest period
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        backtest_data = market_data[
            (market_data['timestamp'] >= start_date) & 
            (market_data['timestamp'] <= end_date)
        ].copy()
        
        if len(backtest_data) == 0:
            raise ValueError(f"No data found for backtest period {start_date} to {end_date}")
        
        # Group by date for daily processing
        daily_data = backtest_data.groupby(backtest_data['timestamp'].dt.date)
        
        for date, day_data in daily_data:
            self._process_trading_day(date, day_data, strategy_func, model)
        
        # Generate final results
        return self._generate_results()
    
    def _process_trading_day(self, date, day_data: pd.DataFrame, 
                           strategy_func: Callable, model: Optional[nn.Module]):
        """Process a single trading day."""
        # Update market data for regulatory compliance
        for _, row in day_data.iterrows():
            compliance_status = self.compliance_engine.update_market_data(
                symbol=row['symbol'],
                timestamp=row['timestamp'], 
                price=row['price'],
                volume=row.get('volume', 1000),
                high=row.get('high', row['price']),
                low=row.get('low', row['price'])
            )
            
            # Update position values
            self._update_position_values(row['symbol'], row['price'])
        
        # Generate trading signals
        signals = strategy_func(day_data, self.positions, self.portfolio_value)
        
        # Execute trades with compliance checking
        for signal in signals:
            self._execute_trade_with_compliance(signal, day_data)
        
        # Update daily portfolio value and returns
        daily_value = self._calculate_portfolio_value(day_data)
        self.equity_curve.append((pd.to_datetime(date), daily_value))
        
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2][1]
            daily_return = (daily_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
            
            # Update max drawdown
            if daily_value > self.peak_value:
                self.peak_value = daily_value
            else:
                drawdown = (self.peak_value - daily_value) / self.peak_value
                self.max_drawdown = max(self.max_drawdown, drawdown)
        
        self.portfolio_value = daily_value
    
    def _execute_trade_with_compliance(self, signal: Dict[str, Any], market_data: pd.DataFrame):
        """Execute trade with full regulatory compliance checking."""
        symbol = signal['symbol']
        side = signal['side']
        target_quantity = signal['quantity']
        timestamp = signal.get('timestamp', market_data['timestamp'].iloc[-1])
        
        # Get current market price
        current_price = market_data[market_data['symbol'] == symbol]['price'].iloc[-1]
        
        # Apply slippage
        if side in ['buy', 'cover']:
            execution_price = current_price * (1 + self.config.slippage)
        else:  # sell, short
            execution_price = current_price * (1 - self.config.slippage)
        
        # Validate trade with compliance engine
        is_valid, rejection_reason, compliance_info = self.compliance_engine.validate_order(
            symbol=symbol,
            side=side,
            quantity=abs(target_quantity),
            price=execution_price,
            timestamp=timestamp
        )
        
        if not is_valid:
            # Record failed trade
            failed_trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                quantity=target_quantity,
                price=execution_price,
                value=0.0,
                commission=0.0,
                compliance_valid=False,
                compliance_reason=rejection_reason
            )
            self.trades.append(failed_trade)
            logger.warning(f"Trade rejected: {rejection_reason}")
            return
        
        # Check position size limits
        position_value = abs(target_quantity) * execution_price
        max_position_value = self.portfolio_value * self.config.max_position_size
        
        if position_value > max_position_value:
            # Scale down position to comply with size limits
            target_quantity = int(target_quantity * (max_position_value / position_value))
            if abs(target_quantity) == 0:
                logger.warning(f"Position too small after size limit adjustment: {symbol}")
                return
        
        # Calculate transaction costs
        commission = position_value * self.config.transaction_costs
        
        # Check if we have enough capital
        required_capital = position_value + commission
        if side in ['buy', 'cover'] and required_capital > self.cash:
            logger.warning(f"Insufficient capital for trade: need ${required_capital:,.2f}, have ${self.cash:,.2f}")
            return
        
        # Execute the trade
        executed_quantity = self._update_position(symbol, side, target_quantity, execution_price)
        
        if executed_quantity != 0:
            # Record successful trade
            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                quantity=executed_quantity,
                price=execution_price,
                value=abs(executed_quantity) * execution_price,
                commission=commission,
                compliance_valid=True,
                compliance_reason="Trade executed",
                pattern_id=signal.get('pattern_id')
            )
            self.trades.append(trade)
            
            # Update cash
            if side in ['buy', 'cover']:
                self.cash -= (trade.value + commission)
            else:  # sell, short
                self.cash += (trade.value - commission)
            
            logger.info(f"Executed: {side} {executed_quantity} {symbol} @ ${execution_price:.2f}")
    
    def _update_position(self, symbol: str, side: str, quantity: int, price: float) -> int:
        """Update position based on trade execution."""
        if symbol not in self.positions:
            # New position
            if side in ['buy', 'short']:
                position_type = 'long' if side == 'buy' else 'short'
                quantity = quantity if side == 'buy' else -quantity
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0.0,
                    position_type=position_type
                )
                return quantity
        
        else:
            # Existing position
            position = self.positions[symbol]
            
            if side == 'buy':
                # Adding to long or covering short
                if position.quantity >= 0:  # Long position
                    new_quantity = position.quantity + quantity
                    new_avg_price = ((position.quantity * position.avg_price) + (quantity * price)) / new_quantity
                    position.quantity = new_quantity
                    position.avg_price = new_avg_price
                else:  # Short position - covering
                    position.quantity += quantity  # Reduces short position
                    if position.quantity > 0:  # Flipped to long
                        position.position_type = 'long'
                        position.avg_price = price
                return quantity
            
            elif side == 'sell':
                # Reducing long position
                if position.quantity > 0:
                    executed_qty = min(quantity, position.quantity)
                    position.quantity -= executed_qty
                    if position.quantity == 0:
                        del self.positions[symbol]
                    return executed_qty
                    
            elif side == 'short':
                # Adding to short or shorting from flat
                if position.quantity <= 0:  # Short or flat
                    new_quantity = position.quantity - quantity
                    if position.quantity == 0:  # New short
                        position.avg_price = price
                    else:  # Adding to short
                        new_avg_price = ((abs(position.quantity) * position.avg_price) + (quantity * price)) / abs(new_quantity)
                        position.avg_price = new_avg_price
                    position.quantity = new_quantity
                    position.position_type = 'short'
                    return -quantity
                else:  # Long position - partial close
                    executed_qty = min(quantity, position.quantity)
                    position.quantity -= executed_qty
                    return executed_qty
        
        return 0
    
    def _update_position_values(self, symbol: str, current_price: float):
        """Update position market values and P&L."""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            position.market_value = position.quantity * current_price
            
            if position.position_type == 'long':
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
            else:  # short
                position.unrealized_pnl = (position.avg_price - current_price) * abs(position.quantity)
    
    def _calculate_portfolio_value(self, market_data: pd.DataFrame) -> float:
        """Calculate total portfolio value."""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            # Get latest price for symbol
            symbol_data = market_data[market_data['symbol'] == symbol]
            if not symbol_data.empty:
                current_price = symbol_data['price'].iloc[-1]
                self._update_position_values(symbol, current_price)
                total_value += position.market_value
        
        return total_value
    
    def _generate_results(self) -> BacktestResults:
        """Generate comprehensive backtest results."""
        # Calculate performance metrics
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 0:
            daily_returns = np.array(self.daily_returns)
            excess_returns = daily_returns - (self.config.risk_free_rate / 252)  # Daily risk-free rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Regulatory compliance analysis
        validator = BacktestComplianceValidator(self.compliance_engine)
        trades_df = pd.DataFrame([{
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'side': trade.side,
            'quantity': trade.quantity,
            'price': trade.price
        } for trade in self.trades if trade.compliance_valid])
        
        compliance_report = validator.validate_backtest_trades(trades_df) if len(trades_df) > 0 else {
            'compliance_rate': 1.0, 'total_violations': 0
        }
        
        # SSR/LULD impact analysis
        ssr_impacts = self._analyze_ssr_impacts()
        luld_impacts = self._analyze_luld_impacts()
        
        # HRM performance analysis (simplified for Phase F)
        hrm_performance = {'total_adaptations': 0, 'note': 'HRM integration simplified in Phase F'}
        
        # Create equity curve series
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'value'])
        equity_series = equity_df.set_index('date')['value']
        daily_returns_series = pd.Series(self.daily_returns)
        
        return BacktestResults(
            trades=self.trades,
            positions=list(self.positions.values()),
            portfolio_value=self.portfolio_value,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            compliance_rate=compliance_report.get('compliance_rate', 1.0),
            regulatory_violations=compliance_report.get('total_violations', 0),
            ssr_impacts=ssr_impacts,
            luld_impacts=luld_impacts,
            hrm_performance=hrm_performance,
            daily_returns=daily_returns_series,
            equity_curve=equity_series
        )
    
    def _analyze_ssr_impacts(self) -> Dict[str, Any]:
        """Analyze impact of SSR restrictions."""
        ssr_stats = {
            'total_ssr_triggers': len(self.compliance_engine.ssr_triggers),
            'rejected_short_trades': 0,
            'ssr_affected_symbols': set(),
            'estimated_impact_pnl': 0.0
        }
        
        # Count SSR-related trade rejections
        for trade in self.trades:
            if not trade.compliance_valid and 'SSR' in trade.compliance_reason:
                ssr_stats['rejected_short_trades'] += 1
                ssr_stats['ssr_affected_symbols'].add(trade.symbol)
        
        ssr_stats['ssr_affected_symbols'] = len(ssr_stats['ssr_affected_symbols'])
        return ssr_stats
    
    def _analyze_luld_impacts(self) -> Dict[str, Any]:
        """Analyze impact of LULD restrictions."""
        luld_stats = {
            'total_luld_violations': len(self.compliance_engine.luld_violations),
            'total_trading_halts': len(self.compliance_engine.halt_history),
            'halt_affected_symbols': set(),
            'avg_halt_duration': 0.0
        }
        
        # Analyze trading halts
        halt_durations = []
        for halt in self.compliance_engine.halt_history:
            if halt.halt_type == RegulatoryRule.LULD:
                luld_stats['halt_affected_symbols'].add(halt.symbol)
                if halt.halt_end:
                    duration = (halt.halt_end - halt.halt_start).total_seconds() / 60  # Minutes
                    halt_durations.append(duration)
        
        if halt_durations:
            luld_stats['avg_halt_duration'] = np.mean(halt_durations)
        
        luld_stats['halt_affected_symbols'] = len(luld_stats['halt_affected_symbols'])
        return luld_stats