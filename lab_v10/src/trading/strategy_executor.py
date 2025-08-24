"""
Strategy Executor for Paper Trading.

Executes DualHRQ strategies in paper trading environment with
real-time signal generation, position management, and risk controls.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import time

import numpy as np
import pandas as pd

from .paper_trading import PaperTradingEngine, TradingState
# Optional imports - may not be available in all environments
try:
    from ..learning.hrm_loop import HRMLoop, HRMConfig
except ImportError:
    HRMLoop = None
    HRMConfig = None

try:
    from ..data.market_data import MarketDataManager
except ImportError:
    MarketDataManager = None

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal from strategy."""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    target_weight: float  # Target portfolio weight
    pattern_id: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class Position:
    """Current position information."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float  # Portfolio weight
    entry_time: datetime


@dataclass
class StrategyConfig:
    """Configuration for strategy execution."""
    max_positions: int = 10
    max_position_weight: float = 0.10  # 10% max per position
    min_signal_confidence: float = 0.60  # Minimum confidence to act
    rebalance_frequency: int = 300  # seconds between rebalancing
    enable_adaptive_sizing: bool = True
    enable_risk_scaling: bool = True
    target_volatility: float = 0.15  # 15% annual target volatility


class PositionManager:
    """
    Manages portfolio positions and sizing decisions.
    
    Implements risk-based position sizing, rebalancing,
    and portfolio optimization.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.target_weights: Dict[str, float] = {}
        self.last_rebalance: Optional[datetime] = None
        
    def update_positions(self, account_info: Dict[str, Any]):
        """Update position information from account data."""
        self.positions.clear()
        
        total_portfolio_value = account_info['portfolio_value']
        
        for pos_data in account_info['positions']:
            if pos_data['qty'] != 0:  # Only active positions
                position = Position(
                    symbol=pos_data['symbol'],
                    quantity=int(pos_data['qty']),
                    entry_price=0.0,  # Not available from Alpaca API
                    current_price=pos_data['market_value'] / abs(pos_data['qty']),
                    market_value=pos_data['market_value'],
                    unrealized_pnl=pos_data['unrealized_pl'],
                    weight=pos_data['market_value'] / total_portfolio_value,
                    entry_time=datetime.now()  # Approximation
                )
                self.positions[position.symbol] = position
    
    def calculate_target_positions(self, signals: List[Signal], 
                                 current_portfolio_value: float) -> Dict[str, float]:
        """Calculate target position sizes based on signals."""
        target_weights = {}
        
        # Filter signals by confidence
        valid_signals = [s for s in signals if s.confidence >= self.config.min_signal_confidence]
        
        if not valid_signals:
            return target_weights
        
        # Calculate raw target weights
        total_weight = 0.0
        for signal in valid_signals:
            if signal.action in ['buy', 'hold']:
                # Base weight from signal
                weight = signal.target_weight * signal.confidence
                
                # Apply position size limits
                weight = min(weight, self.config.max_position_weight)
                
                target_weights[signal.symbol] = weight
                total_weight += weight
        
        # Normalize if total weight exceeds 1.0
        if total_weight > 1.0:
            for symbol in target_weights:
                target_weights[symbol] /= total_weight
        
        # Apply adaptive sizing based on portfolio volatility
        if self.config.enable_adaptive_sizing:
            target_weights = self._apply_adaptive_sizing(target_weights, current_portfolio_value)
        
        self.target_weights = target_weights
        return target_weights
    
    def _apply_adaptive_sizing(self, weights: Dict[str, float], 
                              portfolio_value: float) -> Dict[str, float]:
        """Apply adaptive position sizing based on recent portfolio performance."""
        # This is a simplified version - real implementation would use
        # historical volatility and correlation analysis
        
        if len(self.positions) == 0:
            return weights  # No adjustment for new portfolios
        
        # Calculate current portfolio volatility (simplified)
        current_vol = self._estimate_portfolio_volatility()
        target_vol = self.config.target_volatility
        
        # Scale positions based on volatility target
        if current_vol > 0:
            vol_adjustment = min(target_vol / current_vol, 1.5)  # Cap adjustment at 1.5x
            
            adjusted_weights = {}
            for symbol, weight in weights.items():
                adjusted_weights[symbol] = weight * vol_adjustment
            
            return adjusted_weights
        
        return weights
    
    def _estimate_portfolio_volatility(self) -> float:
        """Estimate current portfolio volatility (simplified)."""
        # In practice, this would use historical return data
        # For now, return a reasonable estimate
        return 0.18  # 18% annual volatility estimate
    
    def calculate_rebalancing_orders(self, target_weights: Dict[str, float],
                                   current_portfolio_value: float) -> List[Dict[str, Any]]:
        """Calculate orders needed for rebalancing."""
        orders = []
        
        # Current weights
        current_weights = {symbol: pos.weight for symbol, pos in self.positions.items()}
        
        # Calculate required trades
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            # Skip small adjustments
            if abs(weight_diff) < 0.01:  # 1% threshold
                continue
            
            # Calculate dollar amount and shares
            dollar_amount = weight_diff * current_portfolio_value
            
            # Estimate shares needed (would need current price in practice)
            if symbol in self.positions:
                current_price = self.positions[symbol].current_price
            else:
                current_price = 100.0  # Placeholder - need real price lookup
            
            shares = int(dollar_amount / current_price)
            
            if shares != 0:
                order = {
                    'symbol': symbol,
                    'side': 'buy' if shares > 0 else 'sell',
                    'quantity': abs(shares),
                    'order_type': 'market',
                    'reason': f'Rebalance: {current_weight:.2%} -> {target_weight:.2%}',
                    'target_weight': target_weight
                }
                orders.append(order)
        
        # Check for positions to close (not in target weights)
        for symbol, position in self.positions.items():
            if symbol not in target_weights and position.quantity != 0:
                order = {
                    'symbol': symbol,
                    'side': 'sell' if position.quantity > 0 else 'buy',
                    'quantity': abs(position.quantity),
                    'order_type': 'market',
                    'reason': 'Close position',
                    'target_weight': 0.0
                }
                orders.append(order)
        
        return orders
    
    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced."""
        if not self.last_rebalance:
            return True
        
        time_since_rebalance = (datetime.now() - self.last_rebalance).total_seconds()
        return time_since_rebalance >= self.config.rebalance_frequency


class StrategyExecutor:
    """
    Main strategy executor coordinating signal generation and execution.
    
    Integrates HRM loop, position management, and paper trading
    in a cohesive execution framework.
    """
    
    def __init__(self, paper_trader: PaperTradingEngine, config: StrategyConfig):
        self.paper_trader = paper_trader
        self.config = config
        self.position_manager = PositionManager(config)
        
        # Strategy components
        self.hrm_loop: Optional[HRMLoop] = None
        self.market_data_manager: Optional[MarketDataManager] = None
        
        # Execution state
        self.is_running = False
        self.last_signal_time: Optional[datetime] = None
        self.execution_log: List[Dict[str, Any]] = []
        
        # Callbacks
        self.on_signal_generated: Optional[Callable] = None
        self.on_order_placed: Optional[Callable] = None
    
    def initialize_strategy(self, universe: List[str], 
                          hrm_config: Optional[Dict] = None):
        """Initialize strategy components."""
        try:
            # Initialize HRM loop (if available)
            if HRMLoop is not None and HRMConfig is not None:
                if hrm_config is None:
                    hrm_config = HRMConfig()
                self.hrm_loop = HRMLoop(hrm_config)
            else:
                logger.warning("HRM loop not available - using simplified strategy")
                self.hrm_loop = None
            
            # Initialize market data manager (if available)
            if MarketDataManager is not None:
                # self.market_data_manager = MarketDataManager()
                pass
            else:
                logger.warning("Market data manager not available - using Alpaca data")
                self.market_data_manager = None
            
            logger.info(f"Strategy initialized for {len(universe)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {e}")
            raise
    
    def start_execution(self):
        """Start strategy execution."""
        if self.paper_trader.state != TradingState.RUNNING:
            raise RuntimeError("Paper trader must be running to start execution")
        
        self.is_running = True
        logger.info("Strategy execution started")
    
    def stop_execution(self):
        """Stop strategy execution."""
        self.is_running = False
        logger.info("Strategy execution stopped")
    
    def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Generate trading signals for given symbols."""
        if not self.is_running:
            return []
        
        signals = []
        
        try:
            # Get market data
            market_data = self.paper_trader.alpaca_trader.get_market_data(
                symbols, timeframe='1Min', limit=100
            )
            
            # Generate signals for each symbol
            for symbol in symbols:
                if symbol not in market_data or market_data[symbol].empty:
                    continue
                
                data = market_data[symbol]
                
                # Simple signal generation (placeholder for actual strategy)
                signal = self._generate_signal_for_symbol(symbol, data)
                if signal:
                    signals.append(signal)
            
            self.last_signal_time = datetime.now()
            
            # Trigger callback
            if self.on_signal_generated:
                self.on_signal_generated(signals)
            
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
        
        return signals
    
    def _generate_signal_for_symbol(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal for a single symbol (simplified implementation)."""
        if len(data) < 20:  # Need minimum data
            return None
        
        try:
            # Simple moving average crossover strategy
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean() if len(data) >= 50 else data['close'].rolling(20).mean()
            
            current_price = data['close'].iloc[-1]
            sma_20 = data['sma_20'].iloc[-1]
            sma_50 = data['sma_50'].iloc[-1]
            
            # Generate signal
            if sma_20 > sma_50 * 1.01:  # 1% buffer
                action = 'buy'
                confidence = min((sma_20 - sma_50) / sma_50 * 10, 1.0)  # Scale to 0-1
                target_weight = 0.05 + confidence * 0.05  # 5-10% weight
            elif sma_20 < sma_50 * 0.99:  # 1% buffer
                action = 'sell'
                confidence = min((sma_50 - sma_20) / sma_50 * 10, 1.0)
                target_weight = 0.0
            else:
                action = 'hold'
                confidence = 0.5
                target_weight = self.position_manager.target_weights.get(symbol, 0.0)
            
            # Add some noise and market regime considerations
            if self.hrm_loop:
                # Use HRM for pattern recognition (simplified)
                pattern_confidence = 0.8  # Placeholder
                confidence *= pattern_confidence
            
            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                target_weight=target_weight,
                pattern_id=f"{symbol}_ma_crossover",
                metadata={
                    'current_price': current_price,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'strategy': 'simple_ma_crossover'
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate signal for {symbol}: {e}")
            return None
    
    def execute_signals(self, signals: List[Signal]) -> bool:
        """Execute trading signals."""
        if not self.is_running or not signals:
            return False
        
        try:
            # Update current positions
            account_info = self.paper_trader.alpaca_trader.get_account_info()
            self.position_manager.update_positions(account_info)
            
            # Calculate target positions
            target_weights = self.position_manager.calculate_target_positions(
                signals, account_info['portfolio_value']
            )
            
            # Check if rebalancing is needed
            if not self.position_manager.should_rebalance():
                return False
            
            # Calculate required orders
            orders = self.position_manager.calculate_rebalancing_orders(
                target_weights, account_info['portfolio_value']
            )
            
            if not orders:
                return False
            
            # Execute orders
            successful_orders = 0
            for order in orders:
                try:
                    result = self.paper_trader.alpaca_trader.place_order(
                        symbol=order['symbol'],
                        side=order['side'],
                        quantity=order['quantity'],
                        order_type=order['order_type']
                    )
                    
                    # Log execution
                    log_entry = {
                        'timestamp': datetime.now(),
                        'order': order,
                        'result': result,
                        'status': 'success'
                    }
                    self.execution_log.append(log_entry)
                    successful_orders += 1
                    
                    logger.info(f"Order executed: {order['side']} {order['quantity']} {order['symbol']}")
                    
                    # Trigger callback
                    if self.on_order_placed:
                        self.on_order_placed(order, result)
                    
                except Exception as e:
                    logger.error(f"Failed to execute order {order}: {e}")
                    log_entry = {
                        'timestamp': datetime.now(),
                        'order': order,
                        'error': str(e),
                        'status': 'failed'
                    }
                    self.execution_log.append(log_entry)
            
            # Update rebalance timestamp
            self.position_manager.last_rebalance = datetime.now()
            
            logger.info(f"Executed {successful_orders}/{len(orders)} orders")
            return successful_orders > 0
            
        except Exception as e:
            logger.error(f"Failed to execute signals: {e}")
            return False
    
    def run_strategy_cycle(self, universe: List[str]) -> Dict[str, Any]:
        """Run one complete strategy cycle."""
        cycle_start = datetime.now()
        
        try:
            # Generate signals
            signals = self.generate_signals(universe)
            
            # Execute signals
            orders_executed = self.execute_signals(signals) if signals else False
            
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            
            return {
                'timestamp': cycle_start,
                'signals_generated': len(signals),
                'orders_executed': orders_executed,
                'cycle_time_seconds': cycle_time,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Strategy cycle failed: {e}")
            return {
                'timestamp': cycle_start,
                'error': str(e),
                'status': 'error'
            }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        if not self.execution_log:
            return {'no_data': True}
        
        successful_orders = [log for log in self.execution_log if log['status'] == 'success']
        failed_orders = [log for log in self.execution_log if log['status'] == 'failed']
        
        return {
            'total_orders': len(self.execution_log),
            'successful_orders': len(successful_orders),
            'failed_orders': len(failed_orders),
            'success_rate': len(successful_orders) / len(self.execution_log) if self.execution_log else 0,
            'last_execution': self.execution_log[-1]['timestamp'] if self.execution_log else None,
            'positions_count': len(self.position_manager.positions),
            'target_weights': self.position_manager.target_weights
        }