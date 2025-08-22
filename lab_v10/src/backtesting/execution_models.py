"""
Advanced Execution Models
Institutional-grade execution algorithms with market impact modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from scipy import optimize
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class ExecutionSlice:
    """Represents a single execution slice in an algorithmic strategy."""
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    expected_price: float
    market_impact: float
    timing_risk: float
    participation_rate: float
    venue: str = 'DEFAULT'

@dataclass
class MarketMicrostructure:
    """Market microstructure parameters for execution modeling."""
    symbol: str
    timestamp: pd.Timestamp
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    volatility: float
    momentum: float
    mean_reversion: float
    liquidity_score: float

class ExecutionModel(ABC):
    """Abstract base class for execution models."""
    
    @abstractmethod
    def calculate_execution_schedule(self, symbol: str, target_quantity: float, 
                                   time_horizon: int, market_data: pd.DataFrame) -> List[ExecutionSlice]:
        """Calculate optimal execution schedule."""
        pass
    
    @abstractmethod
    def calculate_market_impact(self, symbol: str, quantity: float, 
                              market_conditions: MarketMicrostructure) -> float:
        """Calculate expected market impact for a trade."""
        pass

class AlmgrenChrissModel(ExecutionModel):
    """
    Almgren-Chriss optimal execution model with extensions.
    
    Implements the seminal optimal execution framework with:
    - Linear market impact model
    - Quadratic risk penalty
    - Optimal trade scheduling
    - VWAP participation constraints
    """
    
    def __init__(self, risk_aversion: float = 1e-6, max_participation: float = 0.1):
        self.risk_aversion = risk_aversion
        self.max_participation = max_participation
        
        # Model parameters (will be calibrated from market data)
        self.permanent_impact_coeff = 0.314  # sqrt(tau) coefficient
        self.temporary_impact_coeff = 0.142  # participation rate coefficient
        self.volatility_adjustment = 1.0
        
    def calculate_execution_schedule(self, symbol: str, target_quantity: float, 
                                   time_horizon: int, market_data: pd.DataFrame) -> List[ExecutionSlice]:
        """
        Calculate Almgren-Chriss optimal execution schedule.
        
        Args:
            symbol: Asset symbol
            target_quantity: Total quantity to execute (positive for buy, negative for sell)
            time_horizon: Execution horizon in time periods
            market_data: Historical market data for calibration
            
        Returns:
            List of execution slices with optimal timing and sizes
        """
        
        if target_quantity == 0 or time_horizon <= 0:
            return []
        
        # Calibrate model parameters from market data
        self._calibrate_parameters(symbol, market_data)
        
        # Calculate optimal execution trajectory
        execution_times = np.arange(0, time_horizon + 1)
        optimal_trajectory = self._calculate_optimal_trajectory(
            target_quantity, time_horizon, execution_times
        )
        
        # Convert trajectory to execution slices
        execution_slices = []
        
        for i in range(len(optimal_trajectory) - 1):
            slice_quantity = optimal_trajectory[i] - optimal_trajectory[i + 1]
            
            if abs(slice_quantity) > 1e-8:  # Only create slice if meaningful quantity
                
                # Estimate market conditions at execution time
                timestamp = market_data.index[min(i, len(market_data) - 1)]
                market_conditions = self._estimate_market_conditions(symbol, timestamp, market_data)
                
                # Calculate expected impact
                market_impact = self.calculate_market_impact(symbol, slice_quantity, market_conditions)
                
                # Calculate participation rate
                daily_volume = market_data.loc[timestamp, 'Volume'] if 'Volume' in market_data.columns else 1000000
                participation_rate = min(abs(slice_quantity) / daily_volume, self.max_participation)
                
                # Calculate timing risk (variance of execution)
                timing_risk = self._calculate_timing_risk(slice_quantity, market_conditions)
                
                execution_slice = ExecutionSlice(
                    timestamp=timestamp,
                    symbol=symbol,
                    quantity=slice_quantity,
                    expected_price=market_conditions.last_price,
                    market_impact=market_impact,
                    timing_risk=timing_risk,
                    participation_rate=participation_rate
                )
                
                execution_slices.append(execution_slice)
        
        return execution_slices
    
    def _calibrate_parameters(self, symbol: str, market_data: pd.DataFrame) -> None:
        """Calibrate model parameters from historical data."""
        
        if len(market_data) < 20:
            return  # Not enough data for calibration
        
        # Calculate returns and volume metrics
        returns = market_data['Close'].pct_change().dropna()
        volumes = market_data['Volume'] if 'Volume' in market_data.columns else pd.Series([1000000] * len(market_data), index=market_data.index)
        
        # Estimate volatility
        volatility = returns.std() * np.sqrt(252)
        self.volatility_adjustment = max(0.5, min(2.0, volatility / 0.20))  # Scale relative to 20% base volatility
        
        # Estimate liquidity from volume patterns
        avg_volume = volumes.mean()
        volume_std = volumes.std()
        liquidity_score = avg_volume / (volume_std + 1e-8)
        
        # Adjust impact coefficients based on liquidity
        liquidity_adjustment = max(0.5, min(2.0, 1.0 / (liquidity_score / 1000 + 1)))
        
        self.permanent_impact_coeff *= liquidity_adjustment
        self.temporary_impact_coeff *= liquidity_adjustment
        
    def _calculate_optimal_trajectory(self, X: float, T: int, times: np.ndarray) -> np.ndarray:
        """
        Calculate optimal execution trajectory using Almgren-Chriss closed-form solution.
        
        Args:
            X: Total quantity to execute
            T: Time horizon
            times: Array of execution times
            
        Returns:
            Optimal holdings trajectory
        """
        
        # Model parameters
        tau = 1.0  # Time interval
        sigma = self.volatility_adjustment * 0.2  # Volatility (20% base)
        eta = self.temporary_impact_coeff  # Temporary impact coefficient
        gamma = self.permanent_impact_coeff  # Permanent impact coefficient
        lambda_risk = self.risk_aversion  # Risk aversion
        
        # Almgren-Chriss parameters
        kappa = np.sqrt(lambda_risk * sigma**2 / eta)
        
        # Optimal trajectory (exponential decay)
        if kappa * tau < 1e-6:  # Linear case (low risk aversion)
            trajectory = X * (1 - times / T)
        else:
            sinh_kappa_T = np.sinh(kappa * (T * tau))
            if abs(sinh_kappa_T) < 1e-10:
                trajectory = X * (1 - times / T)
            else:
                trajectory = X * np.sinh(kappa * (T - times) * tau) / sinh_kappa_T
        
        # Ensure trajectory starts at X and ends at 0
        trajectory[0] = X
        trajectory[-1] = 0
        
        return trajectory
    
    def calculate_market_impact(self, symbol: str, quantity: float, 
                              market_conditions: MarketMicrostructure) -> float:
        """
        Calculate market impact using extended Almgren-Chriss model.
        
        Includes:
        - Permanent impact (sqrt model)
        - Temporary impact (linear in participation rate)
        - Volatility adjustments
        - Liquidity adjustments
        """
        
        if abs(quantity) < 1e-8:
            return 0.0
        
        # Calculate participation rate
        participation_rate = abs(quantity) / max(market_conditions.volume, 1000)
        participation_rate = min(participation_rate, self.max_participation)
        
        # Permanent impact (square-root model)
        permanent_impact = self.permanent_impact_coeff * np.sqrt(abs(quantity)) * np.sqrt(participation_rate)
        
        # Temporary impact (linear model)
        temporary_impact = self.temporary_impact_coeff * participation_rate
        
        # Volatility adjustment
        vol_adjustment = market_conditions.volatility / 0.20  # Relative to 20% base volatility
        vol_adjustment = max(0.5, min(3.0, vol_adjustment))
        
        # Liquidity adjustment based on bid-ask spread
        spread = market_conditions.ask_price - market_conditions.bid_price
        spread_bps = spread / market_conditions.last_price * 10000
        liquidity_adjustment = max(1.0, spread_bps / 10.0)  # Scale based on 10bps normal spread
        
        # Momentum adjustment
        momentum_adjustment = 1.0 + 0.1 * abs(market_conditions.momentum)
        
        # Total impact as fraction of price
        total_impact_fraction = (permanent_impact + temporary_impact) * vol_adjustment * liquidity_adjustment * momentum_adjustment
        
        # Convert to absolute impact
        impact_price = total_impact_fraction * market_conditions.last_price
        
        # Apply direction
        if quantity > 0:  # Buy order pushes price up
            return impact_price
        else:  # Sell order pushes price down
            return -impact_price
    
    def _estimate_market_conditions(self, symbol: str, timestamp: pd.Timestamp, 
                                  market_data: pd.DataFrame) -> MarketMicrostructure:
        """Estimate market microstructure from available data."""
        
        # Find nearest data point
        available_times = market_data.index[market_data.index <= timestamp]
        if len(available_times) == 0:
            available_times = market_data.index[:1]
        
        latest_time = available_times[-1]
        current_data = market_data.loc[latest_time]
        
        # Extract basic price data
        last_price = current_data['Close']
        volume = current_data.get('Volume', 1000000)
        
        # Estimate bid-ask spread (typically 0.1% for liquid stocks)
        spread = max(0.01, last_price * 0.001)
        bid_price = last_price - spread / 2
        ask_price = last_price + spread / 2
        
        # Estimate volatility from recent data
        lookback_data = market_data.loc[:latest_time].tail(20)
        if len(lookback_data) > 1:
            returns = lookback_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.20
        else:
            volatility = 0.20
        
        # Estimate momentum (short-term trend)
        if len(lookback_data) >= 5:
            short_ma = lookback_data['Close'].tail(5).mean()
            momentum = (last_price - short_ma) / short_ma
        else:
            momentum = 0.0
        
        # Estimate mean reversion tendency
        if len(lookback_data) >= 10:
            long_ma = lookback_data['Close'].tail(10).mean()
            mean_reversion = (long_ma - last_price) / last_price
        else:
            mean_reversion = 0.0
        
        # Liquidity score based on volume consistency
        if len(lookback_data) >= 5:
            vol_std = lookback_data['Volume'].std() if 'Volume' in lookback_data.columns else 100000
            liquidity_score = volume / (vol_std + 1e-8)
        else:
            liquidity_score = 1.0
        
        return MarketMicrostructure(
            symbol=symbol,
            timestamp=timestamp,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=volume * 0.01,  # Estimate 1% of volume at bid
            ask_size=volume * 0.01,  # Estimate 1% of volume at ask
            last_price=last_price,
            volume=volume,
            volatility=volatility,
            momentum=momentum,
            mean_reversion=mean_reversion,
            liquidity_score=liquidity_score
        )
    
    def _calculate_timing_risk(self, quantity: float, market_conditions: MarketMicrostructure) -> float:
        """Calculate timing risk (variance) for execution slice."""
        
        # Base timing risk from volatility
        base_risk = market_conditions.volatility * abs(quantity) * market_conditions.last_price
        
        # Adjust for market conditions
        liquidity_adjustment = 1.0 / max(0.1, market_conditions.liquidity_score)
        momentum_adjustment = 1.0 + abs(market_conditions.momentum)
        
        return base_risk * liquidity_adjustment * momentum_adjustment

class VWAPExecutionModel(ExecutionModel):
    """
    Volume Weighted Average Price (VWAP) execution model.
    
    Follows historical volume patterns to minimize market impact
    while targeting VWAP execution.
    """
    
    def __init__(self, target_participation: float = 0.05, max_participation: float = 0.15):
        self.target_participation = target_participation
        self.max_participation = max_participation
    
    def calculate_execution_schedule(self, symbol: str, target_quantity: float, 
                                   time_horizon: int, market_data: pd.DataFrame) -> List[ExecutionSlice]:
        """Calculate VWAP-following execution schedule."""
        
        if target_quantity == 0 or time_horizon <= 0:
            return []
        
        # Get historical volume profile
        volume_profile = self._calculate_volume_profile(market_data, time_horizon)
        
        # Distribute quantity according to volume profile
        execution_slices = []
        remaining_quantity = target_quantity
        
        for i, (timestamp, volume_weight) in enumerate(volume_profile.items()):
            
            # Calculate slice quantity based on volume weight
            if i == len(volume_profile) - 1:  # Last slice gets remaining quantity
                slice_quantity = remaining_quantity
            else:
                slice_quantity = target_quantity * volume_weight
                remaining_quantity -= slice_quantity
            
            if abs(slice_quantity) > 1e-8:
                
                # Estimate market conditions
                market_conditions = self._estimate_market_conditions_vwap(symbol, timestamp, market_data)
                
                # Calculate market impact
                market_impact = self.calculate_market_impact(symbol, slice_quantity, market_conditions)
                
                # Calculate participation rate
                participation_rate = min(abs(slice_quantity) / market_conditions.volume, self.max_participation)
                
                execution_slice = ExecutionSlice(
                    timestamp=timestamp,
                    symbol=symbol,
                    quantity=slice_quantity,
                    expected_price=market_conditions.last_price,
                    market_impact=market_impact,
                    timing_risk=market_conditions.volatility * abs(slice_quantity) * market_conditions.last_price,
                    participation_rate=participation_rate
                )
                
                execution_slices.append(execution_slice)
        
        return execution_slices
    
    def calculate_market_impact(self, symbol: str, quantity: float, 
                              market_conditions: MarketMicrostructure) -> float:
        """Calculate market impact for VWAP execution."""
        
        if abs(quantity) < 1e-8:
            return 0.0
        
        # Simple linear impact model for VWAP
        participation_rate = abs(quantity) / max(market_conditions.volume, 1000)
        participation_rate = min(participation_rate, self.max_participation)
        
        # Base impact: 0.1% for 5% participation
        base_impact = 0.001 * (participation_rate / 0.05)
        
        # Volatility adjustment
        vol_adjustment = market_conditions.volatility / 0.20
        
        # Total impact
        impact_fraction = base_impact * vol_adjustment
        impact_price = impact_fraction * market_conditions.last_price
        
        return impact_price if quantity > 0 else -impact_price
    
    def _calculate_volume_profile(self, market_data: pd.DataFrame, time_horizon: int) -> Dict[pd.Timestamp, float]:
        """Calculate normalized volume profile for execution scheduling."""
        
        if 'Volume' not in market_data.columns:
            # Uniform distribution if no volume data
            timestamps = market_data.index[-time_horizon:]
            weight = 1.0 / len(timestamps)
            return {ts: weight for ts in timestamps}
        
        # Use recent volume data to create profile
        recent_data = market_data.tail(time_horizon)
        volumes = recent_data['Volume']
        total_volume = volumes.sum()
        
        if total_volume <= 0:
            weight = 1.0 / len(recent_data)
            return {ts: weight for ts in recent_data.index}
        
        # Normalize volumes to weights
        volume_weights = {}
        for timestamp, volume in volumes.items():
            volume_weights[timestamp] = volume / total_volume
        
        return volume_weights
    
    def _estimate_market_conditions_vwap(self, symbol: str, timestamp: pd.Timestamp, 
                                        market_data: pd.DataFrame) -> MarketMicrostructure:
        """Simplified market conditions estimation for VWAP."""
        
        # Find nearest data point
        available_times = market_data.index[market_data.index <= timestamp]
        if len(available_times) == 0:
            available_times = market_data.index[:1]
        
        latest_time = available_times[-1]
        current_data = market_data.loc[latest_time]
        
        last_price = current_data['Close']
        volume = current_data.get('Volume', 1000000)
        
        # Simple volatility estimate
        lookback_data = market_data.loc[:latest_time].tail(10)
        if len(lookback_data) > 1:
            returns = lookback_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.20
        else:
            volatility = 0.20
        
        # Estimate spread
        spread = last_price * 0.001
        
        return MarketMicrostructure(
            symbol=symbol,
            timestamp=timestamp,
            bid_price=last_price - spread / 2,
            ask_price=last_price + spread / 2,
            bid_size=volume * 0.01,
            ask_size=volume * 0.01,
            last_price=last_price,
            volume=volume,
            volatility=volatility,
            momentum=0.0,
            mean_reversion=0.0,
            liquidity_score=1.0
        )

class TwapExecutionModel(ExecutionModel):
    """
    Time Weighted Average Price (TWAP) execution model.
    
    Executes equal quantities at regular time intervals.
    Simple but effective for low-impact execution.
    """
    
    def __init__(self, max_participation: float = 0.10):
        self.max_participation = max_participation
    
    def calculate_execution_schedule(self, symbol: str, target_quantity: float, 
                                   time_horizon: int, market_data: pd.DataFrame) -> List[ExecutionSlice]:
        """Calculate TWAP execution schedule with equal time slices."""
        
        if target_quantity == 0 or time_horizon <= 0:
            return []
        
        # Calculate equal quantity per slice
        quantity_per_slice = target_quantity / time_horizon
        
        execution_slices = []
        timestamps = market_data.index[-time_horizon:]
        
        for i, timestamp in enumerate(timestamps):
            
            # Last slice gets any remaining quantity due to rounding
            if i == len(timestamps) - 1:
                slice_quantity = target_quantity - (quantity_per_slice * (time_horizon - 1))
            else:
                slice_quantity = quantity_per_slice
            
            if abs(slice_quantity) > 1e-8:
                
                # Estimate market conditions
                market_conditions = self._estimate_market_conditions_twap(symbol, timestamp, market_data)
                
                # Calculate market impact
                market_impact = self.calculate_market_impact(symbol, slice_quantity, market_conditions)
                
                # Calculate participation rate
                participation_rate = min(abs(slice_quantity) / market_conditions.volume, self.max_participation)
                
                execution_slice = ExecutionSlice(
                    timestamp=timestamp,
                    symbol=symbol,
                    quantity=slice_quantity,
                    expected_price=market_conditions.last_price,
                    market_impact=market_impact,
                    timing_risk=market_conditions.volatility * abs(slice_quantity) * market_conditions.last_price * 0.5,  # Lower timing risk
                    participation_rate=participation_rate
                )
                
                execution_slices.append(execution_slice)
        
        return execution_slices
    
    def calculate_market_impact(self, symbol: str, quantity: float, 
                              market_conditions: MarketMicrostructure) -> float:
        """Calculate market impact for TWAP execution (typically lower)."""
        
        if abs(quantity) < 1e-8:
            return 0.0
        
        # Conservative impact model for TWAP
        participation_rate = abs(quantity) / max(market_conditions.volume, 1000)
        participation_rate = min(participation_rate, self.max_participation)
        
        # Lower base impact: 0.05% for 5% participation
        base_impact = 0.0005 * (participation_rate / 0.05)
        
        # Volatility adjustment
        vol_adjustment = market_conditions.volatility / 0.20
        
        # Total impact
        impact_fraction = base_impact * vol_adjustment
        impact_price = impact_fraction * market_conditions.last_price
        
        return impact_price if quantity > 0 else -impact_price
    
    def _estimate_market_conditions_twap(self, symbol: str, timestamp: pd.Timestamp, 
                                        market_data: pd.DataFrame) -> MarketMicrostructure:
        """Simplified market conditions for TWAP."""
        
        # Find nearest data point
        available_times = market_data.index[market_data.index <= timestamp]
        if len(available_times) == 0:
            available_times = market_data.index[:1]
        
        latest_time = available_times[-1]
        current_data = market_data.loc[latest_time]
        
        last_price = current_data['Close']
        volume = current_data.get('Volume', 1000000)
        
        # Conservative volatility estimate
        volatility = 0.15  # Assume 15% volatility for TWAP
        
        # Tight spread for TWAP
        spread = last_price * 0.0005  # 5 bps
        
        return MarketMicrostructure(
            symbol=symbol,
            timestamp=timestamp,
            bid_price=last_price - spread / 2,
            ask_price=last_price + spread / 2,
            bid_size=volume * 0.02,
            ask_size=volume * 0.02,
            last_price=last_price,
            volume=volume,
            volatility=volatility,
            momentum=0.0,
            mean_reversion=0.0,
            liquidity_score=1.0
        )

def create_execution_model(model_type: str, **kwargs) -> ExecutionModel:
    """Factory function to create execution models."""
    
    if model_type.upper() == 'ALMGREN_CHRISS':
        return AlmgrenChrissModel(**kwargs)
    elif model_type.upper() == 'VWAP':
        return VWAPExecutionModel(**kwargs)
    elif model_type.upper() == 'TWAP':
        return TwapExecutionModel(**kwargs)
    else:
        raise ValueError(f"Unknown execution model: {model_type}")