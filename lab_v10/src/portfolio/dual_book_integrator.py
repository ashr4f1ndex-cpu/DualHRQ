"""
Dual-Book Portfolio Integration System

Advanced multi-strategy portfolio management:
- HRM-powered signal generation and combination
- Dynamic risk budgeting with regime detection
- Multi-asset class optimization (options + equities)
- Real-time portfolio rebalancing
- Advanced risk management with Greeks hedging
- Performance attribution across strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class StrategySignal:
    """Individual strategy signal with metadata."""
    timestamp: pd.Timestamp
    strategy_name: str
    signal_strength: float  # -1 to +1
    confidence: float  # 0 to 1
    asset_class: str  # 'options', 'equity'
    target_symbol: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PortfolioAllocation:
    """Portfolio allocation with risk metrics."""
    timestamp: pd.Timestamp
    allocations: Dict[str, float]  # strategy_name -> weight
    expected_return: float
    expected_volatility: float
    risk_contributions: Dict[str, float]
    diversification_ratio: float

class SignalCombiner(ABC):
    """Abstract base for signal combination methods."""
    
    @abstractmethod
    def combine_signals(self, signals: List[StrategySignal]) -> StrategySignal:
        """Combine multiple signals into single composite signal."""
        pass

class HRMEnhancedCombiner(SignalCombiner):
    """HRM-enhanced signal combination using learned weightings."""
    
    def __init__(self, hrm_model=None, lookback_window: int = 252):
        self.hrm_model = hrm_model
        self.lookback_window = lookback_window
        self.signal_history = []
        self.performance_history = []
        
    def combine_signals(self, signals: List[StrategySignal]) -> StrategySignal:
        """Combine signals using HRM-learned dynamic weights."""
        
        if not signals:
            return None
        
        # If HRM model available, use it for dynamic weighting
        if self.hrm_model is not None:
            weights = self._hrm_dynamic_weights(signals)
        else:
            # Fallback to confidence-weighted combination
            weights = self._confidence_weighted_combination(signals)
        
        # Combine signals
        combined_strength = sum(w * s.signal_strength for w, s in zip(weights, signals))
        combined_confidence = sum(w * s.confidence for w, s in zip(weights, signals))
        
        # Create combined signal
        combined_signal = StrategySignal(
            timestamp=signals[0].timestamp,
            strategy_name="combined",
            signal_strength=np.tanh(combined_strength),  # Bounded [-1, 1]
            confidence=min(combined_confidence, 1.0),
            asset_class="mixed",
            target_symbol="portfolio",
            metadata={
                'component_signals': len(signals),
                'weights': weights,
                'combination_method': 'hrm_enhanced'
            }
        )
        
        return combined_signal
    
    def _hrm_dynamic_weights(self, signals: List[StrategySignal]) -> List[float]:
        """Use HRM to determine dynamic signal weights."""
        
        # Extract features for HRM
        features = []
        for signal in signals:
            features.extend([
                signal.signal_strength,
                signal.confidence,
                1.0 if signal.asset_class == 'options' else 0.0,
                1.0 if signal.asset_class == 'equity' else 0.0
            ])
        
        # Pad/truncate to fixed size
        feature_vector = np.array(features + [0] * max(0, 32 - len(features)))[:32]
        
        try:
            # Use HRM H-module for slow strategic weighting
            # This would interface with the actual HRM model
            with torch.no_grad():
                weights_logits = self.hrm_model.h_module(
                    torch.tensor(feature_vector).unsqueeze(0)
                ).squeeze()
                
                # Softmax to get valid weights
                weights = torch.softmax(weights_logits[:len(signals)], dim=0).numpy()
        except:
            # Fallback if HRM fails
            weights = self._confidence_weighted_combination(signals)
        
        return weights.tolist()
    
    def _confidence_weighted_combination(self, signals: List[StrategySignal]) -> List[float]:
        """Simple confidence-weighted signal combination."""
        
        confidences = [s.confidence for s in signals]
        total_confidence = sum(confidences)
        
        if total_confidence > 0:
            weights = [c / total_confidence for c in confidences]
        else:
            weights = [1.0 / len(signals)] * len(signals)
        
        return weights

class RiskBudgetOptimizer:
    """Advanced risk budgeting for multi-strategy portfolios."""
    
    def __init__(self, target_volatility: float = 0.15):
        self.target_volatility = target_volatility
        self.covariance_estimator = 'sample'  # 'sample', 'ledoit_wolf', 'shrunk'
        
    def optimize_risk_budget(self, expected_returns: pd.Series, 
                           covariance_matrix: pd.DataFrame,
                           risk_budget: Dict[str, float] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights to achieve target risk budget.
        
        Args:
            expected_returns: Expected returns for each strategy
            covariance_matrix: Covariance matrix of strategy returns
            risk_budget: Target risk contribution for each strategy
            
        Returns:
            Optimal portfolio weights
        """
        
        n_strategies = len(expected_returns)
        
        if risk_budget is None:
            # Equal risk contribution
            risk_budget = {asset: 1.0/n_strategies for asset in expected_returns.index}
        
        # Convert to arrays for optimization
        mu = expected_returns.values
        sigma = covariance_matrix.values
        target_rc = np.array([risk_budget.get(asset, 1.0/n_strategies) 
                             for asset in expected_returns.index])
        
        # Objective function: minimize sum of squared deviations from target risk contributions
        def objective(weights):
            weights = np.abs(weights)  # Ensure positive weights
            weights = weights / weights.sum()  # Normalize
            
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights.T @ sigma @ weights)
            marginal_contrib = (sigma @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Penalize deviations from target risk budget
            rc_error = np.sum((risk_contrib - target_rc) ** 2)
            
            # Add penalty for extreme weights
            concentration_penalty = 10 * np.sum(weights ** 3)
            
            return rc_error + concentration_penalty
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: w + 0.01},  # Non-negative (with small buffer)
            {'type': 'ineq', 'fun': lambda w: 0.5 - w}    # Max 50% per strategy
        ]
        
        # Initial guess
        x0 = np.ones(n_strategies) / n_strategies
        
        # Optimization
        result = minimize(
            objective, x0, 
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = np.abs(result.x)
            optimal_weights = optimal_weights / optimal_weights.sum()
            
            return {asset: weight for asset, weight in 
                   zip(expected_returns.index, optimal_weights)}
        else:
            logger.warning("Risk budget optimization failed, using equal weights")
            return {asset: 1.0/n_strategies for asset in expected_returns.index}
    
    def calculate_risk_contributions(self, weights: Dict[str, float],
                                   covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate actual risk contributions for given weights."""
        
        w = np.array([weights[asset] for asset in covariance_matrix.index])
        sigma = covariance_matrix.values
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(w.T @ sigma @ w)
        
        # Marginal risk contributions
        marginal_contrib = (sigma @ w) / portfolio_vol
        
        # Risk contributions
        risk_contrib = w * marginal_contrib / portfolio_vol
        
        return {asset: rc for asset, rc in 
               zip(covariance_matrix.index, risk_contrib)}

class RegimeDetector:
    """Volatility and correlation regime detection for dynamic allocation."""
    
    def __init__(self, lookback_window: int = 63):
        self.lookback_window = lookback_window
        self.regimes = ['low_vol', 'high_vol', 'crisis']
        
    def detect_current_regime(self, returns_data: pd.DataFrame) -> Dict[str, any]:
        """
        Detect current market regime based on volatility and correlation.
        
        Args:
            returns_data: DataFrame with strategy returns
            
        Returns:
            Dictionary with regime information
        """
        
        if len(returns_data) < self.lookback_window:
            return {'regime': 'low_vol', 'confidence': 0.5, 'indicators': {}}
        
        recent_data = returns_data.tail(self.lookback_window)
        
        # Calculate regime indicators
        indicators = {}
        
        # 1. Realized volatility
        portfolio_returns = recent_data.mean(axis=1)
        realized_vol = portfolio_returns.std() * np.sqrt(252)
        indicators['realized_volatility'] = realized_vol
        
        # 2. Average pairwise correlation
        corr_matrix = recent_data.corr()
        n = len(corr_matrix)
        avg_correlation = (corr_matrix.values.sum() - n) / (n * (n - 1))
        indicators['average_correlation'] = avg_correlation
        
        # 3. Maximum drawdown
        cumulative_returns = portfolio_returns.cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - running_max
        max_drawdown = drawdowns.min()
        indicators['max_drawdown'] = max_drawdown
        
        # 4. VIX-like volatility clustering
        vol_series = recent_data.rolling(5).std().mean(axis=1) * np.sqrt(252)
        vol_of_vol = vol_series.std()
        indicators['volatility_clustering'] = vol_of_vol
        
        # Regime classification logic
        if realized_vol > 0.25 or max_drawdown < -0.15 or avg_correlation > 0.7:
            regime = 'crisis'
            confidence = min(1.0, (realized_vol - 0.15) / 0.15 + 
                           abs(max_drawdown) / 0.15 + 
                           (avg_correlation - 0.5) / 0.3)
        elif realized_vol > 0.18 or avg_correlation > 0.5:
            regime = 'high_vol'
            confidence = min(1.0, (realized_vol - 0.12) / 0.10 + 
                           (avg_correlation - 0.3) / 0.3)
        else:
            regime = 'low_vol'
            confidence = max(0.3, 1.0 - realized_vol / 0.15)
        
        return {
            'regime': regime,
            'confidence': min(confidence, 1.0),
            'indicators': indicators,
            'regime_probabilities': self._calculate_regime_probabilities(indicators)
        }
    
    def _calculate_regime_probabilities(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Calculate probability distribution over regimes."""
        
        vol = indicators['realized_volatility']
        corr = indicators['average_correlation']
        dd = abs(indicators['max_drawdown'])
        
        # Simple logistic-style probability model
        crisis_score = 2 * vol + 3 * corr + 5 * dd
        high_vol_score = 1.5 * vol + 2 * corr + 2 * dd
        low_vol_score = max(0, 1 - vol - corr - dd)
        
        # Softmax transformation
        scores = np.array([crisis_score, high_vol_score, low_vol_score])
        exp_scores = np.exp(scores - scores.max())  # Numerical stability
        probabilities = exp_scores / exp_scores.sum()
        
        return {
            'crisis': probabilities[0],
            'high_vol': probabilities[1], 
            'low_vol': probabilities[2]
        }

class DualBookPortfolioManager:
    """Main dual-book portfolio management system."""
    
    def __init__(self, hrm_model=None, initial_capital: float = 1000000):
        self.hrm_model = hrm_model
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Core components
        self.signal_combiner = HRMEnhancedCombiner(hrm_model)
        self.risk_optimizer = RiskBudgetOptimizer()
        self.regime_detector = RegimeDetector()
        
        # Portfolio state
        self.current_allocation = {}
        self.strategy_returns = pd.DataFrame()
        self.portfolio_returns = pd.Series()
        self.risk_metrics = {}
        
        # Strategy configuration
        self.strategy_configs = {
            'options_straddles': {
                'asset_class': 'options',
                'target_allocation': 0.4,
                'max_allocation': 0.6,
                'risk_budget': 0.3
            },
            'intraday_short': {
                'asset_class': 'equity',
                'target_allocation': 0.4,
                'max_allocation': 0.6,
                'risk_budget': 0.4
            },
            'cash': {
                'asset_class': 'cash',
                'target_allocation': 0.2,
                'max_allocation': 0.5,
                'risk_budget': 0.3
            }
        }
    
    def process_signals(self, signals: List[StrategySignal]) -> PortfolioAllocation:
        """
        Process incoming signals and determine portfolio allocation.
        
        Args:
            signals: List of strategy signals
            
        Returns:
            Optimal portfolio allocation
        """
        
        if not signals:
            return self._maintain_current_allocation()
        
        timestamp = signals[0].timestamp
        
        # 1. Detect market regime
        if len(self.strategy_returns) > 0:
            regime_info = self.regime_detector.detect_current_regime(self.strategy_returns)
        else:
            regime_info = {'regime': 'low_vol', 'confidence': 0.5}
        
        # 2. Adjust strategy configurations based on regime
        adjusted_configs = self._adjust_for_regime(regime_info)
        
        # 3. Combine signals using HRM
        combined_signal = self.signal_combiner.combine_signals(signals)
        
        # 4. Calculate expected returns and covariance
        expected_returns, covariance_matrix = self._estimate_returns_and_risk()
        
        # 5. Optimize risk budget
        if expected_returns is not None and covariance_matrix is not None:
            risk_budget = {strategy: config['risk_budget'] 
                          for strategy, config in adjusted_configs.items()}
            
            optimal_weights = self.risk_optimizer.optimize_risk_budget(
                expected_returns, covariance_matrix, risk_budget
            )
        else:
            # Fallback to target allocations
            optimal_weights = {strategy: config['target_allocation'] 
                              for strategy, config in adjusted_configs.items()}
        
        # 6. Apply signal-based adjustments
        if combined_signal:
            optimal_weights = self._apply_signal_adjustments(
                optimal_weights, combined_signal, adjusted_configs
            )
        
        # 7. Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            optimal_weights, expected_returns, covariance_matrix
        )
        
        # 8. Create allocation object
        allocation = PortfolioAllocation(
            timestamp=timestamp,
            allocations=optimal_weights,
            expected_return=portfolio_metrics.get('expected_return', 0),
            expected_volatility=portfolio_metrics.get('expected_volatility', 0),
            risk_contributions=portfolio_metrics.get('risk_contributions', {}),
            diversification_ratio=portfolio_metrics.get('diversification_ratio', 1)
        )
        
        # 9. Update portfolio state
        self.current_allocation = optimal_weights
        
        logger.info(f"Portfolio allocation updated: {optimal_weights}")
        
        return allocation
    
    def _adjust_for_regime(self, regime_info: Dict) -> Dict[str, Dict]:
        """Adjust strategy configurations based on market regime."""
        
        adjusted_configs = self.strategy_configs.copy()
        regime = regime_info['regime']
        confidence = regime_info['confidence']
        
        if regime == 'crisis':
            # Crisis mode: reduce risk, increase cash
            for strategy in ['options_straddles', 'intraday_short']:
                adjusted_configs[strategy]['target_allocation'] *= (1 - 0.3 * confidence)
                adjusted_configs[strategy]['max_allocation'] *= (1 - 0.2 * confidence)
            
            # Increase cash allocation
            cash_boost = 0.3 * confidence
            adjusted_configs['cash']['target_allocation'] += cash_boost
            adjusted_configs['cash']['max_allocation'] += cash_boost
            
        elif regime == 'high_vol':
            # High volatility: moderate risk reduction
            for strategy in ['options_straddles', 'intraday_short']:
                adjusted_configs[strategy]['target_allocation'] *= (1 - 0.1 * confidence)
            
            cash_boost = 0.1 * confidence
            adjusted_configs['cash']['target_allocation'] += cash_boost
        
        # Normalize to ensure allocations sum to 1
        total_target = sum(config['target_allocation'] for config in adjusted_configs.values())
        for config in adjusted_configs.values():
            config['target_allocation'] /= total_target
        
        return adjusted_configs
    
    def _estimate_returns_and_risk(self) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
        """Estimate expected returns and covariance matrix."""
        
        if len(self.strategy_returns) < 30:  # Need minimum history
            return None, None
        
        # Use last 252 days for estimation
        lookback_data = self.strategy_returns.tail(252)
        
        # Expected returns (annualized)
        expected_returns = lookback_data.mean() * 252
        
        # Covariance matrix (annualized)
        covariance_matrix = lookback_data.cov() * 252
        
        return expected_returns, covariance_matrix
    
    def _apply_signal_adjustments(self, base_weights: Dict[str, float],
                                 signal: StrategySignal, 
                                 configs: Dict[str, Dict]) -> Dict[str, float]:
        """Apply signal-based adjustments to base allocation."""
        
        adjusted_weights = base_weights.copy()
        
        # Determine which strategies to adjust based on signal
        if signal.asset_class == 'options':
            primary_strategy = 'options_straddles'
            secondary_strategy = 'intraday_short'
        elif signal.asset_class == 'equity':
            primary_strategy = 'intraday_short'
            secondary_strategy = 'options_straddles'
        else:
            # Mixed signal - apply to both
            primary_strategy = None
            secondary_strategy = None
        
        signal_strength = abs(signal.signal_strength)
        signal_confidence = signal.confidence
        
        # Calculate adjustment magnitude
        adjustment = signal_strength * signal_confidence * 0.1  # Max 10% adjustment
        
        if primary_strategy:
            # Increase primary strategy allocation
            max_increase = (configs[primary_strategy]['max_allocation'] - 
                           adjusted_weights[primary_strategy])
            increase = min(adjustment, max_increase)
            adjusted_weights[primary_strategy] += increase
            
            # Decrease cash allocation to compensate
            adjusted_weights['cash'] = max(0, adjusted_weights['cash'] - increase)
        else:
            # Mixed signal - small adjustments to both
            for strategy in ['options_straddles', 'intraday_short']:
                if strategy in adjusted_weights:
                    max_increase = (configs[strategy]['max_allocation'] - 
                                   adjusted_weights[strategy])
                    increase = min(adjustment * 0.5, max_increase)
                    adjusted_weights[strategy] += increase
            
            # Reduce cash
            total_increase = sum(adjustment * 0.5 for _ in range(2))
            adjusted_weights['cash'] = max(0, adjusted_weights['cash'] - total_increase)
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _calculate_portfolio_metrics(self, weights: Dict[str, float],
                                   expected_returns: Optional[pd.Series],
                                   covariance_matrix: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        
        metrics = {}
        
        if expected_returns is not None and covariance_matrix is not None:
            # Align weights with data
            aligned_weights = np.array([weights.get(asset, 0) 
                                      for asset in expected_returns.index])
            
            # Expected return
            metrics['expected_return'] = (aligned_weights * expected_returns.values).sum()
            
            # Expected volatility
            portfolio_variance = aligned_weights.T @ covariance_matrix.values @ aligned_weights
            metrics['expected_volatility'] = np.sqrt(portfolio_variance)
            
            # Risk contributions
            if metrics['expected_volatility'] > 0:
                marginal_contrib = (covariance_matrix.values @ aligned_weights) / metrics['expected_volatility']
                risk_contributions = aligned_weights * marginal_contrib / metrics['expected_volatility']
                
                metrics['risk_contributions'] = {
                    asset: rc for asset, rc in zip(expected_returns.index, risk_contributions)
                }
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(covariance_matrix.values))
            weighted_avg_vol = (aligned_weights * individual_vols).sum()
            
            if metrics['expected_volatility'] > 0:
                metrics['diversification_ratio'] = weighted_avg_vol / metrics['expected_volatility']
            else:
                metrics['diversification_ratio'] = 1.0
        
        return metrics
    
    def _maintain_current_allocation(self) -> PortfolioAllocation:
        """Maintain current allocation when no new signals."""
        
        return PortfolioAllocation(
            timestamp=pd.Timestamp.now(),
            allocations=self.current_allocation,
            expected_return=0,
            expected_volatility=0,
            risk_contributions={},
            diversification_ratio=1
        )
    
    def update_strategy_returns(self, strategy_returns: Dict[str, float], 
                              timestamp: pd.Timestamp) -> None:
        """Update strategy return history."""
        
        # Add new returns to history
        new_row = pd.Series(strategy_returns, name=timestamp)
        
        if self.strategy_returns.empty:
            self.strategy_returns = pd.DataFrame([new_row])
        else:
            self.strategy_returns = pd.concat([self.strategy_returns, new_row.to_frame().T])
        
        # Calculate portfolio return
        if self.current_allocation:
            portfolio_return = sum(self.current_allocation.get(strategy, 0) * ret 
                                 for strategy, ret in strategy_returns.items())
            
            if self.portfolio_returns.empty:
                self.portfolio_returns = pd.Series([portfolio_return], index=[timestamp])
            else:
                self.portfolio_returns = pd.concat([
                    self.portfolio_returns, 
                    pd.Series([portfolio_return], index=[timestamp])
                ])
        
        # Keep only recent history (last 2 years)
        max_history = 504  # ~2 years of daily data
        if len(self.strategy_returns) > max_history:
            self.strategy_returns = self.strategy_returns.tail(max_history)
            self.portfolio_returns = self.portfolio_returns.tail(max_history)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""
        
        if self.portfolio_returns.empty:
            return {}
        
        returns = self.portfolio_returns.dropna()
        
        if len(returns) < 2:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Risk metrics
        cumulative_returns = returns.cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - running_max
        max_drawdown = drawdowns.min()
        
        # Downside metrics
        negative_returns = returns[returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
        
        # Risk-adjusted metrics
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'current_allocation': self.current_allocation,
            'observations': len(returns)
        }
    
    def rebalance_portfolio(self, target_allocation: PortfolioAllocation,
                          current_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate rebalancing trades to achieve target allocation.
        
        Args:
            target_allocation: Target portfolio allocation
            current_positions: Current position sizes
            
        Returns:
            Dictionary of trades to execute (positive = buy, negative = sell)
        """
        
        current_value = sum(current_positions.values())
        trades = {}
        
        for strategy, target_weight in target_allocation.allocations.items():
            target_value = target_weight * current_value
            current_value_strategy = current_positions.get(strategy, 0)
            
            trade_value = target_value - current_value_strategy
            trades[strategy] = trade_value
        
        # Apply minimum trade size filter
        min_trade_size = current_value * 0.01  # 1% minimum
        filtered_trades = {strategy: trade for strategy, trade in trades.items()
                          if abs(trade) >= min_trade_size}
        
        return filtered_trades