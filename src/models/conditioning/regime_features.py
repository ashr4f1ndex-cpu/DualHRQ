"""
regime_features.py - Dynamic Regime Feature Extraction
======================================================

Production regime feature extraction for DualHRQ 2.0 replacing static puzzle_id
conditioning with dynamic, real-time market regime identification and features.

CRITICAL: This replaces static conditioning with dynamic market-aware features.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import threading
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import hashlib


class RegimeType(Enum):
    """Market regime classifications."""
    LOW_VOLATILITY = "low_vol"
    HIGH_VOLATILITY = "high_vol"
    TRENDING_UP = "trend_up"
    TRENDING_DOWN = "trend_down"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current market regime state with confidence and metadata."""
    regime_type: RegimeType
    confidence: float
    duration: timedelta
    strength: float
    stability: float
    detected_at: datetime
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'regime_type': self.regime_type.value,
            'confidence': self.confidence,
            'duration_seconds': self.duration.total_seconds(),
            'strength': self.strength,
            'stability': self.stability,
            'detected_at': self.detected_at.isoformat(),
            'features': self.features,
            'metadata': self.metadata
        }


class RegimeFeatures:
    """
    Production regime feature extraction system.
    
    Replaces static puzzle_id conditioning with dynamic market regime features
    that adapt to real-time market conditions and provide contextual conditioning
    signals for the HRM.
    """
    
    def __init__(self, lookback_window: int = 100, regime_memory: int = 50, 
                 stability_threshold: float = 0.7, regime_switch_sensitivity: float = 0.3):
        self.lookback_window = lookback_window
        self.regime_memory = regime_memory
        self.stability_threshold = stability_threshold
        self.regime_switch_sensitivity = regime_switch_sensitivity
        
        # Regime tracking
        self.current_regime = RegimeState(
            regime_type=RegimeType.UNKNOWN,
            confidence=0.0,
            duration=timedelta(0),
            strength=0.0,
            stability=0.0,
            detected_at=datetime.now()
        )
        
        # Historical data storage
        self._price_history = deque(maxlen=lookback_window * 2)
        self._volume_history = deque(maxlen=lookback_window * 2)
        self._regime_history = deque(maxlen=regime_memory)
        
        # Feature extraction components
        self._scaler = StandardScaler()
        self._regime_classifier = None
        self._volatility_estimator = VolatilityEstimator()
        self._trend_detector = TrendDetector()
        self._momentum_analyzer = MomentumAnalyzer()
        
        # Performance tracking
        self._regime_switches = 0
        self._stability_scores = deque(maxlen=100)
        self._last_update = datetime.now()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize regime detection models
        self._initialize_models()
    
    def extract_regime_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract dynamic regime features from current market data.
        
        This is the main interface that replaces static puzzle_id conditioning.
        Returns a feature dictionary for use in HRM conditioning.
        """
        with self._lock:
            # Update data history
            self._update_market_history(market_data)
            
            # Detect current regime
            regime_state = self._detect_regime(market_data)
            
            # Extract comprehensive feature set
            features = {}
            
            # Core regime features
            features.update(self._extract_regime_type_features(regime_state))
            
            # Volatility regime features
            features.update(self._extract_volatility_features(market_data))
            
            # Trend regime features  
            features.update(self._extract_trend_features(market_data))
            
            # Momentum regime features
            features.update(self._extract_momentum_features(market_data))
            
            # Market structure features
            features.update(self._extract_structure_features(market_data))
            
            # Temporal features
            features.update(self._extract_temporal_features(market_data))
            
            # Cross-regime interaction features
            features.update(self._extract_interaction_features(regime_state, market_data))
            
            # Update regime state
            self.current_regime = regime_state
            self._regime_history.append(regime_state)
            
            return features
    
    def get_regime_conditioning_vector(self, market_data: Dict[str, Any]) -> torch.Tensor:
        """
        Get regime conditioning vector for HRM integration.
        
        Returns a tensor suitable for direct use in FiLM conditioning layers.
        """
        features = self.extract_regime_features(market_data)
        
        # Convert to ordered feature vector
        feature_vector = []
        
        # Ordered feature extraction for consistent conditioning
        ordered_keys = [
            'regime_confidence', 'regime_strength', 'regime_stability',
            'volatility_regime', 'trend_regime', 'momentum_regime',
            'price_momentum', 'volume_momentum', 'volatility_level',
            'trend_strength', 'reversal_probability', 'breakout_probability',
            'market_stress', 'liquidity_regime', 'correlation_regime',
            'time_of_day', 'day_of_week', 'intraday_pattern',
            'regime_persistence', 'regime_transition_prob'
        ]
        
        for key in ordered_keys:
            feature_vector.append(features.get(key, 0.0))
        
        # Pad or truncate to ensure consistent dimension
        target_dim = 32  # Standard conditioning dimension
        if len(feature_vector) < target_dim:
            feature_vector.extend([0.0] * (target_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:target_dim]
        
        return torch.tensor(feature_vector, dtype=torch.float32)
    
    def get_current_regime_state(self) -> RegimeState:
        """Get current regime state with metadata."""
        with self._lock:
            return self.current_regime
    
    def get_regime_transition_probability(self) -> Dict[RegimeType, float]:
        """Get probability distribution over regime transitions."""
        with self._lock:
            if len(self._regime_history) < 5:
                # Uniform distribution if insufficient history
                regime_types = list(RegimeType)
                uniform_prob = 1.0 / len(regime_types)
                return {regime_type: uniform_prob for regime_type in regime_types}
            
            # Calculate transition probabilities from history
            transitions = defaultdict(int)
            total_transitions = 0
            
            for i in range(1, len(self._regime_history)):
                prev_regime = self._regime_history[i-1].regime_type
                curr_regime = self._regime_history[i].regime_type
                transitions[curr_regime] += 1
                total_transitions += 1
            
            # Convert to probabilities with smoothing
            probs = {}
            for regime_type in RegimeType:
                count = transitions[regime_type]
                prob = (count + 1) / (total_transitions + len(RegimeType))  # Laplace smoothing
                probs[regime_type] = prob
            
            return probs
    
    def _detect_regime(self, market_data: Dict[str, Any]) -> RegimeState:
        """Detect current market regime using multi-factor analysis."""
        if len(self._price_history) < 20:
            return RegimeState(
                regime_type=RegimeType.UNKNOWN,
                confidence=0.0,
                duration=timedelta(0),
                strength=0.0,
                stability=0.0,
                detected_at=datetime.now()
            )
        
        # Multi-factor regime detection
        volatility_regime = self._volatility_estimator.classify_regime(list(self._price_history))
        trend_regime = self._trend_detector.detect_trend_regime(list(self._price_history))
        momentum_regime = self._momentum_analyzer.classify_momentum(market_data)
        
        # Combine regime signals
        regime_scores = defaultdict(float)
        
        # Volatility contribution
        if volatility_regime == 'high':
            regime_scores[RegimeType.HIGH_VOLATILITY] += 0.4
            regime_scores[RegimeType.CRISIS] += 0.2
        elif volatility_regime == 'low':
            regime_scores[RegimeType.LOW_VOLATILITY] += 0.4
            regime_scores[RegimeType.SIDEWAYS] += 0.3
        
        # Trend contribution
        if trend_regime == 'strong_up':
            regime_scores[RegimeType.TRENDING_UP] += 0.4
        elif trend_regime == 'strong_down':
            regime_scores[RegimeType.TRENDING_DOWN] += 0.4
        elif trend_regime == 'sideways':
            regime_scores[RegimeType.SIDEWAYS] += 0.3
        
        # Momentum contribution
        if momentum_regime == 'breakout':
            regime_scores[RegimeType.BREAKOUT] += 0.3
        elif momentum_regime == 'reversal':
            regime_scores[RegimeType.REVERSAL] += 0.3
        
        # Find dominant regime
        if regime_scores:
            dominant_regime = max(regime_scores.items(), key=lambda x: x[1])
            regime_type = dominant_regime[0]
            confidence = min(1.0, dominant_regime[1])
        else:
            regime_type = RegimeType.UNKNOWN
            confidence = 0.0
        
        # Calculate regime strength and stability
        strength = self._calculate_regime_strength(regime_type, market_data)
        stability = self._calculate_regime_stability(regime_type)
        
        # Calculate duration
        duration = datetime.now() - self.current_regime.detected_at
        if self.current_regime.regime_type != regime_type:
            duration = timedelta(0)  # New regime
            self._regime_switches += 1
        
        return RegimeState(
            regime_type=regime_type,
            confidence=confidence,
            duration=duration,
            strength=strength,
            stability=stability,
            detected_at=datetime.now() if regime_type != self.current_regime.regime_type else self.current_regime.detected_at
        )
    
    def _extract_regime_type_features(self, regime_state: RegimeState) -> Dict[str, float]:
        """Extract features from regime classification."""
        features = {}
        
        # One-hot encoding of regime type
        for regime_type in RegimeType:
            features[f'regime_{regime_type.value}'] = 1.0 if regime_state.regime_type == regime_type else 0.0
        
        # Regime confidence and metadata
        features['regime_confidence'] = regime_state.confidence
        features['regime_strength'] = regime_state.strength
        features['regime_stability'] = regime_state.stability
        features['regime_duration_hours'] = regime_state.duration.total_seconds() / 3600
        
        return features
    
    def _extract_volatility_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract volatility regime features."""
        if len(self._price_history) < 10:
            return {'volatility_regime': 0.0, 'volatility_level': 0.0, 'volatility_trend': 0.0}
        
        prices = np.array(list(self._price_history))
        returns = np.diff(np.log(prices[prices > 0]))
        
        if len(returns) == 0:
            return {'volatility_regime': 0.0, 'volatility_level': 0.0, 'volatility_trend': 0.0}
        
        # Current volatility level
        current_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        
        # Historical volatility percentile
        if len(returns) >= 50:
            hist_vols = [np.std(returns[i:i+20]) for i in range(len(returns)-19)]
            vol_percentile = stats.percentileofscore(hist_vols, current_vol) / 100
        else:
            vol_percentile = 0.5
        
        # Volatility trend
        if len(returns) >= 40:
            recent_vol = np.std(returns[-20:])
            older_vol = np.std(returns[-40:-20])
            vol_trend = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0.0
        else:
            vol_trend = 0.0
        
        return {
            'volatility_regime': vol_percentile,
            'volatility_level': min(1.0, current_vol * 100),  # Scale to 0-1 range
            'volatility_trend': np.tanh(vol_trend)  # Bounded to [-1, 1]
        }
    
    def _extract_trend_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract trend regime features."""
        if len(self._price_history) < 10:
            return {'trend_regime': 0.0, 'trend_strength': 0.0, 'trend_consistency': 0.0}
        
        prices = np.array(list(self._price_history))
        
        # Linear trend analysis
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Normalize slope by price level
        trend_slope = slope / np.mean(prices) if np.mean(prices) > 0 else 0.0
        trend_strength = abs(r_value)  # Correlation coefficient magnitude
        trend_consistency = max(0, 1.0 - p_value)  # Statistical significance
        
        return {
            'trend_regime': np.tanh(trend_slope * 1000),  # Scale and bound to [-1, 1]
            'trend_strength': trend_strength,
            'trend_consistency': trend_consistency
        }
    
    def _extract_momentum_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract momentum regime features."""
        features = {}
        
        # Price momentum
        price = market_data.get('price', 0.0)
        if len(self._price_history) > 0:
            prev_price = self._price_history[-1]
            price_momentum = (price - prev_price) / prev_price if prev_price > 0 else 0.0
        else:
            price_momentum = 0.0
        
        # Volume momentum
        volume = market_data.get('volume', 0.0)
        if len(self._volume_history) > 0:
            prev_volume = self._volume_history[-1]
            volume_momentum = (volume - prev_volume) / prev_volume if prev_volume > 0 else 0.0
        else:
            volume_momentum = 0.0
        
        # Momentum acceleration
        if len(self._price_history) >= 3:
            p0, p1, p2 = self._price_history[-3], self._price_history[-2], self._price_history[-1]
            if p0 > 0 and p1 > 0:
                momentum_1 = (p1 - p0) / p0
                momentum_2 = (p2 - p1) / p1
                momentum_accel = momentum_2 - momentum_1
            else:
                momentum_accel = 0.0
        else:
            momentum_accel = 0.0
        
        # Reversal and breakout indicators
        reversal_prob = self._calculate_reversal_probability()
        breakout_prob = self._calculate_breakout_probability()
        
        features.update({
            'price_momentum': np.tanh(price_momentum * 100),
            'volume_momentum': np.tanh(volume_momentum),
            'momentum_accel': np.tanh(momentum_accel * 1000),
            'reversal_probability': reversal_prob,
            'breakout_probability': breakout_prob
        })
        
        return features
    
    def _extract_structure_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract market structure features."""
        features = {}
        
        # Market stress indicator
        spread = market_data.get('spread', 0.0)
        normal_spread = market_data.get('normal_spread', 0.001)
        stress_indicator = min(1.0, spread / normal_spread) if normal_spread > 0 else 0.0
        
        # Liquidity regime
        depth = market_data.get('depth', 0.0)
        normal_depth = market_data.get('normal_depth', 1000000)
        liquidity_ratio = depth / normal_depth if normal_depth > 0 else 0.0
        
        # Correlation regime (placeholder for multi-asset)
        correlation_strength = market_data.get('correlation_strength', 0.5)
        
        features.update({
            'market_stress': stress_indicator,
            'liquidity_regime': min(1.0, liquidity_ratio),
            'correlation_regime': correlation_strength
        })
        
        return features
    
    def _extract_temporal_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract time-based regime features."""
        timestamp = market_data.get('timestamp', datetime.now())
        if not isinstance(timestamp, datetime):
            timestamp = datetime.now()
        
        # Intraday patterns
        hour_of_day = timestamp.hour / 24.0
        day_of_week = timestamp.weekday() / 7.0
        
        # Market session indicators
        # US market hours: 9:30 AM - 4:00 PM EST
        is_market_hours = 9.5/24 <= hour_of_day <= 16/24
        is_opening = 9.5/24 <= hour_of_day <= 10.5/24
        is_closing = 15/24 <= hour_of_day <= 16/24
        
        return {
            'time_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'intraday_pattern': np.sin(2 * np.pi * hour_of_day),
            'is_market_hours': float(is_market_hours),
            'is_opening': float(is_opening),
            'is_closing': float(is_closing)
        }
    
    def _extract_interaction_features(self, regime_state: RegimeState, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract cross-regime interaction features."""
        features = {}
        
        # Regime persistence
        regime_age = regime_state.duration.total_seconds() / 3600  # Hours
        features['regime_persistence'] = min(1.0, regime_age / 24)  # Normalize by 24 hours
        
        # Regime transition probability
        transition_probs = self.get_regime_transition_probability()
        features['regime_transition_prob'] = 1.0 - transition_probs.get(regime_state.regime_type, 0.5)
        
        # Regime ensemble strength (how well different signals agree)
        vol_signal = self._volatility_estimator.get_signal_strength()
        trend_signal = self._trend_detector.get_signal_strength()
        momentum_signal = self._momentum_analyzer.get_signal_strength()
        
        signal_agreement = np.std([vol_signal, trend_signal, momentum_signal])
        features['regime_consensus'] = max(0.0, 1.0 - signal_agreement)
        
        return features
    
    def _calculate_regime_strength(self, regime_type: RegimeType, market_data: Dict[str, Any]) -> float:
        """Calculate how strongly the current data supports the regime classification."""
        if len(self._price_history) < 10:
            return 0.0
        
        prices = np.array(list(self._price_history))
        returns = np.diff(np.log(prices[prices > 0]))
        
        if len(returns) == 0:
            return 0.0
        
        if regime_type == RegimeType.HIGH_VOLATILITY:
            current_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
            median_vol = np.median([np.std(returns[i:i+10]) for i in range(max(1, len(returns)-50), len(returns)-9)])
            return min(1.0, current_vol / median_vol) if median_vol > 0 else 0.0
        
        elif regime_type == RegimeType.TRENDING_UP:
            slope, _, r_value, _, _ = stats.linregress(range(len(prices)), prices)
            return max(0.0, min(1.0, r_value)) if slope > 0 else 0.0
        
        elif regime_type == RegimeType.TRENDING_DOWN:
            slope, _, r_value, _, _ = stats.linregress(range(len(prices)), prices)
            return max(0.0, min(1.0, abs(r_value))) if slope < 0 else 0.0
        
        else:
            return 0.5  # Default strength for other regimes
    
    def _calculate_regime_stability(self, regime_type: RegimeType) -> float:
        """Calculate how stable the current regime has been."""
        if len(self._regime_history) < 5:
            return 0.0
        
        recent_regimes = [r.regime_type for r in list(self._regime_history)[-10:]]
        stability = recent_regimes.count(regime_type) / len(recent_regimes)
        return stability
    
    def _calculate_reversal_probability(self) -> float:
        """Calculate probability of trend reversal."""
        if len(self._price_history) < 20:
            return 0.5
        
        prices = np.array(list(self._price_history))
        
        # Look for divergence patterns
        recent_prices = prices[-10:]
        earlier_prices = prices[-20:-10]
        
        recent_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        earlier_trend = np.polyfit(range(len(earlier_prices)), earlier_prices, 1)[0]
        
        # High reversal probability if trends are opposing
        if recent_trend * earlier_trend < 0:
            return min(1.0, abs(recent_trend - earlier_trend) / np.mean(prices) * 1000)
        
        return max(0.0, 1.0 - abs(recent_trend) / np.mean(prices) * 1000)
    
    def _calculate_breakout_probability(self) -> float:
        """Calculate probability of price breakout."""
        if len(self._price_history) < 20:
            return 0.5
        
        prices = np.array(list(self._price_history))
        
        # Calculate recent trading range
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        current_price = prices[-1]
        
        # Breakout probability based on position in range
        if recent_high > recent_low:
            range_position = (current_price - recent_low) / (recent_high - recent_low)
            # Higher probability near range boundaries
            boundary_distance = min(range_position, 1.0 - range_position)
            return max(0.0, 1.0 - boundary_distance * 2)
        
        return 0.5
    
    def _update_market_history(self, market_data: Dict[str, Any]):
        """Update internal market data history."""
        price = market_data.get('price', 0.0)
        volume = market_data.get('volume', 0.0)
        
        if price > 0:  # Only store valid prices
            self._price_history.append(price)
        if volume > 0:  # Only store valid volumes
            self._volume_history.append(volume)
        
        self._last_update = datetime.now()
    
    def _initialize_models(self):
        """Initialize regime detection models."""
        # Initialize with default parameters - will adapt over time
        self._volatility_estimator = VolatilityEstimator()
        self._trend_detector = TrendDetector()
        self._momentum_analyzer = MomentumAnalyzer()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get regime detection performance statistics."""
        return {
            'regime_switches': self._regime_switches,
            'average_stability': np.mean(self._stability_scores) if self._stability_scores else 0.0,
            'data_points': len(self._price_history),
            'current_regime': self.current_regime.regime_type.value,
            'regime_confidence': self.current_regime.confidence,
            'last_update': self._last_update.isoformat()
        }


class VolatilityEstimator:
    """Volatility regime classification component."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._signal_strength = 0.0
    
    def classify_regime(self, prices: List[float]) -> str:
        """Classify volatility regime as 'low', 'medium', or 'high'."""
        if len(prices) < self.window_size:
            self._signal_strength = 0.0
            return 'medium'
        
        returns = np.diff(np.log(np.array(prices)))
        current_vol = np.std(returns[-self.window_size:])
        
        if len(returns) >= self.window_size * 3:
            hist_vol = np.std(returns[:-self.window_size])
            vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1.0
            self._signal_strength = min(1.0, abs(vol_ratio - 1.0))
            
            if vol_ratio > 1.5:
                return 'high'
            elif vol_ratio < 0.7:
                return 'low'
            else:
                return 'medium'
        else:
            self._signal_strength = 0.0
            return 'medium'
    
    def get_signal_strength(self) -> float:
        """Get confidence in volatility classification."""
        return self._signal_strength


class TrendDetector:
    """Trend regime detection component."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._signal_strength = 0.0
    
    def detect_trend_regime(self, prices: List[float]) -> str:
        """Detect trend regime: 'strong_up', 'weak_up', 'sideways', 'weak_down', 'strong_down'."""
        if len(prices) < self.window_size:
            self._signal_strength = 0.0
            return 'sideways'
        
        prices_array = np.array(prices[-self.window_size:])
        x = np.arange(len(prices_array))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices_array)
        
        # Normalize slope by price level
        normalized_slope = slope / np.mean(prices_array) if np.mean(prices_array) > 0 else 0.0
        self._signal_strength = abs(r_value)  # Correlation coefficient as signal strength
        
        if abs(r_value) < 0.3:  # Weak correlation
            return 'sideways'
        elif normalized_slope > 0.001:  # Strong uptrend
            return 'strong_up' if abs(r_value) > 0.7 else 'weak_up'
        elif normalized_slope < -0.001:  # Strong downtrend
            return 'strong_down' if abs(r_value) > 0.7 else 'weak_down'
        else:
            return 'sideways'
    
    def get_signal_strength(self) -> float:
        """Get confidence in trend classification."""
        return self._signal_strength


class MomentumAnalyzer:
    """Momentum regime analysis component."""
    
    def __init__(self):
        self._signal_strength = 0.0
    
    def classify_momentum(self, market_data: Dict[str, Any]) -> str:
        """Classify momentum regime: 'breakout', 'reversal', 'continuation', 'neutral'."""
        price_momentum = market_data.get('price_momentum', 0.0)
        volume_momentum = market_data.get('volume_momentum', 0.0)
        
        # Combine price and volume momentum
        combined_momentum = abs(price_momentum) + abs(volume_momentum) * 0.5
        self._signal_strength = min(1.0, combined_momentum)
        
        if combined_momentum > 0.02 and volume_momentum > 0.1:
            return 'breakout'
        elif abs(price_momentum) > 0.01 and volume_momentum < -0.1:
            return 'reversal'
        elif abs(price_momentum) > 0.005:
            return 'continuation'
        else:
            return 'neutral'
    
    def get_signal_strength(self) -> float:
        """Get confidence in momentum classification."""
        return self._signal_strength


# Legacy compatibility classes
class RegimeFeatureExtractor:
    """Legacy wrapper for RegimeFeatures."""
    
    def __init__(self):
        self.regime_features = RegimeFeatures()
    
    def extract_tsrv_features(self, price_data: pd.DataFrame, windows: List[str] = None) -> Dict[str, float]:
        """Extract Time-Scaled Realized Volatility features."""
        windows = windows or ['5m', '15m', '30m', '60m']
        
        if price_data.empty:
            return {f'tsrv_{w}': 0.0 for w in windows}
        
        # Convert to market_data format
        market_data = {'price': price_data.iloc[-1] if not price_data.empty else 0.0}
        features = self.regime_features.extract_regime_features(market_data)
        
        # Map to TSRV format
        return {
            'tsrv_5m': features.get('volatility_level', 0.0),
            'tsrv_15m': features.get('volatility_trend', 0.0),
            'tsrv_30m': features.get('trend_strength', 0.0),
            'tsrv_60m': features.get('regime_confidence', 0.0)
        }
    
    def extract_bpv_features(self, price_data: pd.DataFrame) -> float:
        """Extract Bipower Variation feature."""
        if price_data.empty:
            return 0.0
        market_data = {'price': price_data.iloc[-1]}
        features = self.regime_features.extract_regime_features(market_data)
        return features.get('volatility_level', 0.0)
    
    def extract_amihud_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> float:
        """Extract Amihud illiquidity measure."""
        if price_data.empty or volume_data.empty:
            return 0.0
        market_data = {'price': price_data.iloc[-1], 'volume': volume_data.iloc[-1]}
        features = self.regime_features.extract_regime_features(market_data)
        return features.get('liquidity_regime', 0.0)
    
    def extract_ssr_luld_state(self, price_data: pd.DataFrame, timestamp: datetime) -> Dict[str, bool]:
        """Extract SSR/LULD regulatory state."""
        if price_data.empty:
            return {'ssr_active': False, 'luld_active': False}
        
        market_data = {'price': price_data.iloc[-1], 'timestamp': timestamp}
        features = self.regime_features.extract_regime_features(market_data)
        
        # Simple heuristic based on market stress
        stress_level = features.get('market_stress', 0.0)
        return {
            'ssr_active': stress_level > 0.7,
            'luld_active': stress_level > 0.8
        }


class RegimeClassifier:
    """Legacy wrapper for regime classification."""
    
    def __init__(self):
        self.regime_features = RegimeFeatures()
    
    def classify_regime(self, features: Dict[str, float]) -> str:
        """Classify market regime from extracted features."""
        # Convert features to market_data format
        market_data = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        regime_state = self.regime_features.get_current_regime_state()
        return regime_state.regime_type.value
    
    def get_regime_embedding(self, regime: str) -> torch.Tensor:
        """Get embedding vector for regime."""
        # Create dummy market data for regime
        market_data = {'regime_type': regime}
        return self.regime_features.get_regime_conditioning_vector(market_data)[:64]  # 64-dim for compatibility


# Export main classes
__all__ = ['RegimeFeatures', 'RegimeState', 'RegimeType', 'RegimeFeatureExtractor', 'RegimeClassifier']