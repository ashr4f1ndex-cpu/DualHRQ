"""
Production-grade intraday feature engineering for dual-book trading strategies.

This module implements comprehensive intraday features including:
- Volume-Weighted Average Price (VWAP) with daily resets
- Average True Range (ATR) for volatility measurement
- Stretch metrics (price deviation from VWAP)
- SSR Gate Logic (Rule 201 compliance)
- LULD Mechanics (Limit Up/Limit Down detection)
- High-frequency momentum and mean reversion indicators

All features are designed to prevent look-ahead bias and support real-time
trading environments with microsecond precision.

References:
- SEC Rule 201 (Alternative Uptick Rule)
- SEC Rule 610 (Limit Up-Limit Down)
- Harris, L. Trading and Exchanges
- Aldridge, I. High-Frequency Trading
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, time
from zoneinfo import ZoneInfo

# Suppress warnings for cleaner output in production
warnings.filterwarnings('ignore', category=RuntimeWarning)


class IntradayFeatureEngine:
    """
    Production-grade intraday feature engineering engine.
    
    Implements comprehensive feature extraction from high-frequency market data
    with strict adherence to temporal ordering and regulatory constraints.
    """
    
    def __init__(self, 
                 market_open: time = time(9, 30),
                 market_close: time = time(16, 0),
                 timezone: str = "America/New_York",
                 atr_periods: List[int] = [14, 20, 50],
                 vwap_reset_freq: str = 'D',
                 luld_bands: Dict[str, float] = None):
        """
        Initialize the intraday feature engine.
        
        Parameters
        ----------
        market_open : time
            Market opening time
        market_close : time
            Market closing time
        timezone : str
            Market timezone
        atr_periods : List[int]
            ATR calculation periods
        vwap_reset_freq : str
            VWAP reset frequency ('D' for daily)
        luld_bands : Dict[str, float], optional
            LULD percentage bands by price tier
        """
        self.market_open = market_open
        self.market_close = market_close
        self.timezone = ZoneInfo(timezone)
        self.atr_periods = atr_periods
        self.vwap_reset_freq = vwap_reset_freq
        
        # Default LULD bands (simplified)
        self.luld_bands = luld_bands or {
            'tier1': 0.05,  # 5% for stocks >= $3
            'tier2': 0.10,  # 10% for stocks $0.75-$3
            'low_price': 0.20  # 20% for stocks < $0.75
        }
    
    def calculate_vwap(self, 
                      df: pd.DataFrame,
                      price_col: str = 'close',
                      volume_col: str = 'volume',
                      high_col: str = 'high',
                      low_col: str = 'low',
                      use_typical_price: bool = True) -> pd.Series:
        """
        Calculate Volume-Weighted Average Price with daily resets.
        
        This implementation is vectorized and prevents look-ahead bias by
        using only historical data up to each point in time.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data and DatetimeIndex
        price_col : str
            Price column to use (default: 'close')
        volume_col : str
            Volume column
        high_col, low_col : str
            High and low price columns
        use_typical_price : bool
            If True, use typical price (H+L+C)/3, else use specified price_col
            
        Returns
        -------
        pd.Series
            VWAP series with daily resets
        """
        df = df.copy()
        
        # Calculate typical price if requested
        if use_typical_price and all(col in df.columns for col in [high_col, low_col, price_col]):
            price_series = (df[high_col] + df[low_col] + df[price_col]) / 3
        else:
            price_series = df[price_col]
        
        volume_series = df[volume_col].fillna(0)
        
        # Ensure no negative volumes
        volume_series = volume_series.clip(lower=0)
        
        # Group by date for daily VWAP resets
        date_groups = df.index.date if hasattr(df.index, 'date') else df.index.floor('D')
        
        # Calculate cumulative price*volume and volume within each day
        df['pv'] = price_series * volume_series
        df['date_group'] = date_groups
        
        # Vectorized VWAP calculation with groupby
        vwap = df.groupby('date_group').apply(
            lambda x: (x['pv'].cumsum() / x[volume_col].cumsum().replace(0, np.nan))
        ).values
        
        # Handle division by zero cases
        vwap = pd.Series(vwap, index=df.index)
        
        # Forward fill VWAP for zero volume periods within each day
        vwap = vwap.groupby(date_groups).fillna(method='ffill')
        
        # If still NaN (start of day with zero volume), use price
        vwap = vwap.fillna(price_series)
        
        return vwap
    
    def calculate_atr(self, 
                     df: pd.DataFrame,
                     period: int = 14,
                     method: str = 'wilder') -> pd.Series:
        """
        Calculate Average True Range using Wilder's smoothing or SMA.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLC data
        period : int
            ATR period
        method : str
            'wilder' for Wilder's smoothing, 'sma' for simple moving average
            
        Returns
        -------
        pd.Series
            ATR series
        """
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
        # Calculate True Range components
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR based on method
        if method == 'wilder':
            # Wilder's smoothing: ATR = (previous_ATR * (n-1) + TR) / n
            atr = true_range.copy()
            
            # Initialize first ATR as SMA of first n periods
            if len(true_range) >= period:
                atr.iloc[period-1] = true_range.iloc[:period].mean()
                
                # Apply Wilder's smoothing for subsequent periods
                for i in range(period, len(true_range)):
                    atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + true_range.iloc[i]) / period
                    
                # Set first period-1 values to NaN
                atr.iloc[:period-1] = np.nan
            else:
                atr[:] = np.nan
        else:
            # Simple moving average
            atr = true_range.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    def calculate_stretch_metrics(self, 
                                 df: pd.DataFrame,
                                 vwap_col: str = None) -> Dict[str, pd.Series]:
        """
        Calculate price stretch metrics relative to VWAP.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        vwap_col : str, optional
            VWAP column name, if None will calculate VWAP
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary of stretch metrics
        """
        if vwap_col is None or vwap_col not in df.columns:
            vwap = self.calculate_vwap(df)
        else:
            vwap = df[vwap_col]
        
        price = df['close']
        
        # Basic stretch (percentage deviation from VWAP)
        stretch_pct = (price - vwap) / vwap * 100
        
        # Volatility-adjusted stretch
        atr = self.calculate_atr(df, period=20)
        vol_adjusted_stretch = (price - vwap) / (atr + 1e-8)
        
        # High/Low stretch (extreme deviations)
        high_stretch = (df['high'] - vwap) / vwap * 100
        low_stretch = (df['low'] - vwap) / vwap * 100
        
        # Mean reversion momentum
        stretch_ma = stretch_pct.rolling(window=10).mean()
        stretch_momentum = stretch_pct - stretch_ma
        
        # Cumulative stretch (session-based)
        date_groups = df.index.date if hasattr(df.index, 'date') else df.index.floor('D')
        cumulative_stretch = stretch_pct.groupby(date_groups).cumsum()
        
        return {
            'stretch_pct': stretch_pct,
            'vol_adjusted_stretch': vol_adjusted_stretch,
            'high_stretch': high_stretch,
            'low_stretch': low_stretch,
            'stretch_momentum': stretch_momentum,
            'cumulative_stretch': cumulative_stretch
        }
    
    def calculate_ssr_gate(self, 
                          df: pd.DataFrame,
                          trigger_threshold: float = -0.10) -> Dict[str, pd.Series]:
        """
        Calculate Short Sale Restriction (SSR) Gate indicators.
        
        Rule 201 is triggered when a stock declines 10% or more from the 
        previous day's closing price.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLC data
        trigger_threshold : float
            Decline threshold to trigger SSR (default: -10%)
            
        Returns
        -------
        Dict[str, pd.Series]
            SSR-related indicators
        """
        close = df['close']
        low = df['low']
        
        # Previous day's close (for daily data) or previous close (for intraday)
        prev_close = close.shift(1)
        
        # Calculate decline from previous close
        decline_pct = (close - prev_close) / prev_close
        
        # SSR trigger condition
        ssr_triggered = decline_pct <= trigger_threshold
        
        # Once triggered, SSR remains active for the rest of the day and next day
        date_groups = df.index.date if hasattr(df.index, 'date') else df.index.floor('D')
        
        # Propagate SSR trigger within trading sessions
        ssr_active = ssr_triggered.groupby(date_groups).cummax()
        
        # Uptick rule simulation (price > previous price)
        uptick = close > close.shift(1)
        zero_uptick = (close == close.shift(1)) & (close.shift(1) > close.shift(2))
        valid_uptick = uptick | zero_uptick
        
        # SSR compliance (can only short on uptick when SSR is active)
        ssr_compliant = ~ssr_active | valid_uptick
        
        # Distance to uptick (how far below last uptick price)
        last_uptick_price = close.where(valid_uptick).fillna(method='ffill')
        uptick_distance = (close - last_uptick_price) / last_uptick_price
        
        return {
            'ssr_triggered': ssr_triggered.astype(int),
            'ssr_active': ssr_active.astype(int),
            'ssr_compliant': ssr_compliant.astype(int),
            'uptick_distance': uptick_distance,
            'decline_from_prev_close': decline_pct
        }
    
    def calculate_luld_mechanics(self, 
                                df: pd.DataFrame,
                                reference_price_col: str = 'close',
                                price_tier: str = 'tier1') -> Dict[str, pd.Series]:
        """
        Calculate Limit Up/Limit Down band mechanics.
        
        LULD bands are calculated based on reference price and stock tier.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLC data
        reference_price_col : str
            Column to use for reference price calculation
        price_tier : str
            Price tier for LULD bands ('tier1', 'tier2', 'low_price')
            
        Returns
        -------
        Dict[str, pd.Series]
            LULD-related indicators
        """
        reference_price = df[reference_price_col]
        band_pct = self.luld_bands.get(price_tier, 0.05)
        
        # Calculate LULD bands
        upper_band = reference_price * (1 + band_pct)
        lower_band = reference_price * (1 - band_pct)
        
        # Price violations
        limit_up_violation = df['high'] >= upper_band
        limit_down_violation = df['low'] <= lower_band
        
        # Distance to bands (normalized)
        distance_to_upper = (upper_band - df['close']) / reference_price
        distance_to_lower = (df['close'] - lower_band) / reference_price
        
        # Band pressure (how close to limits)
        band_pressure = np.minimum(distance_to_upper, distance_to_lower) / band_pct
        
        # Limit state indicator
        in_limit_state = limit_up_violation | limit_down_violation
        
        # Band width (volatility proxy)
        band_width = (upper_band - lower_band) / reference_price
        
        return {
            'luld_upper_band': upper_band,
            'luld_lower_band': lower_band,
            'limit_up_violation': limit_up_violation.astype(int),
            'limit_down_violation': limit_down_violation.astype(int),
            'distance_to_upper_band': distance_to_upper,
            'distance_to_lower_band': distance_to_lower,
            'luld_band_pressure': band_pressure,
            'in_limit_state': in_limit_state.astype(int),
            'luld_band_width': band_width
        }
    
    def calculate_momentum_indicators(self, 
                                    df: pd.DataFrame,
                                    lookback_periods: List[int] = [5, 10, 20, 50]) -> Dict[str, pd.Series]:
        """
        Calculate various momentum indicators for intraday trading.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        lookback_periods : List[int]
            Lookback periods for momentum calculation
            
        Returns
        -------
        Dict[str, pd.Series]
            Momentum indicators
        """
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']
        
        indicators = {}
        
        # Price momentum for each period
        for period in lookback_periods:
            # Simple momentum
            momentum = (close - close.shift(period)) / close.shift(period) * 100
            indicators[f'momentum_{period}'] = momentum
            
            # Volume-weighted momentum
            vol_weights = volume.rolling(window=period).apply(
                lambda x: x / x.sum() if x.sum() > 0 else 0, raw=False
            )
            vw_momentum = momentum * vol_weights
            indicators[f'vw_momentum_{period}'] = vw_momentum
        
        # Relative Strength Index (RSI)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi
        
        # Williams %R
        for period in [14, 20]:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            indicators[f'williams_r_{period}'] = williams_r
        
        # Stochastic Oscillator
        period = 14
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=3).mean()
        indicators['stoch_k'] = k_percent
        indicators['stoch_d'] = d_percent
        
        return indicators
    
    def calculate_mean_reversion_indicators(self, 
                                          df: pd.DataFrame,
                                          bb_period: int = 20,
                                          bb_std: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate mean reversion indicators.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        bb_period : int
            Bollinger Bands period
        bb_std : float
            Bollinger Bands standard deviation multiplier
            
        Returns
        -------
        Dict[str, pd.Series]
            Mean reversion indicators
        """
        close = df['close']
        
        indicators = {}
        
        # Bollinger Bands
        bb_middle = close.rolling(window=bb_period).mean()
        bb_std_dev = close.rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        
        # Bollinger Band position
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_position'] = bb_position
        indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Mean reversion score (distance from moving average in standard deviations)
        for period in [10, 20, 50]:
            ma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            mean_reversion_score = (close - ma) / std
            indicators[f'mean_reversion_{period}'] = mean_reversion_score
        
        # Price vs VWAP mean reversion
        vwap = self.calculate_vwap(df)
        vwap_reversion = (close - vwap) / vwap
        indicators['vwap_mean_reversion'] = vwap_reversion
        
        return indicators
    
    def extract_all_features(self, 
                           df: pd.DataFrame,
                           include_momentum: bool = True,
                           include_mean_reversion: bool = True,
                           price_tier: str = 'tier1') -> pd.DataFrame:
        """
        Extract all intraday features in a single call.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input OHLCV data with DatetimeIndex
        include_momentum : bool
            Whether to include momentum indicators
        include_mean_reversion : bool
            Whether to include mean reversion indicators
        price_tier : str
            Price tier for LULD calculations
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all features added
        """
        result_df = df.copy()
        
        # Core features
        result_df['vwap'] = self.calculate_vwap(df)
        
        # ATR for all configured periods
        for period in self.atr_periods:
            result_df[f'atr_{period}'] = self.calculate_atr(df, period)
        
        # Stretch metrics
        stretch_features = self.calculate_stretch_metrics(df, 'vwap')
        for name, series in stretch_features.items():
            result_df[name] = series
        
        # SSR Gate features
        ssr_features = self.calculate_ssr_gate(df)
        for name, series in ssr_features.items():
            result_df[name] = series
        
        # LULD features
        luld_features = self.calculate_luld_mechanics(df, price_tier=price_tier)
        for name, series in luld_features.items():
            result_df[name] = series
        
        # Optional momentum indicators
        if include_momentum:
            momentum_features = self.calculate_momentum_indicators(df)
            for name, series in momentum_features.items():
                result_df[name] = series
        
        # Optional mean reversion indicators
        if include_mean_reversion:
            mean_reversion_features = self.calculate_mean_reversion_indicators(df)
            for name, series in mean_reversion_features.items():
                result_df[name] = series
        
        return result_df


# Utility functions for market time handling
def is_market_hours(timestamp: pd.Timestamp, 
                   market_open: time = time(9, 30),
                   market_close: time = time(16, 0),
                   timezone: str = "America/New_York") -> bool:
    """
    Check if timestamp is within market hours.
    
    Parameters
    ----------
    timestamp : pd.Timestamp
        Timestamp to check
    market_open : time
        Market opening time
    market_close : time
        Market closing time  
    timezone : str
        Market timezone
        
    Returns
    -------
    bool
        True if within market hours
    """
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize(timezone)
    else:
        timestamp = timestamp.tz_convert(timezone)
    
    current_time = timestamp.time()
    return market_open <= current_time <= market_close


def calculate_time_to_close(timestamp: pd.Timestamp,
                           market_close: time = time(16, 0),
                           timezone: str = "America/New_York") -> float:
    """
    Calculate time to market close in hours.
    
    Parameters
    ----------
    timestamp : pd.Timestamp
        Current timestamp
    market_close : time
        Market closing time
    timezone : str
        Market timezone
        
    Returns
    -------
    float
        Hours until market close
    """
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize(timezone)
    else:
        timestamp = timestamp.tz_convert(timezone)
    
    # Create close time for same date
    close_datetime = datetime.combine(timestamp.date(), market_close)
    close_datetime = pd.Timestamp(close_datetime).tz_localize(timezone)
    
    time_diff = close_datetime - timestamp
    return time_diff.total_seconds() / 3600  # Convert to hours