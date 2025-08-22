"""
Comprehensive options feature engineering module for dual-book trading strategies.

This module implements the complete options feature pipeline including:
- Implied volatility term structure analysis
- Greeks calculation (delta, gamma, theta, vega, rho)
- Volatility regime indicators and vol-of-vol metrics
- Smile metrics (skew, kurtosis, term structure shape)
- Forward/spot basis and carry calculations

All features are designed to prevent look-ahead bias and integrate seamlessly
with the leakage prevention framework.

References:
- Hull, J. Options, Futures, and Other Derivatives
- Gatheral, J. The Volatility Surface
- Rebonato, R. Volatility and Correlation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d, UnivariateSpline
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Suppress warnings for cleaner output in production
warnings.filterwarnings('ignore', category=RuntimeWarning)


class OptionsFeatureEngine:
    """
    Production-grade options feature engineering engine.
    
    Implements comprehensive feature extraction from options market data
    with strict adherence to temporal ordering and leakage prevention.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.05,
                 dividend_yield: float = 0.02,
                 min_ttm_days: int = 1,
                 max_ttm_days: int = 365,
                 vol_lookback_windows: List[int] = [1, 5, 21, 63]):
        """
        Initialize the options feature engine.
        
        Parameters
        ----------
        risk_free_rate : float
            Risk-free interest rate for Greeks calculation
        dividend_yield : float  
            Dividend yield for forward calculations
        min_ttm_days : int
            Minimum time to maturity in days
        max_ttm_days : int
            Maximum time to maturity in days
        vol_lookback_windows : List[int]
            Lookback windows for realized volatility calculation
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.min_ttm_days = min_ttm_days
        self.max_ttm_days = max_ttm_days
        self.vol_lookback_windows = vol_lookback_windows
        
    def black_scholes_price(self, 
                           S: float, 
                           K: float, 
                           T: float, 
                           r: float, 
                           q: float, 
                           sigma: float, 
                           option_type: str = 'call') -> float:
        """
        Black-Scholes option pricing formula.
        
        Parameters
        ----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration in years
        r : float
            Risk-free rate
        q : float
            Dividend yield
        sigma : float
            Implied volatility
        option_type : str
            'call' or 'put'
            
        Returns
        -------
        float
            Option price
        """
        if T <= 0 or sigma <= 0:
            return max(0, (S - K) if option_type == 'call' else (K - S))
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    def calculate_greeks(self, 
                        S: float, 
                        K: float, 
                        T: float, 
                        r: float, 
                        q: float, 
                        sigma: float, 
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks using Black-Scholes model.
        
        Parameters
        ----------
        S, K, T, r, q, sigma : float
            Black-Scholes parameters
        option_type : str
            'call' or 'put'
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing delta, gamma, theta, vega, rho
        """
        if T <= 0 or sigma <= 0:
            return {
                'delta': 1.0 if option_type == 'call' and S > K else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate Greeks
        if option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        theta_call = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2) 
                     + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        
        theta_put = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                    - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
        
        theta = theta_call if option_type == 'call' else theta_put
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta),
            'vega': float(vega),
            'rho': float(rho)
        }
    
    def iv_term_structure_features(self, 
                                  iv_surface: Dict[Tuple[float, float], float],
                                  spot_price: float) -> Dict[str, float]:
        """
        Extract implied volatility term structure features.
        
        Parameters
        ----------
        iv_surface : Dict[Tuple[float, float], float]
            IV surface with keys (strike, ttm_years) and values IV
        spot_price : float
            Current spot price
            
        Returns
        -------
        Dict[str, float]
            Term structure features
        """
        features = {}
        
        # Group by time to maturity
        ttm_groups = {}
        for (strike, ttm), iv in iv_surface.items():
            if ttm not in ttm_groups:
                ttm_groups[ttm] = []
            ttm_groups[ttm].append((strike, iv))
        
        # Calculate ATM IV for each expiry
        atm_ivs = {}
        for ttm, strikes_ivs in ttm_groups.items():
            if len(strikes_ivs) < 2:
                continue
                
            strikes, ivs = zip(*strikes_ivs)
            strikes = np.array(strikes)
            ivs = np.array(ivs)
            
            # Find ATM IV using interpolation
            if spot_price >= strikes.min() and spot_price <= strikes.max():
                atm_iv = np.interp(spot_price, strikes, ivs)
                atm_ivs[ttm] = atm_iv
        
        if len(atm_ivs) >= 2:
            ttms = np.array(sorted(atm_ivs.keys()))
            ivs = np.array([atm_ivs[ttm] for ttm in ttms])
            
            # Term structure slope (front to back)
            features['iv_ts_slope'] = float((ivs[-1] - ivs[0]) / (ttms[-1] - ttms[0] + 1e-8))
            
            # Term structure curvature
            if len(ttms) >= 3:
                try:
                    poly_coeffs = np.polyfit(ttms, ivs, 2)
                    features['iv_ts_curvature'] = float(poly_coeffs[0])
                except:
                    features['iv_ts_curvature'] = 0.0
            else:
                features['iv_ts_curvature'] = 0.0
                
            # Vol-of-vol (volatility of term structure)
            if len(ivs) > 1:
                features['iv_ts_vol_of_vol'] = float(np.std(ivs))
            else:
                features['iv_ts_vol_of_vol'] = 0.0
        else:
            features.update({
                'iv_ts_slope': 0.0,
                'iv_ts_curvature': 0.0,
                'iv_ts_vol_of_vol': 0.0
            })
        
        return features
    
    def volatility_smile_features(self, 
                                 iv_surface: Dict[Tuple[float, float], float],
                                 spot_price: float,
                                 target_ttm: float = None) -> Dict[str, float]:
        """
        Extract volatility smile characteristics.
        
        Parameters
        ----------
        iv_surface : Dict[Tuple[float, float], float]
            IV surface with keys (strike, ttm_years) and values IV
        spot_price : float
            Current spot price
        target_ttm : float, optional
            Target time to maturity for smile analysis
            
        Returns
        -------
        Dict[str, float]
            Smile features including skew, kurtosis, and convexity
        """
        features = {}
        
        # If no target TTM specified, use the nearest expiry with sufficient strikes
        if target_ttm is None:
            ttm_strike_counts = {}
            for (strike, ttm), iv in iv_surface.items():
                if ttm not in ttm_strike_counts:
                    ttm_strike_counts[ttm] = 0
                ttm_strike_counts[ttm] += 1
            
            # Choose TTM with most strikes
            if ttm_strike_counts:
                target_ttm = max(ttm_strike_counts.keys(), key=ttm_strike_counts.get)
            else:
                return {
                    'smile_skew': 0.0,
                    'smile_kurtosis': 0.0,
                    'smile_convexity': 0.0,
                    'put_call_skew': 0.0
                }
        
        # Extract strikes and IVs for target expiry
        strikes_ivs = [(strike, iv) for (strike, ttm), iv in iv_surface.items() 
                       if abs(ttm - target_ttm) < 1e-6]
        
        if len(strikes_ivs) < 3:
            return {
                'smile_skew': 0.0,
                'smile_kurtosis': 0.0,
                'smile_convexity': 0.0,
                'put_call_skew': 0.0
            }
        
        strikes, ivs = zip(*sorted(strikes_ivs))
        strikes = np.array(strikes)
        ivs = np.array(ivs)
        
        # Convert to log-moneyness
        log_moneyness = np.log(strikes / spot_price)
        
        # Fit polynomial to smile
        try:
            if len(strikes) >= 3:
                poly_coeffs = np.polyfit(log_moneyness, ivs, 2)
                features['smile_convexity'] = float(poly_coeffs[0])
                features['smile_skew'] = float(poly_coeffs[1])
            else:
                features['smile_convexity'] = 0.0
                features['smile_skew'] = 0.0
        except:
            features['smile_convexity'] = 0.0
            features['smile_skew'] = 0.0
        
        # Calculate put-call skew (OTM put IV - OTM call IV)
        otm_puts = strikes[strikes < spot_price * 0.95]
        otm_calls = strikes[strikes > spot_price * 1.05]
        
        if len(otm_puts) > 0 and len(otm_calls) > 0:
            put_idx = np.where(strikes < spot_price * 0.95)[0]
            call_idx = np.where(strikes > spot_price * 1.05)[0]
            
            if len(put_idx) > 0 and len(call_idx) > 0:
                put_iv = ivs[put_idx[-1]]  # Closest OTM put
                call_iv = ivs[call_idx[0]]  # Closest OTM call
                features['put_call_skew'] = float(put_iv - call_iv)
            else:
                features['put_call_skew'] = 0.0
        else:
            features['put_call_skew'] = 0.0
        
        # Smile kurtosis (fourth moment)
        if len(ivs) > 4:
            try:
                poly_coeffs = np.polyfit(log_moneyness, ivs, 4)
                features['smile_kurtosis'] = float(poly_coeffs[0])
            except:
                features['smile_kurtosis'] = 0.0
        else:
            features['smile_kurtosis'] = 0.0
        
        return features
    
    def realized_volatility_features(self, 
                                   price_series: pd.Series,
                                   current_time: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate realized volatility metrics across multiple windows.
        
        Parameters
        ----------
        price_series : pd.Series
            Historical price series with timestamp index
        current_time : pd.Timestamp
            Current timestamp (to prevent look-ahead bias)
            
        Returns
        -------
        Dict[str, float]
            Realized volatility features
        """
        features = {}
        
        # Filter data up to current time to prevent look-ahead bias
        historical_prices = price_series[price_series.index <= current_time]
        
        if len(historical_prices) < 2:
            for window in self.vol_lookback_windows:
                features[f'realized_vol_{window}d'] = 0.0
            features.update({
                'vol_regime_indicator': 0.0,
                'vol_persistence': 0.0,
                'vol_mean_reversion': 0.0
            })
            return features
        
        # Calculate log returns
        log_returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
        
        # Calculate realized volatility for each window
        for window in self.vol_lookback_windows:
            if len(log_returns) >= window:
                recent_returns = log_returns.tail(window)
                annualized_vol = np.sqrt(252) * recent_returns.std()
                features[f'realized_vol_{window}d'] = float(annualized_vol)
            else:
                features[f'realized_vol_{window}d'] = 0.0
        
        # Vol regime indicator (current vs long-term vol)
        if len(log_returns) >= max(21, max(self.vol_lookback_windows)):
            short_vol = features.get('realized_vol_21d', 0.0)
            long_vol = features.get(f'realized_vol_{max(self.vol_lookback_windows)}d', 0.0)
            
            if long_vol > 0:
                features['vol_regime_indicator'] = float(short_vol / long_vol)
            else:
                features['vol_regime_indicator'] = 1.0
        else:
            features['vol_regime_indicator'] = 1.0
        
        # Vol persistence (autocorrelation of vol)
        if len(log_returns) >= 42:  # Need enough data for rolling vol
            rolling_vol = log_returns.rolling(21).std().dropna()
            if len(rolling_vol) >= 2:
                vol_returns = rolling_vol.pct_change().dropna()
                if len(vol_returns) >= 10:
                    features['vol_persistence'] = float(vol_returns.autocorr(lag=1))
                else:
                    features['vol_persistence'] = 0.0
            else:
                features['vol_persistence'] = 0.0
        else:
            features['vol_persistence'] = 0.0
        
        # Vol mean reversion (based on current vol vs historical median)
        if len(log_returns) >= 63:
            current_vol = features.get('realized_vol_21d', 0.0)
            historical_vol = log_returns.rolling(21).std().dropna()
            if len(historical_vol) >= 20:
                vol_median = historical_vol.median()
                if vol_median > 0:
                    features['vol_mean_reversion'] = float((current_vol - vol_median) / vol_median)
                else:
                    features['vol_mean_reversion'] = 0.0
            else:
                features['vol_mean_reversion'] = 0.0
        else:
            features['vol_mean_reversion'] = 0.0
        
        return features
    
    def forward_basis_features(self, 
                             spot_price: float,
                             forward_prices: Dict[float, float],
                             dividend_schedule: List[Tuple[pd.Timestamp, float]] = None) -> Dict[str, float]:
        """
        Calculate forward/spot basis and carry features.
        
        Parameters
        ----------
        spot_price : float
            Current spot price
        forward_prices : Dict[float, float]
            Forward prices by time to maturity (years)
        dividend_schedule : List[Tuple[pd.Timestamp, float]], optional
            Dividend payment schedule
            
        Returns
        -------
        Dict[str, float]
            Forward basis and carry features
        """
        features = {}
        
        if not forward_prices:
            features.update({
                'forward_basis_1m': 0.0,
                'forward_basis_3m': 0.0,
                'forward_basis_6m': 0.0,
                'carry_slope': 0.0,
                'dividend_yield_estimate': self.dividend_yield
            })
            return features
        
        # Calculate forward basis for standard tenors
        target_tenors = {'1m': 1/12, '3m': 0.25, '6m': 0.5}
        
        for tenor_name, tenor_years in target_tenors.items():
            # Find closest forward
            if forward_prices:
                closest_ttm = min(forward_prices.keys(), key=lambda x: abs(x - tenor_years))
                forward_price = forward_prices[closest_ttm]
                
                # Calculate basis (forward premium/discount)
                basis = (forward_price - spot_price) / spot_price
                features[f'forward_basis_{tenor_name}'] = float(basis)
            else:
                features[f'forward_basis_{tenor_name}'] = 0.0
        
        # Calculate carry slope (term structure of forward basis)
        if len(forward_prices) >= 2:
            ttms = np.array(sorted(forward_prices.keys()))
            bases = np.array([(forward_prices[ttm] - spot_price) / spot_price for ttm in ttms])
            
            # Linear fit to get carry slope
            if len(ttms) >= 2:
                slope, _ = np.polyfit(ttms, bases, 1)
                features['carry_slope'] = float(slope)
            else:
                features['carry_slope'] = 0.0
        else:
            features['carry_slope'] = 0.0
        
        # Estimate dividend yield from forward curve
        if len(forward_prices) >= 2 and dividend_schedule:
            # Use forward curve to back out dividend yield
            # F = S * exp((r - q) * T)
            # q = r - ln(F/S) / T
            
            ttms = np.array(sorted(forward_prices.keys()))
            implied_yields = []
            
            for ttm in ttms:
                if ttm > 0:
                    forward_price = forward_prices[ttm]
                    implied_q = self.risk_free_rate - np.log(forward_price / spot_price) / ttm
                    implied_yields.append(implied_q)
            
            if implied_yields:
                features['dividend_yield_estimate'] = float(np.median(implied_yields))
            else:
                features['dividend_yield_estimate'] = self.dividend_yield
        else:
            features['dividend_yield_estimate'] = self.dividend_yield
        
        return features
    
    def atm_straddle_features(self, 
                            iv_surface: Dict[Tuple[float, float], float],
                            spot_price: float,
                            target_ttm: float = None) -> Dict[str, float]:
        """
        Calculate ATM straddle Greeks and metrics.
        
        Parameters
        ----------
        iv_surface : Dict[Tuple[float, float], float]
            IV surface with keys (strike, ttm_years) and values IV
        spot_price : float
            Current spot price
        target_ttm : float, optional
            Target time to maturity
            
        Returns
        -------
        Dict[str, float]
            ATM straddle features
        """
        features = {}
        
        # Find ATM strike and appropriate expiry
        if target_ttm is None:
            # Use front month
            available_ttms = set(ttm for (strike, ttm), iv in iv_surface.items())
            if available_ttms:
                target_ttm = min(available_ttms)
            else:
                return {
                    'atm_straddle_delta': 0.0,
                    'atm_straddle_gamma': 0.0,
                    'atm_straddle_theta': 0.0,
                    'atm_straddle_vega': 0.0,
                    'atm_iv': 0.0
                }
        
        # Find closest ATM strike for target expiry
        atm_candidates = [(strike, iv) for (strike, ttm), iv in iv_surface.items() 
                         if abs(ttm - target_ttm) < 1e-6]
        
        if not atm_candidates:
            return {
                'atm_straddle_delta': 0.0,
                'atm_straddle_gamma': 0.0,
                'atm_straddle_theta': 0.0,
                'atm_straddle_vega': 0.0,
                'atm_iv': 0.0
            }
        
        # Find strike closest to spot
        atm_strike, atm_iv = min(atm_candidates, key=lambda x: abs(x[0] - spot_price))
        
        # Calculate Greeks for ATM call and put
        call_greeks = self.calculate_greeks(
            spot_price, atm_strike, target_ttm, 
            self.risk_free_rate, self.dividend_yield, atm_iv, 'call'
        )
        
        put_greeks = self.calculate_greeks(
            spot_price, atm_strike, target_ttm,
            self.risk_free_rate, self.dividend_yield, atm_iv, 'put'
        )
        
        # Straddle Greeks (sum of call and put)
        features['atm_straddle_delta'] = float(call_greeks['delta'] + put_greeks['delta'])
        features['atm_straddle_gamma'] = float(call_greeks['gamma'] + put_greeks['gamma'])
        features['atm_straddle_theta'] = float(call_greeks['theta'] + put_greeks['theta'])
        features['atm_straddle_vega'] = float(call_greeks['vega'] + put_greeks['vega'])
        features['atm_iv'] = float(atm_iv)
        
        return features
    
    def extract_all_features(self, 
                           iv_surface: Dict[Tuple[float, float], float],
                           spot_price: float,
                           price_history: pd.Series,
                           current_time: pd.Timestamp,
                           forward_prices: Dict[float, float] = None,
                           dividend_schedule: List[Tuple[pd.Timestamp, float]] = None) -> Dict[str, float]:
        """
        Extract all options features in a single call.
        
        Parameters
        ----------
        iv_surface : Dict[Tuple[float, float], float]
            Complete IV surface
        spot_price : float
            Current spot price
        price_history : pd.Series
            Historical price series
        current_time : pd.Timestamp
            Current timestamp
        forward_prices : Dict[float, float], optional
            Forward prices by TTM
        dividend_schedule : List[Tuple[pd.Timestamp, float]], optional
            Dividend schedule
            
        Returns
        -------
        Dict[str, float]
            Complete feature dictionary
        """
        all_features = {}
        
        # IV term structure features
        all_features.update(self.iv_term_structure_features(iv_surface, spot_price))
        
        # Volatility smile features
        all_features.update(self.volatility_smile_features(iv_surface, spot_price))
        
        # Realized volatility features
        all_features.update(self.realized_volatility_features(price_history, current_time))
        
        # Forward basis features
        if forward_prices is None:
            forward_prices = {}
        all_features.update(self.forward_basis_features(spot_price, forward_prices, dividend_schedule))
        
        # ATM straddle features
        all_features.update(self.atm_straddle_features(iv_surface, spot_price))
        
        return all_features


def calculate_iv_from_price(price: float, 
                           S: float, 
                           K: float, 
                           T: float, 
                           r: float, 
                           q: float, 
                           option_type: str = 'call',
                           max_iterations: int = 100,
                           tolerance: float = 1e-6) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters
    ----------
    price : float
        Observed option price
    S, K, T, r, q : float
        Black-Scholes parameters
    option_type : str
        'call' or 'put'
    max_iterations : int
        Maximum iterations for convergence
    tolerance : float
        Convergence tolerance
        
    Returns
    -------
    float
        Implied volatility
    """
    engine = OptionsFeatureEngine(r, q)
    
    # Initial guess
    sigma = 0.3
    
    for _ in range(max_iterations):
        # Calculate theoretical price and vega
        theoretical_price = engine.black_scholes_price(S, K, T, r, q, sigma, option_type)
        greeks = engine.calculate_greeks(S, K, T, r, q, sigma, option_type)
        vega = greeks['vega'] * 100  # Convert to percentage
        
        # Newton-Raphson update
        if abs(vega) < 1e-10:
            break
            
        price_diff = theoretical_price - price
        
        if abs(price_diff) < tolerance:
            break
            
        sigma = sigma - price_diff / vega
        
        # Ensure sigma stays positive
        sigma = max(0.01, min(5.0, sigma))
    
    return float(sigma)