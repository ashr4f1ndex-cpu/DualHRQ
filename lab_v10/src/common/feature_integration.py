"""
Comprehensive feature integration module for dual-book trading strategies.

This module integrates all feature engineering components into a unified pipeline:
- Options features (IV term structure, Greeks, volatility regimes)
- Intraday features (VWAP, ATR, SSR gates, LULD mechanics)
- Corporate action adjustments (CRSP methodology)
- Time alignment (market calendars, timezone handling)
- Leakage prevention (CPCV with purging/embargo)

The integrated pipeline ensures proper temporal ordering, prevents look-ahead bias,
and provides production-ready feature vectors for the HRM training pipeline.

References:
- López de Prado, M. Advances in Financial Machine Learning
- Hull, J. Options, Futures, and Other Derivatives
- Harris, L. Trading and Exchanges
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta

# Import all feature engineering modules
from .options_features import OptionsFeatureEngine
from .intraday_features import IntradayFeatureEngine
from .corporate_actions import CorporateActionAdjuster, CorporateActionDatabase
from .data_alignment import DataAligner, USEquityCalendar, OptionsCalendar
from .leakage_prevention import CombinatorialPurgedCV, WalkForwardCV, LeakageAuditor

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class FeatureConfig:
    """Configuration for integrated feature pipeline."""
    
    # Options features
    enable_options_features: bool = True
    iv_lookback_windows: List[int] = None
    volatility_regime_windows: List[int] = None
    
    # Intraday features
    enable_intraday_features: bool = True
    atr_periods: List[int] = None
    vwap_reset_freq: str = 'D'
    luld_price_tier: str = 'tier1'
    
    # Corporate actions
    enable_corporate_actions: bool = True
    adjustment_method: str = "total_return"
    include_special_dividends: bool = True
    
    # Time alignment
    primary_market: str = 'equity'
    reference_timezone: str = "America/New_York"
    target_frequency: str = '1min'
    fill_market_gaps: bool = True
    
    # Leakage prevention
    enable_leakage_prevention: bool = True
    cv_method: str = 'cpcv'  # 'cpcv', 'walkforward'
    purge_hours: int = 1
    embargo_hours: int = 2
    n_splits: int = 6
    n_test_groups: int = 2
    
    def __post_init__(self):
        """Set default values."""
        if self.iv_lookback_windows is None:
            self.iv_lookback_windows = [1, 5, 21, 63]
        if self.volatility_regime_windows is None:
            self.volatility_regime_windows = [10, 20, 50]
        if self.atr_periods is None:
            self.atr_periods = [14, 20, 50]


class IntegratedFeatureEngine:
    """
    Comprehensive feature engineering engine for dual-book trading strategies.
    
    This class orchestrates all feature engineering components to produce
    a unified, leak-free feature pipeline suitable for HRM model training.
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize the integrated feature engine.
        
        Parameters
        ----------
        config : FeatureConfig, optional
            Configuration for feature pipeline
        """
        self.config = config or FeatureConfig()
        
        # Initialize component engines
        if self.config.enable_options_features:
            self.options_engine = OptionsFeatureEngine(
                vol_lookback_windows=self.config.iv_lookback_windows
            )
        
        if self.config.enable_intraday_features:
            self.intraday_engine = IntradayFeatureEngine(
                atr_periods=self.config.atr_periods,
                vwap_reset_freq=self.config.vwap_reset_freq
            )
        
        if self.config.enable_corporate_actions:
            self.corp_action_adjuster = CorporateActionAdjuster(
                adjustment_method=self.config.adjustment_method,
                include_special_dividends=self.config.include_special_dividends
            )
            self.corp_action_db = CorporateActionDatabase()
        
        # Data alignment engine
        self.data_aligner = DataAligner(
            reference_timezone=self.config.reference_timezone
        )
        
        if self.config.enable_leakage_prevention:
            self.leakage_auditor = LeakageAuditor()
    
    def prepare_raw_data(self,
                        equity_data: pd.DataFrame,
                        options_data: pd.DataFrame = None,
                        iv_surface_data: Dict = None,
                        corporate_actions_data: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        Prepare and align raw data inputs.
        
        Parameters
        ----------
        equity_data : pd.DataFrame
            OHLCV equity data with DatetimeIndex
        options_data : pd.DataFrame, optional
            Options chain data
        iv_surface_data : Dict, optional
            Implied volatility surface data
        corporate_actions_data : pd.DataFrame, optional
            Corporate actions data
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Prepared data dictionary
        """
        prepared_data = {}
        
        # 1. Load corporate actions if provided
        if (self.config.enable_corporate_actions and 
            corporate_actions_data is not None):
            self.corp_action_db.load_from_dataframe(corporate_actions_data)
            
            # Apply corporate action adjustments to equity data
            symbol = equity_data.get('symbol', 'UNKNOWN')
            if isinstance(symbol, str):
                actions = self.corp_action_db.get_actions_by_symbol(symbol)
                if actions:
                    equity_data = self.corp_action_adjuster.apply_adjustments(
                        equity_data, actions
                    )
        
        # 2. Time zone and market hours alignment
        equity_data = self.data_aligner.convert_timezone(
            equity_data, 
            self.config.reference_timezone
        )
        
        # Align to trading days and market hours
        equity_data = self.data_aligner.align_to_trading_days(
            equity_data, 
            self.config.primary_market
        )
        
        equity_data = self.data_aligner.align_to_market_hours(
            equity_data,
            self.config.primary_market
        )
        
        # 3. Resample to target frequency
        if self.config.target_frequency:
            equity_data = self.data_aligner.resample_to_frequency(
                equity_data,
                self.config.target_frequency
            )
        
        # 4. Fill market gaps if enabled
        if self.config.fill_market_gaps:
            equity_data = self.data_aligner.fill_market_gaps(
                equity_data,
                self.config.primary_market,
                self.config.target_frequency
            )
        
        prepared_data['equity'] = equity_data
        
        # Process options data if provided
        if options_data is not None:
            options_data = self.data_aligner.convert_timezone(
                options_data,
                self.config.reference_timezone
            )
            prepared_data['options'] = options_data
        
        if iv_surface_data is not None:
            prepared_data['iv_surface'] = iv_surface_data
        
        return prepared_data
    
    def extract_daily_features(self,
                              prepared_data: Dict[str, pd.DataFrame],
                              current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Extract daily-frequency features for a specific date.
        
        Parameters
        ----------
        prepared_data : Dict[str, pd.DataFrame]
            Prepared data dictionary
        current_date : pd.Timestamp
            Current date for feature extraction
            
        Returns
        -------
        Dict[str, float]
            Daily features dictionary
        """
        daily_features = {}
        equity_data = prepared_data['equity']
        
        # Filter data up to current date to prevent look-ahead bias
        historical_equity = equity_data[equity_data.index <= current_date]
        
        if len(historical_equity) == 0:
            return daily_features
        
        # Extract options features if enabled
        if (self.config.enable_options_features and 
            'iv_surface' in prepared_data and
            prepared_data['iv_surface']):
            
            # Get current spot price
            spot_price = historical_equity['close'].iloc[-1]
            
            # Get IV surface for current date
            iv_surface = prepared_data['iv_surface'].get(current_date, {})
            
            if iv_surface:
                options_features = self.options_engine.extract_all_features(
                    iv_surface=iv_surface,
                    spot_price=spot_price,
                    price_history=historical_equity['close'],
                    current_time=current_date
                )
                daily_features.update(options_features)
        
        # Extract volatility regime features
        if len(historical_equity) >= 63:  # Need sufficient history
            price_series = historical_equity['close']
            
            # Rolling volatility features
            for window in self.config.volatility_regime_windows:
                if len(price_series) >= window:
                    returns = price_series.pct_change().dropna()
                    if len(returns) >= window:
                        vol = returns.tail(window).std() * np.sqrt(252)
                        daily_features[f'realized_vol_{window}d'] = float(vol)
        
        # Market microstructure features
        if len(historical_equity) >= 20:
            # Price momentum
            for lookback in [5, 10, 20]:
                if len(historical_equity) >= lookback:
                    momentum = (historical_equity['close'].iloc[-1] / 
                              historical_equity['close'].iloc[-lookback] - 1) * 100
                    daily_features[f'momentum_{lookback}d'] = float(momentum)
            
            # Volume features
            if 'volume' in historical_equity.columns:
                vol_ma_20 = historical_equity['volume'].tail(20).mean()
                current_vol = historical_equity['volume'].iloc[-1]
                daily_features['volume_ratio_20d'] = float(current_vol / vol_ma_20) if vol_ma_20 > 0 else 1.0
        
        return daily_features
    
    def extract_intraday_features(self,
                                 prepared_data: Dict[str, pd.DataFrame],
                                 current_date: pd.Timestamp) -> pd.DataFrame:
        """
        Extract intraday features for a specific date.
        
        Parameters
        ----------
        prepared_data : Dict[str, pd.DataFrame]
            Prepared data dictionary
        current_date : pd.Timestamp
            Current date for feature extraction
            
        Returns
        -------
        pd.DataFrame
            Intraday features for the date
        """
        equity_data = prepared_data['equity']
        
        # Get intraday data for current date
        date_mask = equity_data.index.normalize() == current_date.normalize()
        day_data = equity_data[date_mask].copy()
        
        if len(day_data) == 0:
            return pd.DataFrame()
        
        # Extract intraday features if enabled
        if self.config.enable_intraday_features:
            intraday_features = self.intraday_engine.extract_all_features(
                day_data,
                price_tier=self.config.luld_price_tier
            )
            return intraday_features
        
        return day_data
    
    def create_feature_pipeline(self,
                               prepared_data: Dict[str, pd.DataFrame],
                               target_dates: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create complete feature pipeline for training/testing.
        
        Parameters
        ----------
        prepared_data : Dict[str, pd.DataFrame]
            Prepared data dictionary
        target_dates : pd.DatetimeIndex
            Dates to create features for
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Daily features DataFrame, Intraday features DataFrame
        """
        daily_features_list = []
        intraday_features_list = []
        
        for date in target_dates:
            # Extract daily features
            daily_feat = self.extract_daily_features(prepared_data, date)
            if daily_feat:
                daily_feat['date'] = date
                daily_features_list.append(daily_feat)
            
            # Extract intraday features
            intraday_feat = self.extract_intraday_features(prepared_data, date)
            if not intraday_feat.empty:
                intraday_feat['date'] = date
                intraday_features_list.append(intraday_feat)
        
        # Combine features
        daily_df = pd.DataFrame(daily_features_list)
        if not daily_df.empty:
            daily_df.set_index('date', inplace=True)
        
        intraday_df = pd.concat(intraday_features_list, ignore_index=True) if intraday_features_list else pd.DataFrame()
        
        return daily_df, intraday_df
    
    def create_cross_validation_splits(self,
                                      feature_data: pd.DataFrame,
                                      target_data: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits with leakage prevention.
        
        Parameters
        ----------
        feature_data : pd.DataFrame
            Feature data with DatetimeIndex
        target_data : pd.Series, optional
            Target variable
            
        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples
        """
        if not self.config.enable_leakage_prevention:
            # Simple chronological split
            n_samples = len(feature_data)
            split_point = int(0.8 * n_samples)
            train_indices = np.arange(split_point)
            test_indices = np.arange(split_point, n_samples)
            return [(train_indices, test_indices)]
        
        # Prepare data for CV
        cv_data = feature_data.copy()
        if 'timestamp' not in cv_data.columns and hasattr(cv_data.index, 'to_series'):
            cv_data['timestamp'] = cv_data.index.to_series()
        
        purge = pd.Timedelta(hours=self.config.purge_hours)
        embargo = pd.Timedelta(hours=self.config.embargo_hours)
        
        if self.config.cv_method == 'cpcv':
            cv = CombinatorialPurgedCV(
                n_splits=self.config.n_splits,
                n_test_groups=self.config.n_test_groups,
                purge=purge,
                embargo=embargo
            )
        else:  # walkforward
            initial_train_size = pd.Timedelta(days=60)
            test_size = pd.Timedelta(days=10)
            cv = WalkForwardCV(
                initial_train_size=initial_train_size,
                test_size=test_size,
                purge=purge,
                embargo=embargo
            )
        
        splits = list(cv.split(cv_data, target_data))
        
        # Audit for leakage
        for i, (train_idx, test_idx) in enumerate(splits):
            train_data = cv_data.iloc[train_idx]
            test_data = cv_data.iloc[test_idx]
            
            if target_data is not None:
                train_features = feature_data.iloc[train_idx]
                train_target = target_data.iloc[train_idx]
                
                audit_results = self.leakage_auditor.comprehensive_audit(
                    train_data, test_data, train_features, train_target,
                    purge, embargo
                )
                
                if not audit_results['summary']['all_checks_passed']:
                    print(f"Warning: Leakage detected in fold {i}")
                    print("Recommendations:", audit_results['summary']['recommendations'])
        
        return splits
    
    def validate_feature_quality(self,
                                daily_features: pd.DataFrame,
                                intraday_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature quality and detect potential issues.
        
        Parameters
        ----------
        daily_features : pd.DataFrame
            Daily features
        intraday_features : pd.DataFrame
            Intraday features
            
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        validation_results = {}
        
        # Daily features validation
        if not daily_features.empty:
            daily_validation = {
                'n_features': len(daily_features.columns),
                'n_samples': len(daily_features),
                'missing_ratio': daily_features.isnull().sum().sum() / daily_features.size,
                'inf_values': np.isinf(daily_features.select_dtypes(include=[np.number])).sum().sum(),
                'constant_features': (daily_features.var() == 0).sum(),
                'high_correlation_pairs': self._find_high_correlation_pairs(daily_features),
            }
            validation_results['daily'] = daily_validation
        
        # Intraday features validation
        if not intraday_features.empty:
            intraday_validation = {
                'n_features': len(intraday_features.columns),
                'n_samples': len(intraday_features),
                'missing_ratio': intraday_features.isnull().sum().sum() / intraday_features.size,
                'inf_values': np.isinf(intraday_features.select_dtypes(include=[np.number])).sum().sum(),
                'constant_features': (intraday_features.var() == 0).sum(),
            }
            validation_results['intraday'] = intraday_validation
        
        # Overall assessment
        validation_results['overall_quality'] = self._assess_overall_quality(validation_results)
        
        return validation_results
    
    def _find_high_correlation_pairs(self, 
                                   features: pd.DataFrame, 
                                   threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """Find pairs of features with high correlation."""
        numeric_features = features.select_dtypes(include=[np.number])
        correlation_matrix = numeric_features.corr().abs()
        
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > threshold:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_value
                    ))
        
        return high_corr_pairs
    
    def _assess_overall_quality(self, validation_results: Dict) -> str:
        """Assess overall feature quality."""
        issues = []
        
        for data_type in ['daily', 'intraday']:
            if data_type in validation_results:
                results = validation_results[data_type]
                
                if results['missing_ratio'] > 0.1:
                    issues.append(f"High missing ratio in {data_type} features")
                
                if results['inf_values'] > 0:
                    issues.append(f"Infinite values in {data_type} features")
                
                if results['constant_features'] > 0:
                    issues.append(f"Constant features in {data_type} data")
                
                if data_type == 'daily' and len(results.get('high_correlation_pairs', [])) > 5:
                    issues.append("Many highly correlated feature pairs")
        
        if not issues:
            return "GOOD"
        elif len(issues) <= 2:
            return "ACCEPTABLE"
        else:
            return "POOR"


# Convenience functions for integration with HRM pipeline
def prepare_hrm_features(raw_data: Dict[str, pd.DataFrame],
                        config: FeatureConfig = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to prepare features for HRM model training.
    
    Parameters
    ----------
    raw_data : Dict[str, pd.DataFrame]
        Dictionary containing raw data (equity, options, etc.)
    config : FeatureConfig, optional
        Feature configuration
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Daily features, Intraday features
    """
    engine = IntegratedFeatureEngine(config)
    
    # Prepare data
    prepared_data = engine.prepare_raw_data(**raw_data)
    
    # Get date range
    equity_data = prepared_data['equity']
    date_range = pd.date_range(
        start=equity_data.index.min().normalize(),
        end=equity_data.index.max().normalize(),
        freq='D'
    )
    
    # Create features
    daily_features, intraday_features = engine.create_feature_pipeline(
        prepared_data, date_range
    )
    
    return daily_features, intraday_features


def create_hrm_cv_splits(daily_features: pd.DataFrame,
                        targets: pd.Series = None,
                        config: FeatureConfig = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits for HRM model training.
    
    Parameters
    ----------
    daily_features : pd.DataFrame
        Daily features with DatetimeIndex
    targets : pd.Series, optional
        Target variable
    config : FeatureConfig, optional
        Feature configuration
        
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        Cross-validation splits
    """
    engine = IntegratedFeatureEngine(config)
    return engine.create_cross_validation_splits(daily_features, targets)