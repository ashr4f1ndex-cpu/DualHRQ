"""
Enhanced HRM input preparation with integrated feature engineering.

This module extends the basic HRM input functionality to incorporate the
comprehensive feature engineering pipeline, including:
- Options features (IV term structure, Greeks, volatility regimes)
- Intraday features (VWAP, ATR, SSR gates, LULD mechanics)
- Corporate action adjustments
- Time alignment and market calendar handling
- Leakage prevention with CPCV

The enhanced pipeline ensures production-ready features with proper
temporal ordering and no look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler

from ..common.feature_integration import (
    IntegratedFeatureEngine, 
    FeatureConfig,
    prepare_hrm_features,
    create_hrm_cv_splits
)
from .hrm_input import TokenConfig, FittedScalers


@dataclass
class EnhancedTokenConfig(TokenConfig):
    """Enhanced token configuration with feature engineering parameters."""
    
    # Feature engineering config
    feature_config: FeatureConfig = None
    
    # Scaling options
    robust_scaling: bool = False
    feature_selection: bool = True
    max_features: int = 100
    
    # Token construction
    daily_window: int = 192
    minutes_per_day: int = 390
    include_technical_features: bool = True
    include_options_features: bool = True
    
    def __post_init__(self):
        """Initialize default feature config."""
        if self.feature_config is None:
            self.feature_config = FeatureConfig()


@dataclass 
class EnhancedFittedScalers:
    """Enhanced scaler container with feature engineering scalers."""
    
    daily: Union[StandardScaler, RobustScaler]
    intraday: Union[StandardScaler, RobustScaler]
    
    # Additional scalers for engineered features
    options: Optional[Union[StandardScaler, RobustScaler]] = None
    technical: Optional[Union[StandardScaler, RobustScaler]] = None
    
    # Feature selection info
    selected_daily_features: Optional[List[str]] = None
    selected_intraday_features: Optional[List[str]] = None


class EnhancedFeatureProcessor:
    """
    Enhanced feature processor for HRM model inputs.
    
    Integrates comprehensive feature engineering with the HRM input pipeline
    to provide production-ready features with leakage prevention.
    """
    
    def __init__(self, config: EnhancedTokenConfig = None):
        """
        Initialize enhanced feature processor.
        
        Parameters
        ----------
        config : EnhancedTokenConfig, optional
            Configuration for feature processing
        """
        self.config = config or EnhancedTokenConfig()
        self.feature_engine = IntegratedFeatureEngine(self.config.feature_config)
        self._scalers = None
        self._feature_importance = None
    
    def prepare_features_from_raw_data(self,
                                     raw_data: Dict[str, pd.DataFrame],
                                     target_dates: pd.DatetimeIndex = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features from raw data sources.
        
        Parameters
        ----------
        raw_data : Dict[str, pd.DataFrame]
            Raw data dictionary with keys: 'equity', 'options', 'iv_surface', 'corporate_actions'
        target_dates : pd.DatetimeIndex, optional
            Specific dates to process
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Daily features, Intraday features
        """
        # Use integrated feature engine to prepare data
        prepared_data = self.feature_engine.prepare_raw_data(**raw_data)
        
        # Determine target dates
        if target_dates is None:
            equity_data = prepared_data['equity']
            target_dates = pd.date_range(
                start=equity_data.index.min().normalize(),
                end=equity_data.index.max().normalize(),
                freq='D'
            )
        
        # Create feature pipeline
        daily_features, intraday_features = self.feature_engine.create_feature_pipeline(
            prepared_data, target_dates
        )
        
        return daily_features, intraday_features
    
    def fit_enhanced_scalers(self,
                           daily_features: pd.DataFrame,
                           intraday_features: pd.DataFrame,
                           train_idx: np.ndarray) -> EnhancedFittedScalers:
        """
        Fit enhanced scalers with feature selection.
        
        Parameters
        ----------
        daily_features : pd.DataFrame
            Daily features indexed by date
        intraday_features : pd.DataFrame
            Intraday features with date column
        train_idx : np.ndarray
            Training indices for daily features
            
        Returns
        -------
        EnhancedFittedScalers
            Fitted scalers with feature selection
        """
        scaler_class = RobustScaler if self.config.robust_scaling else StandardScaler
        
        # Prepare training data
        train_dates = daily_features.index[train_idx]
        daily_train = daily_features.loc[train_dates]
        
        # Filter intraday features to training dates
        intraday_train = intraday_features[
            intraday_features['date'].isin(train_dates)
        ] if 'date' in intraday_features.columns else pd.DataFrame()
        
        # Feature selection for daily features
        selected_daily_features = self._select_features(
            daily_train, 
            max_features=self.config.max_features // 2
        )
        
        # Feature selection for intraday features
        selected_intraday_features = []
        if not intraday_train.empty:
            # Remove date column for feature selection
            intraday_numeric = intraday_train.select_dtypes(include=[np.number])
            selected_intraday_features = self._select_features(
                intraday_numeric,
                max_features=self.config.max_features // 2
            )
        
        # Fit scalers on selected features
        daily_scaler = scaler_class()
        intraday_scaler = scaler_class()
        
        if selected_daily_features:
            daily_scaler.fit(daily_train[selected_daily_features].values)
        
        if selected_intraday_features and not intraday_train.empty:
            intraday_numeric = intraday_train[selected_intraday_features]
            intraday_scaler.fit(intraday_numeric.values)
        
        # Fit specialized scalers if needed
        options_scaler = None
        technical_scaler = None
        
        if self.config.include_options_features:
            options_cols = [col for col in daily_train.columns 
                          if any(keyword in col.lower() for keyword in 
                               ['iv', 'vol', 'delta', 'gamma', 'theta', 'vega', 'rho'])]
            if options_cols:
                options_scaler = scaler_class()
                options_scaler.fit(daily_train[options_cols].values)
        
        if self.config.include_technical_features:
            technical_cols = [col for col in daily_train.columns
                            if any(keyword in col.lower() for keyword in
                                 ['momentum', 'rsi', 'williams', 'stoch', 'bb_'])]
            if technical_cols:
                technical_scaler = scaler_class()
                technical_scaler.fit(daily_train[technical_cols].values)
        
        self._scalers = EnhancedFittedScalers(
            daily=daily_scaler,
            intraday=intraday_scaler,
            options=options_scaler,
            technical=technical_scaler,
            selected_daily_features=selected_daily_features,
            selected_intraday_features=selected_intraday_features
        )
        
        return self._scalers
    
    def _select_features(self, 
                        features: pd.DataFrame, 
                        max_features: int) -> List[str]:
        """
        Select most important features using variance and correlation filtering.
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        max_features : int
            Maximum number of features to select
            
        Returns
        -------
        List[str]
            Selected feature names
        """
        if not self.config.feature_selection or features.empty:
            return list(features.columns)
        
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Step 1: Remove constant features
        feature_variance = numeric_features.var()
        non_constant = feature_variance[feature_variance > 1e-6].index.tolist()
        
        if len(non_constant) == 0:
            return []
        
        # Step 2: Remove highly correlated features
        if len(non_constant) > 1:
            correlation_matrix = numeric_features[non_constant].corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation > 0.95
            high_corr_features = [column for column in upper_triangle.columns 
                                if any(upper_triangle[column] > 0.95)]
            
            selected_features = [col for col in non_constant if col not in high_corr_features]
        else:
            selected_features = non_constant
        
        # Step 3: Select top features by variance if needed
        if len(selected_features) > max_features:
            feature_variances = numeric_features[selected_features].var()
            top_features = feature_variances.nlargest(max_features).index.tolist()
            selected_features = top_features
        
        return selected_features
    
    def make_enhanced_h_tokens(self,
                             daily_features: pd.DataFrame,
                             dates: pd.DatetimeIndex,
                             scalers: EnhancedFittedScalers) -> torch.Tensor:
        """
        Create enhanced H tokens with feature engineering.
        
        Parameters
        ----------
        daily_features : pd.DataFrame
            Daily features indexed by date
        dates : pd.DatetimeIndex
            Target dates for token creation
        scalers : EnhancedFittedScalers
            Fitted scalers
            
        Returns
        -------
        torch.Tensor
            H tokens with shape (len(dates), daily_window, num_features)
        """
        D = self.config.daily_window
        
        # Use selected features
        if scalers.selected_daily_features:
            feature_data = daily_features[scalers.selected_daily_features]
        else:
            feature_data = daily_features.select_dtypes(include=[np.number])
        
        tokens = []
        for date in dates:
            # Get historical window ending at date
            window_data = feature_data.loc[:date].tail(D)
            
            # Handle insufficient history with padding
            if len(window_data) < D:
                if len(window_data) > 0:
                    # Pad with first available row
                    pad_rows = D - len(window_data)
                    pad_data = np.tile(window_data.iloc[0].values, (pad_rows, 1))
                    window_matrix = np.vstack([pad_data, window_data.values])
                else:
                    # No data available, use zeros
                    window_matrix = np.zeros((D, len(feature_data.columns)))
            else:
                window_matrix = window_data.values
            
            tokens.append(window_matrix)
        
        # Stack and scale
        token_array = np.array(tokens)  # Shape: (B, T, F)
        B, T, F = token_array.shape
        
        # Reshape for scaling
        token_array_reshaped = token_array.reshape(B * T, F)
        
        # Apply scaling
        token_array_scaled = scalers.daily.transform(token_array_reshaped)
        
        # Reshape back
        token_array_final = token_array_scaled.reshape(B, T, F)
        
        return torch.from_numpy(token_array_final).float()
    
    def make_enhanced_l_tokens(self,
                             intraday_features: pd.DataFrame,
                             dates: pd.DatetimeIndex,
                             scalers: EnhancedFittedScalers) -> torch.Tensor:
        """
        Create enhanced L tokens with intraday feature engineering.
        
        Parameters
        ----------
        intraday_features : pd.DataFrame
            Intraday features with date column
        dates : pd.DatetimeIndex
            Target dates for token creation
        scalers : EnhancedFittedScalers
            Fitted scalers
            
        Returns
        -------
        torch.Tensor
            L tokens with shape (len(dates), minutes_per_day, num_features)
        """
        M = self.config.minutes_per_day
        
        if intraday_features.empty or not scalers.selected_intraday_features:
            # Return zero tokens if no intraday data
            return torch.zeros((len(dates), M, 1), dtype=torch.float32)
        
        # Use selected features
        feature_cols = scalers.selected_intraday_features
        
        tokens = []
        for date in dates:
            # Get intraday data for the date
            if 'date' in intraday_features.columns:
                day_mask = intraday_features['date'].dt.normalize() == date.normalize()
                day_data = intraday_features[day_mask][feature_cols]
            else:
                # Assume index is datetime
                day_mask = intraday_features.index.normalize() == date.normalize()
                day_data = intraday_features[day_mask][feature_cols]
            
            if len(day_data) == 0:
                # No data for this date, use zeros
                day_matrix = np.zeros((M, len(feature_cols)))
            elif len(day_data) < M:
                # Pad with last observation
                pad_rows = M - len(day_data)
                if len(day_data) > 0:
                    pad_data = np.tile(day_data.iloc[-1].values, (pad_rows, 1))
                    day_matrix = np.vstack([day_data.values, pad_data])
                else:
                    day_matrix = np.zeros((M, len(feature_cols)))
            else:
                # Take first M observations
                day_matrix = day_data.iloc[:M].values
            
            tokens.append(day_matrix)
        
        # Stack and scale
        token_array = np.array(tokens)  # Shape: (B, M, F)
        B, M_actual, F = token_array.shape
        
        # Reshape for scaling
        token_array_reshaped = token_array.reshape(B * M_actual, F)
        
        # Apply scaling
        token_array_scaled = scalers.intraday.transform(token_array_reshaped)
        
        # Reshape back
        token_array_final = token_array_scaled.reshape(B, M_actual, F)
        
        return torch.from_numpy(token_array_final).float()
    
    def create_enhanced_dataset(self,
                              raw_data: Dict[str, pd.DataFrame],
                              targets: Dict[str, pd.Series],
                              train_idx: np.ndarray,
                              val_idx: np.ndarray = None) -> Dict[str, torch.Tensor]:
        """
        Create enhanced dataset for HRM training with comprehensive features.
        
        Parameters
        ----------
        raw_data : Dict[str, pd.DataFrame]
            Raw data sources
        targets : Dict[str, pd.Series]
            Target variables (yA for regression, yB for classification)
        train_idx : np.ndarray
            Training indices
        val_idx : np.ndarray, optional
            Validation indices
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Enhanced dataset with H, L tokens and targets
        """
        # Prepare features
        daily_features, intraday_features = self.prepare_features_from_raw_data(raw_data)
        
        # Validate feature quality
        validation_results = self.feature_engine.validate_feature_quality(
            daily_features, intraday_features
        )
        
        if validation_results['overall_quality'] == 'POOR':
            print("Warning: Poor feature quality detected")
            print("Consider reviewing data preprocessing steps")
        
        # Fit scalers on training data
        scalers = self.fit_enhanced_scalers(daily_features, intraday_features, train_idx)
        
        # Get all dates
        all_dates = daily_features.index
        
        # Create tokens
        H_tokens = self.make_enhanced_h_tokens(daily_features, all_dates, scalers)
        L_tokens = self.make_enhanced_l_tokens(intraday_features, all_dates, scalers)
        
        # Prepare targets
        yA = torch.tensor(targets['yA'].values, dtype=torch.float32)
        yB = torch.tensor(targets['yB'].values, dtype=torch.float32)
        
        dataset = {
            'H_tokens': H_tokens,
            'L_tokens': L_tokens,
            'yA': yA,
            'yB': yB,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'scalers': scalers,
            'daily_features': daily_features,
            'intraday_features': intraday_features,
            'validation_results': validation_results
        }
        
        return dataset
    
    def create_cross_validation_datasets(self,
                                       raw_data: Dict[str, pd.DataFrame],
                                       targets: Dict[str, pd.Series]) -> List[Dict[str, torch.Tensor]]:
        """
        Create cross-validation datasets with leakage prevention.
        
        Parameters
        ----------
        raw_data : Dict[str, pd.DataFrame]
            Raw data sources
        targets : Dict[str, pd.Series]
            Target variables
            
        Returns
        -------
        List[Dict[str, torch.Tensor]]
            List of CV datasets
        """
        # Prepare features
        daily_features, intraday_features = self.prepare_features_from_raw_data(raw_data)
        
        # Create CV splits with leakage prevention
        cv_splits = create_hrm_cv_splits(
            daily_features, 
            targets.get('yA'),
            self.config.feature_config
        )
        
        cv_datasets = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"Creating CV dataset for fold {fold_idx + 1}/{len(cv_splits)}")
            
            dataset = self.create_enhanced_dataset(
                raw_data, targets, train_idx, val_idx
            )
            dataset['fold_idx'] = fold_idx
            cv_datasets.append(dataset)
        
        return cv_datasets


# Convenience functions for backward compatibility
def create_enhanced_hrm_inputs(raw_data: Dict[str, pd.DataFrame],
                             targets: Dict[str, pd.Series],
                             config: EnhancedTokenConfig = None) -> Dict[str, torch.Tensor]:
    """
    Convenience function to create enhanced HRM inputs.
    
    Parameters
    ----------
    raw_data : Dict[str, pd.DataFrame]
        Raw data dictionary
    targets : Dict[str, pd.Series]
        Target variables
    config : EnhancedTokenConfig, optional
        Configuration
        
    Returns
    -------
    Dict[str, torch.Tensor]
        Enhanced HRM dataset
    """
    processor = EnhancedFeatureProcessor(config)
    
    # Simple train/val split (80/20)
    n_samples = len(targets['yA'])
    train_size = int(0.8 * n_samples)
    train_idx = np.arange(train_size)
    val_idx = np.arange(train_size, n_samples)
    
    return processor.create_enhanced_dataset(raw_data, targets, train_idx, val_idx)