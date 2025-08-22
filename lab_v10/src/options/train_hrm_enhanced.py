"""
Enhanced HRM training script with comprehensive feature engineering.

This script demonstrates the complete dual-book feature engineering pipeline
integrated with HRM model training, including:
- Options features (IV term structure, Greeks, volatility regimes)
- Intraday features (VWAP, ATR, SSR gates, LULD mechanics)
- Corporate action adjustments with CRSP methodology
- Time alignment with market calendars and timezone handling
- Leakage prevention using CPCV with purging/embargo
- Production-ready training pipeline

Usage:
    python train_hrm_enhanced.py --config config/enhanced_training.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.common.feature_integration import FeatureConfig, IntegratedFeatureEngine
from src.options.hrm_input_enhanced import (
    EnhancedFeatureProcessor, 
    EnhancedTokenConfig,
    create_enhanced_hrm_inputs
)
from src.options.hrm_net import HRMNet, HRMConfig
from src.options.hrm_train import HRMTrainer, TrainConfig, MultiTaskDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Load sample data for demonstration.
    
    In production, this would load from your data sources:
    - Market data feeds
    - Options chains
    - Corporate actions database
    - IV surface data
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Sample data dictionary
    """
    logger.info("Loading sample data...")
    
    # Generate sample equity data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1min')
    n_samples = len(dates)
    
    # Market hours filtering (9:30-16:00 ET, weekdays only)
    market_hours = dates[
        (dates.hour >= 9) & (dates.hour < 16) & 
        (dates.weekday < 5) &
        ~((dates.hour == 9) & (dates.minute < 30))
    ]
    
    # Sample OHLCV data with realistic patterns
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.02, len(market_hours))
    prices = base_price * np.cumprod(1 + returns)
    
    equity_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(market_hours))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(market_hours)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(market_hours)))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, len(market_hours))
    }, index=market_hours)
    
    # Generate sample IV surface data
    iv_surface_data = {}
    daily_dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    for date in daily_dates:
        if date.weekday() < 5:  # Only trading days
            spot_price = base_price + np.random.normal(0, 10)
            strikes = np.arange(spot_price * 0.8, spot_price * 1.2, 5)
            ttms = [30/365, 60/365, 90/365]  # 30, 60, 90 day options
            
            surface = {}
            for strike in strikes:
                for ttm in ttms:
                    moneyness = strike / spot_price
                    # Simple IV smile model
                    base_iv = 0.20 + 0.05 * abs(moneyness - 1.0) + 0.02 * np.sqrt(ttm)
                    iv = base_iv + np.random.normal(0, 0.01)
                    surface[(strike, ttm)] = max(0.05, iv)
            
            iv_surface_data[date] = surface
    
    # Generate sample corporate actions
    corporate_actions = pd.DataFrame({
        'symbol': ['SAMPLE'] * 4,
        'action_type': ['CASH_DIVIDEND', 'CASH_DIVIDEND', 'STOCK_SPLIT', 'CASH_DIVIDEND'],
        'ex_date': ['2023-03-15', '2023-06-15', '2023-08-15', '2023-12-15'],
        'cash_amount': [0.50, 0.55, None, 0.60],
        'split_ratio': [None, None, 2.0, None]
    })
    
    return {
        'equity': equity_data,
        'iv_surface': iv_surface_data,
        'corporate_actions': corporate_actions
    }


def create_sample_targets(equity_data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Create sample target variables for HRM training.
    
    Parameters
    ----------
    equity_data : pd.DataFrame
        Equity data
        
    Returns
    -------
    Dict[str, pd.Series]
        Target variables (yA: regression, yB: classification)
    """
    logger.info("Creating sample targets...")
    
    # Group by date for daily targets
    daily_data = equity_data.groupby(equity_data.index.date).agg({
        'close': 'last',
        'volume': 'sum',
        'high': 'max',
        'low': 'min'
    })
    
    # Target A: Next-day volatility (regression)
    returns = daily_data['close'].pct_change()
    realized_vol = returns.rolling(5).std() * np.sqrt(252)
    yA = realized_vol.shift(-1).dropna()  # Next period volatility
    
    # Target B: High volume day indicator (classification)
    volume_ma = daily_data['volume'].rolling(20).mean()
    volume_ratio = daily_data['volume'] / volume_ma
    yB = (volume_ratio > 1.5).astype(float)  # High volume indicator
    
    # Align targets
    common_dates = yA.index.intersection(yB.index)
    yA = yA.loc[common_dates]
    yB = yB.loc[common_dates]
    
    return {
        'yA': yA,
        'yB': yB
    }


def run_enhanced_training(config_path: Optional[str] = None):
    """
    Run enhanced HRM training with comprehensive feature engineering.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file
    """
    logger.info("Starting enhanced HRM training...")
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        logger.info("Using default configuration")
        config_dict = {}
    
    # Feature engineering configuration
    feature_config = FeatureConfig(
        enable_options_features=config_dict.get('enable_options_features', True),
        enable_intraday_features=config_dict.get('enable_intraday_features', True),
        enable_corporate_actions=config_dict.get('enable_corporate_actions', True),
        enable_leakage_prevention=config_dict.get('enable_leakage_prevention', True),
        cv_method=config_dict.get('cv_method', 'cpcv'),
        n_splits=config_dict.get('n_splits', 6),
        n_test_groups=config_dict.get('n_test_groups', 2)
    )
    
    # Token configuration
    token_config = EnhancedTokenConfig(
        feature_config=feature_config,
        daily_window=config_dict.get('daily_window', 192),
        minutes_per_day=config_dict.get('minutes_per_day', 390),
        robust_scaling=config_dict.get('robust_scaling', False),
        feature_selection=config_dict.get('feature_selection', True),
        max_features=config_dict.get('max_features', 100)
    )
    
    # HRM model configuration
    hrm_config = HRMConfig(
        d_embed=config_dict.get('d_embed', 128),
        n_layers=config_dict.get('n_layers', 6),
        n_heads=config_dict.get('n_heads', 8),
        use_heteroscedastic=config_dict.get('use_heteroscedastic', True),
        deq_style=config_dict.get('deq_style', True),
        act_enable=config_dict.get('act_enable', True)
    )
    
    # Training configuration
    train_config = TrainConfig(
        lr=config_dict.get('learning_rate', 1e-4),
        batch_size=config_dict.get('batch_size', 32),
        max_epochs=config_dict.get('max_epochs', 50),
        early_stop_patience=config_dict.get('early_stop_patience', 10),
        use_gradnorm=config_dict.get('use_gradnorm', True),
        uncertainty_weighting=config_dict.get('uncertainty_weighting', True)
    )
    
    # Load sample data
    raw_data = load_sample_data()
    targets = create_sample_targets(raw_data['equity'])
    
    logger.info(f"Loaded data: {len(raw_data['equity'])} equity samples, "
                f"{len(targets['yA'])} target samples")
    
    # Initialize enhanced feature processor
    processor = EnhancedFeatureProcessor(token_config)
    
    # Option 1: Single train/validation split
    if not feature_config.enable_leakage_prevention:
        logger.info("Creating single train/validation dataset...")
        
        dataset = create_enhanced_hrm_inputs(raw_data, targets, token_config)
        
        # Create data loaders
        train_dataset = MultiTaskDataset(
            dataset['H_tokens'][dataset['train_idx']],
            dataset['L_tokens'][dataset['train_idx']],
            dataset['yA'][dataset['train_idx']],
            dataset['yB'][dataset['train_idx']]
        )
        
        val_dataset = MultiTaskDataset(
            dataset['H_tokens'][dataset['val_idx']],
            dataset['L_tokens'][dataset['val_idx']],
            dataset['yA'][dataset['val_idx']],
            dataset['yB'][dataset['val_idx']]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)
        
        # Initialize and train model
        net = HRMNet(hrm_config)
        trainer = HRMTrainer(net, train_config)
        
        logger.info("Starting training...")
        results = trainer.fit(train_loader, val_loader)
        logger.info(f"Training completed. Best validation loss: {results['val_loss']:.4f}")
        
        # Load best model
        trainer.load_best()
        
    else:
        # Option 2: Cross-validation with leakage prevention
        logger.info("Creating cross-validation datasets with leakage prevention...")
        
        cv_datasets = processor.create_cross_validation_datasets(raw_data, targets)
        
        cv_results = []
        for fold_idx, dataset in enumerate(cv_datasets):
            logger.info(f"Training fold {fold_idx + 1}/{len(cv_datasets)}...")
            
            # Create data loaders for this fold
            train_dataset = MultiTaskDataset(
                dataset['H_tokens'][dataset['train_idx']],
                dataset['L_tokens'][dataset['train_idx']],
                dataset['yA'][dataset['train_idx']],
                dataset['yB'][dataset['train_idx']]
            )
            
            val_dataset = MultiTaskDataset(
                dataset['H_tokens'][dataset['val_idx']],
                dataset['L_tokens'][dataset['val_idx']],
                dataset['yA'][dataset['val_idx']],
                dataset['yB'][dataset['val_idx']]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)
            
            # Initialize and train model for this fold
            net = HRMNet(hrm_config)
            trainer = HRMTrainer(net, train_config)
            
            results = trainer.fit(train_loader, val_loader)
            cv_results.append(results['val_loss'])
            
            logger.info(f"Fold {fold_idx + 1} completed. Validation loss: {results['val_loss']:.4f}")
            
            # Save fold model
            torch.save(net.state_dict(), f".hrm_fold_{fold_idx}.pt")
        
        # Report CV results
        mean_cv_loss = np.mean(cv_results)
        std_cv_loss = np.std(cv_results)
        logger.info(f"Cross-validation completed.")
        logger.info(f"CV Results: {mean_cv_loss:.4f} ± {std_cv_loss:.4f}")
        logger.info(f"Individual fold losses: {cv_results}")
    
    logger.info("Enhanced HRM training completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced HRM Training")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.gpu and torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        logger.info("Using CPU")
    
    try:
        run_enhanced_training(args.config)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()