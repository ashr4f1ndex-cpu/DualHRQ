"""
Corporate actions handler with CRSP-style adjustment methodology.

This module implements comprehensive corporate action adjustments for financial
time series data, following CRSP (Center for Research in Security Prices) 
methodology and industry best practices.

Supported corporate actions:
- Stock splits and stock dividends
- Cash dividends (regular and special)
- Spin-offs and distributions
- Rights offerings
- Mergers and acquisitions
- Symbol changes and delistings

The adjustment factors ensure price and volume continuity while preserving
returns and maintaining the integrity of technical analysis.

References:
- CRSP Data Guide
- Wharton Research Data Services (WRDS) Documentation
- Fama, E. F. & French, K. R. Common risk factors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime, date

warnings.filterwarnings('ignore', category=RuntimeWarning)


class CorporateActionType(Enum):
    """Enumeration of corporate action types."""
    CASH_DIVIDEND = "CASH_DIVIDEND"
    STOCK_DIVIDEND = "STOCK_DIVIDEND"
    STOCK_SPLIT = "STOCK_SPLIT"
    SPIN_OFF = "SPIN_OFF"
    RIGHTS_OFFERING = "RIGHTS_OFFERING"
    MERGER = "MERGER"
    ACQUISITION = "ACQUISITION"
    SYMBOL_CHANGE = "SYMBOL_CHANGE"
    DELISTING = "DELISTING"
    SPECIAL_DIVIDEND = "SPECIAL_DIVIDEND"
    RETURN_OF_CAPITAL = "RETURN_OF_CAPITAL"


@dataclass
class CorporateAction:
    """
    Container for corporate action information.
    """
    symbol: str
    action_type: CorporateActionType
    ex_date: pd.Timestamp
    record_date: Optional[pd.Timestamp] = None
    payment_date: Optional[pd.Timestamp] = None
    
    # Action-specific parameters
    cash_amount: Optional[float] = None  # For dividends
    split_ratio: Optional[float] = None  # For splits (new/old)
    distribution_ratio: Optional[float] = None  # For spin-offs
    exchange_ratio: Optional[float] = None  # For mergers
    new_symbol: Optional[str] = None  # For symbol changes
    
    # Additional metadata
    currency: str = "USD"
    description: Optional[str] = None
    source: Optional[str] = None
    
    def __post_init__(self):
        """Validate corporate action data."""
        if self.action_type == CorporateActionType.CASH_DIVIDEND and self.cash_amount is None:
            raise ValueError("Cash dividend requires cash_amount")
        
        if self.action_type == CorporateActionType.STOCK_SPLIT and self.split_ratio is None:
            raise ValueError("Stock split requires split_ratio")
        
        if self.action_type == CorporateActionType.SPIN_OFF and self.distribution_ratio is None:
            raise ValueError("Spin-off requires distribution_ratio")


class CorporateActionAdjuster:
    """
    CRSP-style corporate action adjustment engine.
    
    This class implements comprehensive adjustment methodologies that ensure
    data continuity while preserving economic relationships in financial
    time series data.
    """
    
    def __init__(self, 
                 adjustment_method: str = "total_return",
                 include_special_dividends: bool = True,
                 min_dividend_threshold: float = 0.01,
                 split_threshold: float = 0.25):
        """
        Initialize corporate action adjuster.
        
        Parameters
        ----------
        adjustment_method : str
            Adjustment methodology ('total_return', 'price_only', 'split_only')
        include_special_dividends : bool
            Whether to adjust for special dividends
        min_dividend_threshold : float
            Minimum dividend amount to trigger adjustment
        split_threshold : float
            Minimum split ratio change to trigger adjustment
        """
        self.adjustment_method = adjustment_method
        self.include_special_dividends = include_special_dividends
        self.min_dividend_threshold = min_dividend_threshold
        self.split_threshold = split_threshold
        
        # Valid adjustment methods
        valid_methods = ["total_return", "price_only", "split_only"]
        if adjustment_method not in valid_methods:
            raise ValueError(f"adjustment_method must be one of {valid_methods}")
    
    def calculate_adjustment_factors(self, 
                                   corporate_actions: List[CorporateAction],
                                   price_series: pd.Series) -> pd.DataFrame:
        """
        Calculate cumulative adjustment factors for all corporate actions.
        
        Parameters
        ----------
        corporate_actions : List[CorporateAction]
            List of corporate actions sorted by ex_date
        price_series : pd.Series
            Historical price series with DatetimeIndex
            
        Returns
        -------
        pd.DataFrame
            DataFrame with adjustment factors by date and type
        """
        # Sort corporate actions by ex_date
        sorted_actions = sorted(corporate_actions, key=lambda x: x.ex_date)
        
        # Initialize adjustment factors DataFrame
        date_range = pd.date_range(
            start=price_series.index.min(),
            end=price_series.index.max(),
            freq='D'
        )
        
        adj_factors = pd.DataFrame(index=date_range)
        adj_factors['price_factor'] = 1.0
        adj_factors['volume_factor'] = 1.0
        adj_factors['dividend_factor'] = 1.0
        adj_factors['split_factor'] = 1.0
        
        # Process each corporate action
        for action in sorted_actions:
            if action.ex_date not in adj_factors.index:
                continue
                
            ex_date_idx = adj_factors.index.get_loc(action.ex_date)
            
            # Calculate individual adjustment factors
            price_adj, vol_adj, div_adj, split_adj = self._calculate_single_adjustment(
                action, price_series
            )
            
            # Apply adjustments to all dates before ex_date
            if ex_date_idx > 0:
                adj_factors.iloc[:ex_date_idx, adj_factors.columns.get_loc('price_factor')] *= price_adj
                adj_factors.iloc[:ex_date_idx, adj_factors.columns.get_loc('volume_factor')] *= vol_adj
                adj_factors.iloc[:ex_date_idx, adj_factors.columns.get_loc('dividend_factor')] *= div_adj
                adj_factors.iloc[:ex_date_idx, adj_factors.columns.get_loc('split_factor')] *= split_adj
        
        # Calculate total adjustment factor
        if self.adjustment_method == "total_return":
            adj_factors['total_factor'] = (adj_factors['price_factor'] * 
                                         adj_factors['dividend_factor'] * 
                                         adj_factors['split_factor'])
        elif self.adjustment_method == "price_only":
            adj_factors['total_factor'] = adj_factors['split_factor']
        else:  # split_only
            adj_factors['total_factor'] = adj_factors['split_factor']
        
        return adj_factors
    
    def _calculate_single_adjustment(self, 
                                   action: CorporateAction,
                                   price_series: pd.Series) -> Tuple[float, float, float, float]:
        """
        Calculate adjustment factors for a single corporate action.
        
        Parameters
        ----------
        action : CorporateAction
            Corporate action to process
        price_series : pd.Series
            Price series for context
            
        Returns
        -------
        Tuple[float, float, float, float]
            price_factor, volume_factor, dividend_factor, split_factor
        """
        price_factor = 1.0
        volume_factor = 1.0
        dividend_factor = 1.0
        split_factor = 1.0
        
        # Get price on day before ex_date for reference
        try:
            pre_ex_dates = price_series.index[price_series.index < action.ex_date]
            if len(pre_ex_dates) > 0:
                reference_price = price_series.loc[pre_ex_dates[-1]]
            else:
                reference_price = price_series.iloc[0]
        except:
            reference_price = 100.0  # Default fallback
        
        if action.action_type == CorporateActionType.CASH_DIVIDEND:
            if action.cash_amount and action.cash_amount >= self.min_dividend_threshold:
                # Dividend adjustment: P_adj = P_raw * (P_close - Dividend) / P_close
                dividend_factor = (reference_price - action.cash_amount) / reference_price
                price_factor = dividend_factor
                
        elif action.action_type == CorporateActionType.SPECIAL_DIVIDEND:
            if (self.include_special_dividends and 
                action.cash_amount and 
                action.cash_amount >= self.min_dividend_threshold):
                dividend_factor = (reference_price - action.cash_amount) / reference_price
                price_factor = dividend_factor
                
        elif action.action_type == CorporateActionType.STOCK_SPLIT:
            if action.split_ratio and abs(action.split_ratio - 1.0) >= self.split_threshold:
                # Split adjustment: P_adj = P_raw / split_ratio
                split_factor = 1.0 / action.split_ratio
                price_factor = split_factor
                volume_factor = action.split_ratio  # Volume multiplied by split ratio
                
        elif action.action_type == CorporateActionType.STOCK_DIVIDEND:
            if action.split_ratio:
                # Stock dividend treated as split
                split_factor = 1.0 / (1.0 + action.split_ratio)
                price_factor = split_factor
                volume_factor = 1.0 + action.split_ratio
                
        elif action.action_type == CorporateActionType.SPIN_OFF:
            if action.distribution_ratio and action.cash_amount:
                # Spin-off: adjust for value distributed
                spin_off_value = action.cash_amount * action.distribution_ratio
                dividend_factor = (reference_price - spin_off_value) / reference_price
                price_factor = dividend_factor
                
        elif action.action_type == CorporateActionType.RIGHTS_OFFERING:
            if action.exchange_ratio and action.cash_amount:
                # Rights offering adjustment
                subscription_price = action.cash_amount
                rights_ratio = action.exchange_ratio
                
                # Theoretical ex-rights price
                ex_rights_price = (reference_price + rights_ratio * subscription_price) / (1 + rights_ratio)
                price_factor = ex_rights_price / reference_price
                
        elif action.action_type in [CorporateActionType.MERGER, CorporateActionType.ACQUISITION]:
            if action.exchange_ratio:
                # Merger/acquisition adjustment
                price_factor = action.exchange_ratio
                volume_factor = 1.0 / action.exchange_ratio
        
        return price_factor, volume_factor, dividend_factor, split_factor
    
    def apply_adjustments(self, 
                         data: pd.DataFrame,
                         corporate_actions: List[CorporateAction],
                         price_cols: List[str] = ['open', 'high', 'low', 'close'],
                         volume_col: str = 'volume',
                         inplace: bool = False) -> pd.DataFrame:
        """
        Apply corporate action adjustments to OHLCV data.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with DatetimeIndex
        corporate_actions : List[CorporateAction]
            List of corporate actions
        price_cols : List[str]
            Column names for price data
        volume_col : str
            Column name for volume data
        inplace : bool
            Whether to modify data in place
            
        Returns
        -------
        pd.DataFrame
            Adjusted data
        """
        if not inplace:
            data = data.copy()
        
        if len(corporate_actions) == 0:
            return data
        
        # Calculate adjustment factors
        adj_factors = self.calculate_adjustment_factors(
            corporate_actions, data[price_cols[0]] if price_cols else data.iloc[:, 0]
        )
        
        # Align adjustment factors with data
        aligned_factors = adj_factors.reindex(data.index, method='ffill').fillna(1.0)
        
        # Apply price adjustments
        for col in price_cols:
            if col in data.columns:
                data[col] = data[col] * aligned_factors['total_factor']
        
        # Apply volume adjustments
        if volume_col in data.columns:
            data[volume_col] = data[volume_col] * aligned_factors['volume_factor']
        
        # Add adjustment factor columns for reference
        data['adj_factor'] = aligned_factors['total_factor']
        data['volume_adj_factor'] = aligned_factors['volume_factor']
        
        return data
    
    def generate_adjusted_returns(self, 
                                price_series: pd.Series,
                                corporate_actions: List[CorporateAction],
                                return_type: str = 'simple') -> pd.Series:
        """
        Generate properly adjusted returns series.
        
        Parameters
        ----------
        price_series : pd.Series
            Raw price series
        corporate_actions : List[CorporateAction]
            List of corporate actions
        return_type : str
            'simple' or 'log' returns
            
        Returns
        -------
        pd.Series
            Adjusted returns series
        """
        # Apply adjustments to prices
        adj_factors = self.calculate_adjustment_factors(corporate_actions, price_series)
        aligned_factors = adj_factors.reindex(price_series.index, method='ffill').fillna(1.0)
        
        adjusted_prices = price_series * aligned_factors['total_factor']
        
        # Calculate returns
        if return_type == 'simple':
            returns = adjusted_prices.pct_change()
        elif return_type == 'log':
            returns = np.log(adjusted_prices / adjusted_prices.shift(1))
        else:
            raise ValueError("return_type must be 'simple' or 'log'")
        
        return returns.dropna()


class CorporateActionDatabase:
    """
    Database manager for corporate actions data.
    
    Provides methods to store, retrieve, and manage corporate action
    information with validation and consistency checks.
    """
    
    def __init__(self):
        """Initialize corporate action database."""
        self.actions = []
        self._symbol_index = {}
        self._date_index = {}
    
    def add_action(self, action: CorporateAction) -> None:
        """
        Add a corporate action to the database.
        
        Parameters
        ----------
        action : CorporateAction
            Corporate action to add
        """
        # Validate action
        self._validate_action(action)
        
        # Add to main list
        self.actions.append(action)
        
        # Update indices
        if action.symbol not in self._symbol_index:
            self._symbol_index[action.symbol] = []
        self._symbol_index[action.symbol].append(action)
        
        date_key = action.ex_date.date()
        if date_key not in self._date_index:
            self._date_index[date_key] = []
        self._date_index[date_key].append(action)
        
        # Keep lists sorted
        self._symbol_index[action.symbol].sort(key=lambda x: x.ex_date)
        self._date_index[date_key].sort(key=lambda x: x.ex_date)
    
    def get_actions_by_symbol(self, 
                             symbol: str,
                             start_date: Optional[pd.Timestamp] = None,
                             end_date: Optional[pd.Timestamp] = None) -> List[CorporateAction]:
        """
        Get corporate actions for a specific symbol.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        start_date : pd.Timestamp, optional
            Start date filter
        end_date : pd.Timestamp, optional
            End date filter
            
        Returns
        -------
        List[CorporateAction]
            Filtered list of corporate actions
        """
        if symbol not in self._symbol_index:
            return []
        
        actions = self._symbol_index[symbol]
        
        # Apply date filters
        if start_date:
            actions = [a for a in actions if a.ex_date >= start_date]
        if end_date:
            actions = [a for a in actions if a.ex_date <= end_date]
        
        return actions
    
    def get_actions_by_date(self, 
                           target_date: Union[pd.Timestamp, date]) -> List[CorporateAction]:
        """
        Get all corporate actions for a specific date.
        
        Parameters
        ----------
        target_date : pd.Timestamp or date
            Target date
            
        Returns
        -------
        List[CorporateAction]
            List of corporate actions on target date
        """
        if isinstance(target_date, pd.Timestamp):
            target_date = target_date.date()
        
        return self._date_index.get(target_date, [])
    
    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load corporate actions from a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with corporate action data
            
        Expected columns:
        - symbol: str
        - action_type: str
        - ex_date: datetime
        - cash_amount: float (optional)
        - split_ratio: float (optional)
        - distribution_ratio: float (optional)
        """
        required_cols = ['symbol', 'action_type', 'ex_date']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        for _, row in df.iterrows():
            try:
                action_type = CorporateActionType(row['action_type'])
                ex_date = pd.to_datetime(row['ex_date'])
                
                action = CorporateAction(
                    symbol=row['symbol'],
                    action_type=action_type,
                    ex_date=ex_date,
                    record_date=pd.to_datetime(row.get('record_date')) if 'record_date' in row and pd.notna(row['record_date']) else None,
                    payment_date=pd.to_datetime(row.get('payment_date')) if 'payment_date' in row and pd.notna(row['payment_date']) else None,
                    cash_amount=row.get('cash_amount') if 'cash_amount' in row and pd.notna(row['cash_amount']) else None,
                    split_ratio=row.get('split_ratio') if 'split_ratio' in row and pd.notna(row['split_ratio']) else None,
                    distribution_ratio=row.get('distribution_ratio') if 'distribution_ratio' in row and pd.notna(row['distribution_ratio']) else None,
                    exchange_ratio=row.get('exchange_ratio') if 'exchange_ratio' in row and pd.notna(row['exchange_ratio']) else None,
                    new_symbol=row.get('new_symbol') if 'new_symbol' in row and pd.notna(row['new_symbol']) else None,
                    description=row.get('description') if 'description' in row and pd.notna(row['description']) else None
                )
                
                self.add_action(action)
            except Exception as e:
                print(f"Warning: Failed to load action for {row.get('symbol', 'unknown')}: {e}")
                continue
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert database to DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all corporate actions
        """
        if not self.actions:
            return pd.DataFrame()
        
        data = []
        for action in self.actions:
            data.append({
                'symbol': action.symbol,
                'action_type': action.action_type.value,
                'ex_date': action.ex_date,
                'record_date': action.record_date,
                'payment_date': action.payment_date,
                'cash_amount': action.cash_amount,
                'split_ratio': action.split_ratio,
                'distribution_ratio': action.distribution_ratio,
                'exchange_ratio': action.exchange_ratio,
                'new_symbol': action.new_symbol,
                'currency': action.currency,
                'description': action.description,
                'source': action.source
            })
        
        return pd.DataFrame(data)
    
    def _validate_action(self, action: CorporateAction) -> None:
        """Validate corporate action data."""
        if not action.symbol:
            raise ValueError("Symbol cannot be empty")
        
        if action.ex_date is None:
            raise ValueError("Ex-date is required")
        
        # Check for duplicates
        existing = self.get_actions_by_symbol(action.symbol)
        for existing_action in existing:
            if (existing_action.ex_date == action.ex_date and 
                existing_action.action_type == action.action_type):
                raise ValueError(f"Duplicate action: {action.symbol} {action.action_type} on {action.ex_date}")


def load_crsp_corporate_actions(file_path: str) -> CorporateActionDatabase:
    """
    Load corporate actions from CRSP-format file.
    
    Parameters
    ----------
    file_path : str
        Path to CRSP corporate actions file
        
    Returns
    -------
    CorporateActionDatabase
        Loaded corporate actions database
    """
    try:
        df = pd.read_csv(file_path)
        
        # Map CRSP action codes to our types
        crsp_mapping = {
            '5523': CorporateActionType.CASH_DIVIDEND,
            '5533': CorporateActionType.SPECIAL_DIVIDEND,
            '5543': CorporateActionType.STOCK_SPLIT,
            '5553': CorporateActionType.STOCK_DIVIDEND,
            '5563': CorporateActionType.SPIN_OFF,
        }
        
        # Convert CRSP format to our format
        if 'actn' in df.columns:
            df['action_type'] = df['actn'].map(crsp_mapping)
            df = df.dropna(subset=['action_type'])
        
        db = CorporateActionDatabase()
        db.load_from_dataframe(df)
        
        return db
        
    except Exception as e:
        print(f"Error loading CRSP data: {e}")
        return CorporateActionDatabase()


# Utility functions
def create_sample_corporate_actions(symbol: str, 
                                  start_date: pd.Timestamp,
                                  end_date: pd.Timestamp) -> List[CorporateAction]:
    """
    Create sample corporate actions for testing.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    start_date : pd.Timestamp
        Start date
    end_date : pd.Timestamp
        End date
        
    Returns
    -------
    List[CorporateAction]
        Sample corporate actions
    """
    actions = []
    
    # Sample dividend
    div_date = start_date + (end_date - start_date) / 3
    actions.append(CorporateAction(
        symbol=symbol,
        action_type=CorporateActionType.CASH_DIVIDEND,
        ex_date=div_date,
        cash_amount=0.50,
        description="Quarterly dividend"
    ))
    
    # Sample split
    split_date = start_date + 2 * (end_date - start_date) / 3
    actions.append(CorporateAction(
        symbol=symbol,
        action_type=CorporateActionType.STOCK_SPLIT,
        ex_date=split_date,
        split_ratio=2.0,
        description="2-for-1 stock split"
    ))
    
    return actions