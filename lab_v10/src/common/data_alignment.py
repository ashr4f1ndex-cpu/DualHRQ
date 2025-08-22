"""
Data alignment utilities for time zones and market calendars.

This module provides comprehensive tools for handling time zone conversions,
market calendar alignment, and data synchronization across different markets
and data sources. Essential for dual-book trading strategies operating across
multiple exchanges and time zones.

Key features:
- Market calendar support (NYSE, NASDAQ, CME, CBOE)
- Time zone conversion and DST handling
- Trading session alignment
- Data resampling and gap filling
- Holiday and half-day trading adjustments
- Cross-market synchronization

References:
- NYSE/NASDAQ Trading Calendar
- CBOE Trading Hours
- CME Group Trading Hours
- Federal Reserve Bank holidays
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class TradingSession:
    """Trading session definition."""
    name: str
    start_time: time
    end_time: time
    timezone: str
    days_of_week: Set[int]  # 0=Monday, 6=Sunday
    
    def is_trading_day(self, date: datetime) -> bool:
        """Check if date is a trading day."""
        return date.weekday() in self.days_of_week
    
    def get_session_start(self, date: datetime) -> pd.Timestamp:
        """Get session start timestamp for given date."""
        return pd.Timestamp.combine(date.date(), self.start_time).tz_localize(self.timezone)
    
    def get_session_end(self, date: datetime) -> pd.Timestamp:
        """Get session end timestamp for given date."""
        return pd.Timestamp.combine(date.date(), self.end_time).tz_localize(self.timezone)


class MarketCalendar(ABC):
    """Abstract base class for market calendars."""
    
    @abstractmethod
    def is_trading_day(self, date: Union[datetime, pd.Timestamp]) -> bool:
        """Check if date is a trading day."""
        pass
    
    @abstractmethod
    def get_trading_days(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        """Get trading days in date range."""
        pass
    
    @abstractmethod
    def get_trading_hours(self, date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get trading hours for specific date."""
        pass


class USEquityCalendar(MarketCalendar):
    """
    US equity market calendar (NYSE/NASDAQ).
    
    Handles regular trading hours, holidays, and early closes.
    """
    
    def __init__(self):
        """Initialize US equity calendar."""
        self.timezone = "America/New_York"
        self.regular_open = time(9, 30)
        self.regular_close = time(16, 0)
        self.early_close = time(13, 0)
        
        # Define known holidays (simplified - in practice would use external calendar)
        self.holidays = [
            "2024-01-01",  # New Year's Day
            "2024-01-15",  # MLK Day
            "2024-02-19",  # Presidents Day
            "2024-03-29",  # Good Friday
            "2024-05-27",  # Memorial Day
            "2024-06-19",  # Juneteenth
            "2024-07-04",  # Independence Day
            "2024-09-02",  # Labor Day
            "2024-11-28",  # Thanksgiving
            "2024-12-25",  # Christmas
            # Add more years as needed
        ]
        
        # Early close days (simplified)
        self.early_close_days = [
            "2024-07-03",  # Day before Independence Day
            "2024-11-29",  # Day after Thanksgiving
            "2024-12-24",  # Christmas Eve
        ]
    
    def is_trading_day(self, date: Union[datetime, pd.Timestamp]) -> bool:
        """Check if date is a trading day."""
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        
        # Check weekends
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check holidays
        date_str = date.strftime("%Y-%m-%d")
        if date_str in self.holidays:
            return False
        
        return True
    
    def get_trading_days(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        """Get trading days in date range."""
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [day for day in all_days if self.is_trading_day(day)]
        return pd.DatetimeIndex(trading_days)
    
    def get_trading_hours(self, date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get trading hours for specific date."""
        if not self.is_trading_day(date):
            raise ValueError(f"{date.date()} is not a trading day")
        
        date_str = date.strftime("%Y-%m-%d")
        
        # Check for early close
        if date_str in self.early_close_days:
            close_time = self.early_close
        else:
            close_time = self.regular_close
        
        open_ts = pd.Timestamp.combine(date.date(), self.regular_open).tz_localize(self.timezone)
        close_ts = pd.Timestamp.combine(date.date(), close_time).tz_localize(self.timezone)
        
        return open_ts, close_ts
    
    def is_market_open(self, timestamp: pd.Timestamp) -> bool:
        """Check if market is open at given timestamp."""
        # Convert to market timezone
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        timestamp = timestamp.tz_convert(self.timezone)
        
        # Check if trading day
        if not self.is_trading_day(timestamp):
            return False
        
        # Check if within trading hours
        try:
            open_time, close_time = self.get_trading_hours(timestamp)
            return open_time <= timestamp <= close_time
        except ValueError:
            return False


class OptionsCalendar(MarketCalendar):
    """
    Options market calendar (CBOE).
    
    Generally follows equity calendar but with some differences.
    """
    
    def __init__(self):
        """Initialize options calendar."""
        self.equity_calendar = USEquityCalendar()
        self.timezone = "America/Chicago"  # CBOE timezone
        self.regular_open = time(8, 30)
        self.regular_close = time(15, 0)
    
    def is_trading_day(self, date: Union[datetime, pd.Timestamp]) -> bool:
        """Check if date is a trading day."""
        return self.equity_calendar.is_trading_day(date)
    
    def get_trading_days(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        """Get trading days in date range."""
        return self.equity_calendar.get_trading_days(start_date, end_date)
    
    def get_trading_hours(self, date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get trading hours for specific date."""
        if not self.is_trading_day(date):
            raise ValueError(f"{date.date()} is not a trading day")
        
        open_ts = pd.Timestamp.combine(date.date(), self.regular_open).tz_localize(self.timezone)
        close_ts = pd.Timestamp.combine(date.date(), self.regular_close).tz_localize(self.timezone)
        
        return open_ts, close_ts


class DataAligner:
    """
    Comprehensive data alignment engine for multi-market trading strategies.
    
    Handles time zone conversions, calendar alignment, and data synchronization
    across different markets and data sources.
    """
    
    def __init__(self, 
                 primary_calendar: MarketCalendar = None,
                 reference_timezone: str = "America/New_York"):
        """
        Initialize data aligner.
        
        Parameters
        ----------
        primary_calendar : MarketCalendar, optional
            Primary market calendar for alignment
        reference_timezone : str
            Reference timezone for alignment
        """
        self.primary_calendar = primary_calendar or USEquityCalendar()
        self.reference_timezone = reference_timezone
        self.calendars = {
            'equity': USEquityCalendar(),
            'options': OptionsCalendar()
        }
    
    def align_to_trading_days(self, 
                             data: pd.DataFrame,
                             calendar_name: str = 'equity',
                             method: str = 'forward_fill') -> pd.DataFrame:
        """
        Align data to trading days only.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with DatetimeIndex
        calendar_name : str
            Calendar to use for alignment
        method : str
            Fill method for non-trading days
            
        Returns
        -------
        pd.DataFrame
            Aligned data
        """
        if calendar_name not in self.calendars:
            raise ValueError(f"Unknown calendar: {calendar_name}")
        
        calendar = self.calendars[calendar_name]
        
        # Get trading days in data range
        start_date = data.index.min()
        end_date = data.index.max()
        trading_days = calendar.get_trading_days(start_date, end_date)
        
        # Filter data to trading days only
        trading_data = data[data.index.normalize().isin(trading_days.normalize())]
        
        # Handle fill method
        if method == 'forward_fill':
            trading_data = trading_data.fillna(method='ffill')
        elif method == 'backward_fill':
            trading_data = trading_data.fillna(method='bfill')
        elif method == 'interpolate':
            trading_data = trading_data.interpolate()
        
        return trading_data
    
    def align_to_market_hours(self, 
                             data: pd.DataFrame,
                             calendar_name: str = 'equity',
                             include_premarket: bool = False,
                             include_afterhours: bool = False) -> pd.DataFrame:
        """
        Align data to market hours only.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with DatetimeIndex
        calendar_name : str
            Calendar to use for alignment
        include_premarket : bool
            Include pre-market hours
        include_afterhours : bool
            Include after-hours trading
            
        Returns
        -------
        pd.DataFrame
            Market hours aligned data
        """
        if calendar_name not in self.calendars:
            raise ValueError(f"Unknown calendar: {calendar_name}")
        
        calendar = self.calendars[calendar_name]
        aligned_data = []
        
        # Group by date
        for date, day_data in data.groupby(data.index.date):
            date_ts = pd.Timestamp(date)
            
            if not calendar.is_trading_day(date_ts):
                continue
            
            try:
                market_open, market_close = calendar.get_trading_hours(date_ts)
                
                # Convert to same timezone as data
                if day_data.index.tz is not None:
                    market_open = market_open.tz_convert(day_data.index.tz)
                    market_close = market_close.tz_convert(day_data.index.tz)
                
                # Filter to market hours
                market_mask = (day_data.index >= market_open) & (day_data.index <= market_close)
                market_hours_data = day_data[market_mask]
                
                if len(market_hours_data) > 0:
                    aligned_data.append(market_hours_data)
                    
            except ValueError:
                continue  # Skip non-trading days
        
        if aligned_data:
            return pd.concat(aligned_data)
        else:
            return pd.DataFrame()
    
    def convert_timezone(self, 
                        data: pd.DataFrame,
                        target_timezone: str,
                        source_timezone: str = None) -> pd.DataFrame:
        """
        Convert data to target timezone.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with DatetimeIndex
        target_timezone : str
            Target timezone
        source_timezone : str, optional
            Source timezone (if not in index)
            
        Returns
        -------
        pd.DataFrame
            Timezone converted data
        """
        data = data.copy()
        
        # Handle timezone localization
        if data.index.tz is None:
            if source_timezone is None:
                raise ValueError("source_timezone required for naive timestamps")
            data.index = data.index.tz_localize(source_timezone)
        
        # Convert to target timezone
        data.index = data.index.tz_convert(target_timezone)
        
        return data
    
    def synchronize_datasets(self, 
                           datasets: Dict[str, pd.DataFrame],
                           method: str = 'outer',
                           fill_method: str = 'forward_fill') -> Dict[str, pd.DataFrame]:
        """
        Synchronize multiple datasets to common time index.
        
        Parameters
        ----------
        datasets : Dict[str, pd.DataFrame]
            Dictionary of datasets to synchronize
        method : str
            Join method ('inner', 'outer', 'left')
        fill_method : str
            Fill method for missing values
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Synchronized datasets
        """
        if not datasets:
            return {}
        
        # Convert all to same timezone
        reference_tz = self.reference_timezone
        aligned_datasets = {}
        
        for name, data in datasets.items():
            if data.index.tz is None:
                # Assume data is in reference timezone if not specified
                data.index = data.index.tz_localize(reference_tz)
            else:
                data.index = data.index.tz_convert(reference_tz)
            aligned_datasets[name] = data
        
        # Find common time index
        if method == 'inner':
            # Intersection of all indices
            common_index = aligned_datasets[list(aligned_datasets.keys())[0]].index
            for data in aligned_datasets.values():
                common_index = common_index.intersection(data.index)
        elif method == 'outer':
            # Union of all indices
            common_index = aligned_datasets[list(aligned_datasets.keys())[0]].index
            for data in aligned_datasets.values():
                common_index = common_index.union(data.index)
        else:  # 'left' - use first dataset's index
            common_index = aligned_datasets[list(aligned_datasets.keys())[0]].index
        
        # Reindex all datasets
        synchronized_datasets = {}
        for name, data in aligned_datasets.items():
            reindexed = data.reindex(common_index)
            
            # Apply fill method
            if fill_method == 'forward_fill':
                reindexed = reindexed.fillna(method='ffill')
            elif fill_method == 'backward_fill':
                reindexed = reindexed.fillna(method='bfill')
            elif fill_method == 'interpolate':
                reindexed = reindexed.interpolate()
            
            synchronized_datasets[name] = reindexed
        
        return synchronized_datasets
    
    def resample_to_frequency(self, 
                             data: pd.DataFrame,
                             frequency: str,
                             agg_method: str = 'last',
                             label: str = 'right') -> pd.DataFrame:
        """
        Resample data to specified frequency.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        frequency : str
            Target frequency (e.g., '1min', '5min', '1H')
        agg_method : str
            Aggregation method for OHLCV data
        label : str
            Labeling method ('left' or 'right')
            
        Returns
        -------
        pd.DataFrame
            Resampled data
        """
        # Define aggregation rules for common columns
        ohlcv_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'vwap': 'mean'
        }
        
        # Custom aggregation for other columns
        custom_agg = {}
        for col in data.columns:
            if col.lower() in ohlcv_agg:
                custom_agg[col] = ohlcv_agg[col.lower()]
            elif data[col].dtype in ['float64', 'int64']:
                custom_agg[col] = agg_method
            else:
                custom_agg[col] = 'last'
        
        # Resample
        resampled = data.resample(frequency, label=label).agg(custom_agg)
        
        # Drop rows with all NaN values
        resampled = resampled.dropna(how='all')
        
        return resampled
    
    def fill_market_gaps(self, 
                        data: pd.DataFrame,
                        calendar_name: str = 'equity',
                        frequency: str = '1min',
                        method: str = 'forward_fill') -> pd.DataFrame:
        """
        Fill gaps in data during market hours.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with potential gaps
        calendar_name : str
            Calendar to use for market hours
        frequency : str
            Expected data frequency
        method : str
            Fill method for gaps
            
        Returns
        -------
        pd.DataFrame
            Data with filled gaps
        """
        if calendar_name not in self.calendars:
            raise ValueError(f"Unknown calendar: {calendar_name}")
        
        calendar = self.calendars[calendar_name]
        filled_data = []
        
        # Process each trading day
        for date, day_data in data.groupby(data.index.date):
            date_ts = pd.Timestamp(date)
            
            if not calendar.is_trading_day(date_ts):
                continue
            
            try:
                market_open, market_close = calendar.get_trading_hours(date_ts)
                
                # Convert to same timezone as data
                if day_data.index.tz is not None:
                    market_open = market_open.tz_convert(day_data.index.tz)
                    market_close = market_close.tz_convert(day_data.index.tz)
                
                # Create complete time index for the day
                complete_index = pd.date_range(
                    start=market_open,
                    end=market_close,
                    freq=frequency
                )
                
                # Reindex and fill
                day_filled = day_data.reindex(complete_index)
                
                if method == 'forward_fill':
                    day_filled = day_filled.fillna(method='ffill')
                elif method == 'backward_fill':
                    day_filled = day_filled.fillna(method='bfill')
                elif method == 'interpolate':
                    day_filled = day_filled.interpolate()
                
                filled_data.append(day_filled)
                
            except ValueError:
                continue
        
        if filled_data:
            return pd.concat(filled_data)
        else:
            return data


# Utility functions
def get_market_calendar(market: str) -> MarketCalendar:
    """
    Get market calendar by name.
    
    Parameters
    ----------
    market : str
        Market name ('equity', 'options', etc.)
        
    Returns
    -------
    MarketCalendar
        Market calendar instance
    """
    calendars = {
        'equity': USEquityCalendar,
        'nasdaq': USEquityCalendar,
        'nyse': USEquityCalendar,
        'options': OptionsCalendar,
        'cboe': OptionsCalendar
    }
    
    if market.lower() not in calendars:
        raise ValueError(f"Unknown market: {market}")
    
    return calendars[market.lower()]()


def align_ohlcv_data(data: pd.DataFrame,
                     target_frequency: str = '1min',
                     market: str = 'equity',
                     fill_gaps: bool = True) -> pd.DataFrame:
    """
    Convenience function to align OHLCV data to market standards.
    
    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data
    target_frequency : str
        Target frequency
    market : str
        Market type
    fill_gaps : bool
        Whether to fill gaps
        
    Returns
    -------
    pd.DataFrame
        Aligned OHLCV data
    """
    aligner = DataAligner()
    
    # Align to trading days
    aligned = aligner.align_to_trading_days(data, market)
    
    # Align to market hours
    aligned = aligner.align_to_market_hours(aligned, market)
    
    # Resample to target frequency
    if target_frequency:
        aligned = aligner.resample_to_frequency(aligned, target_frequency)
    
    # Fill gaps if requested
    if fill_gaps:
        aligned = aligner.fill_market_gaps(aligned, market, target_frequency)
    
    return aligned


def check_data_quality(data: pd.DataFrame,
                      expected_frequency: str = '1min',
                      market: str = 'equity') -> Dict[str, Union[bool, int, float]]:
    """
    Check data quality and alignment issues.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to check
    expected_frequency : str
        Expected data frequency
    market : str
        Market type
        
    Returns
    -------
    Dict[str, Union[bool, int, float]]
        Quality check results
    """
    results = {}
    
    # Check index properties
    results['has_datetime_index'] = isinstance(data.index, pd.DatetimeIndex)
    results['has_timezone'] = data.index.tz is not None
    results['is_monotonic'] = data.index.is_monotonic_increasing
    
    # Check for duplicates
    results['has_duplicates'] = data.index.has_duplicates
    results['duplicate_count'] = data.index.duplicated().sum()
    
    # Check frequency consistency
    if len(data) > 1:
        inferred_freq = pd.infer_freq(data.index)
        results['inferred_frequency'] = inferred_freq
        results['frequency_consistent'] = inferred_freq == expected_frequency
    else:
        results['inferred_frequency'] = None
        results['frequency_consistent'] = False
    
    # Check for gaps
    if expected_frequency and len(data) > 1:
        expected_periods = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq=expected_frequency
        )
        missing_periods = expected_periods.difference(data.index)
        results['missing_periods'] = len(missing_periods)
        results['gap_ratio'] = len(missing_periods) / len(expected_periods)
    else:
        results['missing_periods'] = 0
        results['gap_ratio'] = 0.0
    
    # Check data completeness
    results['missing_values'] = data.isnull().sum().sum()
    results['completeness_ratio'] = 1.0 - (data.isnull().sum().sum() / data.size)
    
    # Market hours alignment
    try:
        calendar = get_market_calendar(market)
        market_hours_data = []
        
        for date, day_data in data.groupby(data.index.date):
            date_ts = pd.Timestamp(date)
            if calendar.is_trading_day(date_ts):
                market_hours_data.append(day_data)
        
        if market_hours_data:
            market_hours_df = pd.concat(market_hours_data)
            results['market_hours_ratio'] = len(market_hours_df) / len(data)
        else:
            results['market_hours_ratio'] = 0.0
            
    except Exception:
        results['market_hours_ratio'] = None
    
    return results