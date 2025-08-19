"""
Shared financial indicators used across the intraday and options modules.

This module centralises the calculation of common technical indicators such as
Volume‐Weighted Average Price (VWAP) and the Average True Range (ATR).  These
functions were previously duplicated in both the simple and video parabolic
scanners.  Centralising them here avoids subtle discrepancies in the
implementations and makes unit testing straightforward.

Notes
-----
- The VWAP implementation assumes a pandas Series or DataFrame with
  ``price`` and ``volume`` columns.  For intraday data the price
  is typically the close.  For series input the function returns a Series
  indexed as per the input.  For DataFrame input with columns ``open``,
  ``high``, ``low``, ``close`` and ``volume`` the VWAP is computed using
  the close.
- The ATR implementation computes a rolling average of the true range.  The
  true range is defined as the maximum of ``high - low``, ``abs(high - prev_close)``
  and ``abs(low - prev_close)``.  This implementation takes arrays or Series
  as inputs and returns a Series of ATR values.  The first value is NaN
  because there is no previous close.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

def vwap(price: pd.Series | pd.DataFrame, volume: pd.Series | None = None) -> pd.Series:
    """Compute the volume weighted average price (VWAP).

    Parameters
    ----------
    price : Series or DataFrame
        If a Series, interpreted as the price (e.g. close) with index representing
        timestamps.  If a DataFrame, must contain a ``close`` and ``volume`` column.
    volume : Series, optional
        If `price` is a Series then `volume` must be provided.  Otherwise ignored.

    Returns
    -------
    Series
        The cumulative VWAP.  The index matches the input index.

    Examples
    --------
    >>> vw = vwap(df["close"], df["volume"])
    >>> vw = vwap(df)  # df has 'close' and 'volume'
    """
    if isinstance(price, pd.DataFrame):
        if volume is not None:
            raise ValueError("When price is a DataFrame, volume should not be passed separately.")
        if "close" not in price.columns or "volume" not in price.columns:
            raise ValueError("DataFrame input must contain 'close' and 'volume' columns")
        price_series = price["close"]
        volume_series = price["volume"]
    else:
        if volume is None:
            raise ValueError("Volume Series must be provided when price is a Series.")
        price_series = price
        volume_series = volume
    price_arr = price_series.fillna(method="ffill").values
    vol_arr = volume_series.fillna(0).values
    # cumulative sums
    cum_pv = np.cumsum(price_arr * vol_arr)
    cum_v = np.cumsum(vol_arr)
    # avoid division by zero
    vwap_vals = np.divide(cum_pv, np.where(cum_v == 0, np.nan, cum_v))
    return pd.Series(vwap_vals, index=price_series.index)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR).

    Parameters
    ----------
    high, low, close : Series
        High, low and close price series indexed by time.
    n : int
        Lookback period over which the ATR is calculated (default: 14).

    Returns
    -------
    Series
        The ATR time series.  The first value is NaN because there is no
        previous close for the true range calculation.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    # compute true range components
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # rolling mean of true range
    atr = tr.rolling(n, min_periods=n).mean()
    return atr
