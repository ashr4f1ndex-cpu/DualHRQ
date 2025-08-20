"""
hrm_input.py
==============

Utilities for preparing inputs to the HRM model.  This includes
fitting scalers on training data to avoid leakage, constructing
fixed-size token sequences for the H- and L-modules, and merging
intraday data into day-level targets for Head-B.  The functions here
support the simplified HRMNet implementation provided in hrm_net.py.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


@dataclass
class TokenConfig:
    daily_window: int = 192
    minutes_per_day: int = 390


@dataclass
class FittedScalers:
    daily: StandardScaler
    intraday: StandardScaler


def fit_scalers(
    X_daily: pd.DataFrame,
    X_intraday: pd.DataFrame,
    train_idx: np.ndarray,
) -> FittedScalers:
    """Fit StandardScaler objects to daily and intraday data.

    The scalers are fit only on the training portion of the data to
    prevent information leakage.  For the daily scaler, values from
    all training dates are used.  For the intraday scaler, values
    from all times whose date is in the training set are used.

    Parameters
    ----------
    X_daily : pd.DataFrame
        The daily feature matrix indexed by date.
    X_intraday : pd.DataFrame
        The intraday feature matrix indexed by timestamp.
    train_idx : np.ndarray
        Integer indices of the training dates in the daily index.

    Returns
    -------
    FittedScalers
        A dataclass containing the fitted daily and intraday scalers.
    """
    scaler_daily = StandardScaler()
    scaler_intraday = StandardScaler()
    # Fit daily on train dates
    train_days = X_daily.index.unique()[train_idx]
    scaler_daily.fit(X_daily.loc[train_days].values)
    # Fit intraday on timestamps belonging to training dates
    train_dayset = set(pd.to_datetime(train_days))
    mask = X_intraday.index.normalize().isin(train_dayset)
    scaler_intraday.fit(X_intraday.loc[mask].values)
    return FittedScalers(scaler_daily, scaler_intraday)


def make_h_tokens(
    X_daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    scalers: FittedScalers,
    cfg: TokenConfig,
) -> torch.Tensor:
    """Create H tokens for each sample in `dates`.

    A sliding window of fixed length `daily_window` is taken from
    X_daily ending at each date.  If insufficient history exists the
    sequence is left padded with the earliest available row.  Values
    are scaled using the fitted daily scaler and returned as a tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape (len(dates), daily_window, num_features).
    """
    D = cfg.daily_window
    feats = []
    for d in dates:
        window = X_daily.loc[:d].tail(D)
        # left pad if needed
        if len(window) < D:
            pad = np.repeat(window.iloc[[0]].values, D - len(window), axis=0)
            mat = np.vstack([pad, window.values])
        else:
            mat = window.values
        feats.append(mat)
    arr = np.asarray(feats)
    # reshape for scaler
    B, T, F = arr.shape
    arr2 = arr.reshape(B * T, F)
    arr2 = scalers.daily.transform(arr2)
    arr = arr2.reshape(B, T, F)
    return torch.from_numpy(arr).float()


def make_l_tokens_for_day(
    X_intraday: pd.DataFrame,
    day: pd.Timestamp,
    scalers: FittedScalers,
    cfg: TokenConfig,
) -> torch.Tensor:
    """Create intraday token sequence for a single day.

    A fixed number of minutes per day (minutes_per_day) is taken from
    X_intraday.  Missing minutes are padded with the last available
    observation.  Values are scaled using the fitted intraday scaler.

    Returns
    -------
    torch.Tensor
        A tensor of shape (1, minutes_per_day, num_features).
    """
    M = cfg.minutes_per_day
    block = X_intraday.loc[X_intraday.index.normalize() == day.normalize()]
    if len(block) == 0:
        # return zeros if no intraday data exists
        return torch.zeros((1, M, X_intraday.shape[1]), dtype=torch.float32)
    if len(block) < M:
        tail = np.repeat(block.iloc[[-1]].values, M - len(block), axis=0)
        mat = np.vstack([block.values, tail])
    else:
        mat = block.iloc[:M].values
    arr = scalers.intraday.transform(mat)
    return torch.from_numpy(arr).unsqueeze(0).float()