"""
validation.py
===============

This module contains utilities for leakage‐aware cross validation and data
splitting.  In time series settings, naive train/test splits can leak
information from the future into the past.  The functions here implement
purged cross validation with optional embargo periods and combinational
purged cross validation (CPCV) for robust model evaluation.

Functions
---------

purge_embargo_splits
    Create a list of train/test index pairs based on a given series of
    timestamps.  A purge window removes the immediate past adjacent to
    each test fold and an embargo window removes observations adjacent to
    the test fold.

combinational_purged_cv
    Generate CPCV splits that combine multiple basic purged splits into
    larger test sets.  This is useful for model selection as it
    evaluates performance across many different regimes.

"""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd


def purge_embargo_splits(
    dates: pd.Series, n_splits: int, embargo_days: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return purged cross validation splits with embargo.

    Parameters
    ----------
    dates : pd.Series
        A series of datetime stamps representing the observation times.
        Assumed to be sorted in ascending order.
    n_splits : int
        The number of folds to create.
    embargo_days : int, optional
        The size of the embargo period (in number of distinct days) to
        remove from the training set on either side of the test set.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of (train_indices, test_indices) tuples where each
        element contains the integer indices into `dates` belonging to
        the training and test sets respectively.
    """
    assert n_splits >= 2, "n_splits must be at least 2"
    # Unique dates for splitting
    unique_days = pd.Index(pd.to_datetime(dates.dt.date).unique())
    fold_edges = np.linspace(0, len(unique_days), n_splits + 1, dtype=int)
    splits = []
    for i in range(n_splits):
        start, end = fold_edges[i], fold_edges[i + 1]
        test_days = unique_days[start:end]
        # skip empty folds
        if len(test_days) == 0:
            continue
        test_mask = dates.dt.date.isin(test_days)
        # embargo region around test days
        left = max(0, start - embargo_days)
        right = min(len(unique_days), end + embargo_days)
        embargo_days_idx = unique_days[left:right]
        embargo_mask = dates.dt.date.isin(embargo_days_idx)
        train_mask = ~embargo_mask
        splits.append((np.flatnonzero(train_mask.values), np.flatnonzero(test_mask.values)))
    return splits


def combinational_purged_cv(
    dates: pd.Series,
    n_splits: int = 5,
    k_folds: int = 2,
    embargo_days: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return combinational purged cross validation splits.

    CPCV is a form of cross validation for time series that tests the
    model on combinations of purged folds.  This yields many splits and
    helps to evaluate robustness across different regime combinations.

    Parameters
    ----------
    dates : pd.Series
        A series of datetime stamps representing the observation times.
        Assumed to be sorted in ascending order.
    n_splits : int, optional
        The number of basic folds to generate before combinations.  Must
        be at least 2.
    k_folds : int, optional
        The number of basic folds to combine into one test split.  For
        example, k_folds=2 will pick two basic folds as one test set.
    embargo_days : int, optional
        The size of the embargo period (in number of distinct days) to
        remove from the training set on either side of the test set.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of (train_indices, test_indices) tuples where each
        element contains the integer indices into `dates` belonging to
        the training and test sets respectively.
    """
    base_splits = purge_embargo_splits(dates, n_splits=n_splits, embargo_days=embargo_days)
    if k_folds <= 1 or not base_splits:
        return base_splits
    from itertools import combinations
    combos = []
    for comb in combinations(range(len(base_splits)), k_folds):
        # accumulate test indices from chosen folds
        test_idx = np.concatenate([base_splits[i][1] for i in comb])
        test_days = pd.to_datetime(dates.iloc[test_idx].dt.date.unique())
        embargo_mask = dates.dt.date.isin(test_days)
        train_idx = np.flatnonzero(~embargo_mask.values)
        combos.append((train_idx, test_idx))
    return combos