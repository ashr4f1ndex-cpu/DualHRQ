
from typing import Iterator
import pandas as pd
import numpy as np

def calendar_walkforward(index: "pd.DatetimeIndex", train_years:int=8, test_years:int=1, embargo_days:int=5
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    years = sorted(set(index.year.tolist()))
    if len(years) < train_years + test_years:
        return
    for i in range(train_years, len(years) - test_years + 1):
        tr_start_year = years[i - train_years]
        tr_end_year   = years[i - 1]
        te_start_year = years[i]
        te_end_year   = years[i + test_years - 1]

        train_start = pd.Timestamp(f"{tr_start_year}-01-01")
        train_end   = pd.Timestamp(f"{tr_end_year}-12-31")
        test_start  = pd.Timestamp(f"{te_start_year}-01-01") + pd.Timedelta(days=embargo_days)
        test_end    = pd.Timestamp(f"{te_end_year}-12-31")

        train_start = max(train_start, index.min())
        train_end   = min(train_end,   index.max())
        test_start  = max(test_start,  index.min())
        test_end    = min(test_end,    index.max())
        if test_start > test_end or train_start > train_end:
            continue
        yield (train_start, train_end, test_start, test_end)

def purged_kfold(index: "pd.DatetimeIndex", n_splits:int=5, embargo_days:int=5):
    """Simple purged K-Fold: split index into contiguous blocks, embargo edges."""
    n = len(index)
    fold_size = n // n_splits
    for k in range(n_splits):
        te_start_i = k*fold_size
        te_end_i = (k+1)*fold_size if k < n_splits-1 else n
        te_start = index[te_start_i]
        te_end = index[te_end_i-1]
        tr_mask = (index < te_start - pd.Timedelta(days=embargo_days)) | (index > te_end + pd.Timedelta(days=embargo_days))
        yield (index[tr_mask].min(), index[tr_mask].max(), te_start, te_end)
