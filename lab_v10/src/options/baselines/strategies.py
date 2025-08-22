
import pandas as pd


def always_long_signal(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(1.0, index=index)

def iv_slope_long_signal(iv_now: pd.Series, thresh: float=0.0) -> pd.Series:
    slope = iv_now.diff()
    return (slope > thresh).astype(float).reindex(iv_now.index).fillna(0.0)
