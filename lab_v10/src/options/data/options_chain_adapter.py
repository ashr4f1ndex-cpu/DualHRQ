
import pandas as pd
import numpy as np
from pathlib import Path

CSV_SCHEMA = """
# Option chain daily snapshot schema (example)
date, close, iv_entry, iv_exit, expiry
2023-01-03, 383.12, 0.196, 0.202, 2023-02-03
...
"""

def load_chain_series(csv_path: str=None, start:str=None, end:str=None):
    """Load or synthesize time series:
      Returns S (close), iv_now (entry IV), iv_future (exit IV proxy), expiry (per-date selected contract expiry).
      If csv provided, expect columns: date, close, iv_entry, iv_exit, expiry (YYYY-MM-DD)
    """
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=["date","expiry"]).set_index("date").sort_index()
        if start: df = df[df.index >= pd.to_datetime(start)]
        if end:   df = df[df.index <= pd.to_datetime(end)]
        S = df["close"].asfreq("B").fillna(method="ffill")
        iv_now = df.get("iv_entry", pd.Series(0.2, index=S.index)).asfreq("B").fillna(method="ffill")
        iv_future = df.get("iv_exit", pd.Series(0.2, index=S.index)).asfreq("B").fillna(method="ffill")
        expiry = df.get("expiry", pd.Series(pd.NaT, index=S.index)).asfreq("B").fillna(method="ffill")
        return S, iv_now, iv_future, expiry

    # Fallback: synthesize series, expiry 30 calendar days ahead
    idx = pd.date_range(start or "2016-01-01", end or "2024-12-31", freq="B")
    S = pd.Series(400.0, index=idx)
    ret = np.random.normal(0, 0.001, size=len(idx))
    S = (S * (1 + pd.Series(ret, index=idx))).cumprod() / (1 + ret[0])
    iv_now = pd.Series(0.20 + 0.03*np.sin(np.linspace(0,20,len(idx))), index=idx)
    iv_future = iv_now.rolling(5, min_periods=1).mean()
    expiry = pd.Series(idx + pd.Timedelta(days=30), index=idx)  # ACT calendar days
    return S, iv_now, iv_future, expiry
