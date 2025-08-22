
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SSRState:
    active_until: pd.Timestamp | None = None

def compute_ssr_states(df: pd.DataFrame) -> pd.Series:
    assert {'low', 'prev_close'}.issubset(df.columns)
    ssr = pd.Series(False, index=df.index)
    by_day = df.groupby(df.index.date)
    for day, d in by_day:
        low_today = float(d['low'].min())
        prev_close = float(d['prev_close'].iloc[0])
        if low_today <= prev_close * 0.9 + 1e-12:
            ssr.loc[d.index] = True
            nxt_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).date()
            nxt = df.index.date == nxt_day
            ssr.loc[ssr.index[nxt]] = True
    return ssr

def luld_bands(ref_price: pd.Series, pct: float, ts: pd.Series) -> tuple[pd.Series, pd.Series]:
    upper = ref_price * (1 + pct)
    lower = ref_price * (1 - pct)
    minutes = ts.dt.time
    mask_last25 = (minutes >= pd.to_datetime("15:35").time())
    upper = upper.where(~mask_last25, ref_price * (1 + 2*pct))
    lower = lower.where(~mask_last25, ref_price * (1 - 2*pct))
    return lower, upper

def _clip_to_bands(px: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.Series:
    return px.clip(lower=lo, upper=hi)

def simulate_short_with_constraints(df: pd.DataFrame, signal: pd.Series,
                                    stop_bps: float=35, base_slip_bps: float=5,
                                    apply_ssr: bool=True, apply_luld: bool=True):
    df = df.copy().sort_index()
    sig = signal.reindex(df.index).fillna(0).astype(bool)
    if apply_ssr:
        df['SSR'] = compute_ssr_states(df[['low', 'prev_close']])
    else:
        df['SSR'] = False
    if apply_luld:
        ref = df['close'].rolling(5, min_periods=1).mean()
        lo, hi = luld_bands(ref, 0.10, df.index.to_series())
    else:
        lo = pd.Series(-np.inf, index=df.index)
        hi = pd.Series(np.inf, index=df.index)

    trades = []
    pnl = np.zeros(len(df), dtype=float)
    for i, (ts, row) in enumerate(df.iterrows()):
        if not sig.iat[i]:
            continue
        if row['SSR']:
            continue
        fill = _clip_to_bands(pd.Series(row['open'], index=[ts]), lo, hi).iloc[0]
        stop = fill * (1 + stop_bps/1e4)
        out = row['close']
        if row['high'] >= stop:
            out = stop
        trade_pnl = (fill - out) - fill*(base_slip_bps/1e4)
        pnl[i] = trade_pnl
        trades.append((ts, float(fill), float(out), float(trade_pnl)))
    pnl = pd.Series(pnl, index=df.index, name="pnl")
    equity = pnl.cumsum().rename("equity")
    trades_df = (pd.DataFrame(trades, columns=["timestamp", "fill", "out", "pnl"]).set_index("timestamp")
                 if trades else pd.DataFrame(columns=["fill", "out", "pnl"]))
    return pnl, equity, trades_df
