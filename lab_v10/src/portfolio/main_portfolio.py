
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from ..common.metrics import sharpe, max_drawdown
from datetime import datetime, timedelta

# Options book
from ..options.main_v3 import run as run_options

# Intraday components
from ..intraday.scanner_video import detect_parabolic_reversal
from ..intraday.backtest_intraday import simulate_intraday_backside_short
from ..intraday.aggregate_daily import aggregate_intraday_features

def _synth_minute(symbol:str, start:str, end:str, seed:int=42):
    """Synthetic 1-minute bars for demonstration; replace with real feed.    Creates a noisy intraday series per business day."""
    rng = np.random.default_rng(seed)
    idx_days = pd.date_range(start, end, freq="B")
    frames = []
    for d in idx_days:
        minutes = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), d + pd.Timedelta(hours=16), freq="T")
        base = 100 + np.cumsum(rng.normal(0, 0.05, size=len(minutes)))
        vol = rng.lognormal(mean=10, sigma=0.25, size=len(minutes)).astype(int)
        high = base + rng.normal(0.05,0.03,len(minutes))
        low = base - rng.normal(0.05,0.03,len(minutes))
        openp = np.r_[base[0], base[:-1]]
        close = base
        df = pd.DataFrame({"open":openp,"high":np.maximum(high, close),"low":np.minimum(low, close),"close":close,"volume":vol}, index=minutes)
        frames.append(df)
    return pd.concat(frames)

def run_portfolio(start:str, end:str, allocation_options: float=0.7, dynamic: bool=True, minute_csv:str=None):
    # Run options book to produce reports (Book A)
    run_options(start, end, outdir="reports/options")
    pnl_opt = pd.read_csv("reports/options/pnl.csv", index_col=0, parse_dates=True).iloc[:,0]
    eq_opt  = pd.read_csv("reports/options/equity_curve.csv", index_col=0, parse_dates=True).iloc[:,0]

    # Book B intraday
    if minute_csv and Path(minute_csv).exists():
        df_min = pd.read_csv(minute_csv, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    else:
        df_min = _synth_minute("SYN", start, end, seed=7)

    # Detect pattern & backtest per day
    signal, lh_level, diag = detect_parabolic_reversal(df_min)
    pnl_day, eq_day, trades = simulate_intraday_backside_short(df_min, signal, lh_level)
    # Aggregate to daily features and merge into options features in future runs if desired
    feats_daily = aggregate_intraday_features(df_min, signal, diag)
    feats_daily.to_csv("reports/intraday_daily_features.csv")

    # Align day PnL to options dates (sum per day)
    pnl_intraday = pnl_day.copy()
    pnl_intraday.index = pd.to_datetime(pnl_intraday.index.date)
    pnl_intraday = pnl_intraday.groupby(level=0).sum()
    pnl_intraday = pnl_intraday.reindex(pnl_opt.index).fillna(0.0)

    # Combine
    if dynamic:
        # Simple dynamic weights: proportional to rolling Sharpe, guarded by drawdown
        def rolling_sharpe(x, w=63):
            # avoid division by zero; clip negative values at zero so weights stay non-negative
            return (x.rolling(w).mean() / (x.rolling(w).std() + 1e-12)).clip(lower=0)
        # compute weights based on historical performance up to previous day (no lookahead)
        w_opt_raw = rolling_sharpe(pnl_opt, 63).shift(1)
        w_day_raw = rolling_sharpe(pnl_intraday, 63).shift(1)
        # normalise weights
        s = (w_opt_raw + w_day_raw).replace(0, np.nan)
        w_opt = (w_opt_raw / s).fillna(0.5)
        w_day = 1 - w_opt
        total_pnl = (w_opt * pnl_opt) + (w_day * pnl_intraday)
        weights = pd.DataFrame({"w_options": w_opt, "w_intraday": w_day})
    else:
        total_pnl = allocation_options*pnl_opt + (1-allocation_options)*pnl_intraday
        weights = pd.DataFrame({"w_options": allocation_options, "w_intraday": 1-allocation_options}, index=pnl_opt.index)

    total_eq  = total_pnl.cumsum()
    total_pnl.to_csv("reports/portfolio_pnl.csv")
    total_eq.to_csv("reports/portfolio_equity.csv")
    weights.to_csv("reports/portfolio_weights.csv")
    # Also persist intraday PnL
    pnl_intraday.to_csv("reports/intraday_pnl_daily.csv")
    print("[portfolio] wrote reports/portfolio_* files and intraday features.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2016-01-01")
    ap.add_argument("--end", type=str, default="2024-12-31")
    ap.add_argument("--alloc_options", type=float, default=0.7)
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--minute_csv", type=str, default=None)
    args = ap.parse_args()
    run_portfolio(args.start, args.end, args.alloc_options, dynamic=args.dynamic, minute_csv=args.minute_csv)
