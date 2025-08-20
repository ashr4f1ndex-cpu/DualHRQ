
import pandas as pd
from lab_v10.src.intraday.backtest_intraday_strict import compute_ssr_states, luld_bands, simulate_short_with_constraints

def _mkt_index(day: str):
    return pd.date_range(f"{day} 09:30", f"{day} 16:00", freq="1min")

def test_ssr_persists_to_next_day():
    idx = _mkt_index("2025-01-06").append(_mkt_index("2025-01-07"))
    df = pd.DataFrame(index=idx)
    df["prev_close"] = 100.0
    df["low"] = 100.0
    df.loc[idx[100], "low"] = 89.0
    ssr = compute_ssr_states(df[["low","prev_close"]])
    d1 = idx[0].date()
    d2 = (idx[0] + pd.Timedelta(days=1)).date()
    assert ssr.loc[ssr.index.date == d1].any()
    assert ssr.loc[ssr.index.date == d2].all()

def test_luld_bands_double_last_25_min():
    idx = _mkt_index("2025-01-08")
    ref = pd.Series(100.0, index=idx)
    lo, hi = luld_bands(ref, 0.10, idx.to_series())
    assert lo.loc[idx[0]] == 90 and hi.loc[idx[0]] == 110
    t = pd.Timestamp("2025-01-08 15:35")
    assert abs(lo.loc[t] - 80.0) < 1e-9 and abs(hi.loc[t] - 120.0) < 1e-9

def test_uptick_block_under_ssr():
    idx = _mkt_index("2025-01-09")
    df = pd.DataFrame(index=idx)
    df["open"]=df["high"]=df["low"]=df["close"]=df["vwap"]=100.0
    df["prev_close"] = 100.0
    df["low"] = 89.0
    sig = pd.Series(False, index=idx); sig.iloc[10] = True
    pnl, eq, trades = simulate_short_with_constraints(df, sig)
    assert trades.empty
