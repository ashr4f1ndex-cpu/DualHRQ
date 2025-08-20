
import pandas as pd
import numpy as np
from pathlib import Path

def _nearest_dte(group: pd.DataFrame, target_days: int = 30):
    # pick contract whose expiration days-from-date is closest to target_days
    g = group.copy()
    g["dte"] = (g["expiration"] - g["date"]).dt.days.abs()
    return g.sort_values("dte").iloc[0:1]

def load_cboe_datashop(csv_path: str, symbol: str=None, target_dte: int = 30):
    """Load a Cboe DataShop *end-of-day* options CSV for a single underlying.
    Expected flexible columns (best effort mapping):
      - date: 'quote_date' or 'date' or 'as_of_date'
      - expiration: 'expiration' or 'expiry'
      - option type: 'option_type' or 'cp_flag' ('C'/'P')
      - strike: 'strike' or 'strike_price'
      - iv: 'implied_volatility' or 'iv'
      - underlying: 'underlying_bid_1545'/'underlying_ask_1545' or 'underlying_price'
    We compute underlying close from mid of bid/ask when available.
    We then select the ATM strike nearest to S using |strike - S| and the contract with DTE closest to target_dte.
    Returns:
      S (Series), iv_entry (Series), iv_exit proxy (Series ~ 5d fwd IV), expiry (Series of per-day chosen contract expiry)
    """
    df = pd.read_csv(csv_path)
    # column mapping
    def pick(*names):
        for n in names:
            if n in df.columns: return n
        return None
    c_date = pick("quote_date","date","as_of_date")
    c_exp  = pick("expiration","expiry")
    c_cp   = pick("option_type","cp_flag")
    c_strk = pick("strike","strike_price")
    c_iv   = pick("implied_volatility","iv")
    c_ubid = pick("underlying_bid_1545","underlying_bid")
    c_uask = pick("underlying_ask_1545","underlying_ask")
    c_u    = pick("underlying_price","underlying_mid")
    # Basic checks
    if c_date is None or c_exp is None or c_strk is None or (c_iv is None and c_u is None and (c_ubid is None or c_uask is None)):
        raise ValueError("CSV missing required columns for Cboe DataShop loader.")
    # Normalize
    df = df.rename(columns={c_date:"date", c_exp:"expiration", c_strk:"strike"})
    if c_iv: df = df.rename(columns={c_iv:"iv"})
    if c_cp: df = df.rename(columns={c_cp:"cp"})
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    # Underlying mid
    if c_u is not None:
        df["S"] = pd.to_numeric(df[c_u], errors="coerce")
    elif c_ubid is not None and c_uask is not None:
        df["S"] = (pd.to_numeric(df[c_ubid], errors="coerce") + pd.to_numeric(df[c_uask], errors="coerce"))/2.0
    else:
        raise ValueError("No underlying price columns in CSV.")
    # Choose ATM per date
    df["abs_moneyness"] = (pd.to_numeric(df["strike"], errors="coerce") - df["S"]).abs()
    # Subset to ATM contracts by date: smallest |K-S| per date & expiration
    atm = df.sort_values(["date","abs_moneyness"]).groupby(["date","expiration"], as_index=False).first()
    # From those, pick expiration near target DTE per date
    atm_dte = atm.groupby("date", as_index=False, group_keys=False).apply(lambda g: _nearest_dte(g, target_days=target_dte))
    atm_dte = atm_dte.sort_values("date").dropna(subset=["iv","S","expiration"])
    # Build daily series
    atm_dte = atm_dte.set_index("date")
    S = atm_dte["S"].asfreq("B").ffill()
    iv_now = atm_dte["iv"].astype(float).asfreq("B").ffill()
    expiry = atm_dte["expiration"].asfreq("B").ffill()
    # Build a rough iv_exit proxy as 5-day forward IV (still from ATM-of-day; in production keep same contract)
    iv_exit = iv_now.shift(-5).ffill()
    return S, iv_now, iv_exit, expiry

def load_optionmetrics_ivydb(option_price_csv: str, underlying_csv: str, target_dte: int = 30):
    """Load OptionMetrics IvyDB Option_Price + Underlying_Price extracts.
    Option_Price expected columns (common fields): Date, Expiration, Strike, Call/Put, Best Bid, Best Offer, Implied Volatility, Security ID/Symbol
    Underlying_Price expected: date, close (or PRC) for the same underlying.
    We compute mid IV and pick ATM and expiration near target DTE per day, similar to DataShop.
    Returns S, iv_entry, iv_exit proxy, expiry
    """
    opt = pd.read_csv(option_price_csv)
    und = pd.read_csv(underlying_csv)
    # normalize columns
    opt = opt.rename(columns={
        "Date":"date","Expiration":"expiration","Strike":"strike","Call/Put":"cp",
        "Best Bid":"best_bid","Best Offer":"best_ask","Implied Volatility":"iv",
        "SECID":"secid","Security ID":"secid","Symbol":"symbol"
    })
    und = und.rename(columns={"date":"date","Date":"date","close":"close","PRC":"close"})
    opt["date"] = pd.to_datetime(opt["date"]); opt["expiration"] = pd.to_datetime(opt["expiration"])
    und["date"] = pd.to_datetime(und["date"])
    # Merge underlying to options by date
    S_daily = und.set_index("date")["close"].astype(float).asfreq("B").ffill()
    opt = opt.merge(und, on="date", how="left")
    opt["S"] = opt["close"].astype(float)
    # ATM selection
    opt["abs_moneyness"] = (pd.to_numeric(opt["strike"], errors="coerce") - opt["S"]).abs()
    atm = opt.sort_values(["date","abs_moneyness"]).groupby(["date","expiration"], as_index=False).first()
    atm_dte = atm.groupby("date", as_index=False, group_keys=False).apply(lambda g: _nearest_dte(g, target_days=target_dte))
    atm_dte = atm_dte.sort_values("date").dropna(subset=["iv","S","expiration"])
    atm_dte = atm_dte.set_index("date")
    S = atm_dte["S"].asfreq("B").ffill()
    iv_now = atm_dte["iv"].astype(float).asfreq("B").ffill()
    expiry = atm_dte["expiration"].asfreq("B").ffill()
    iv_exit = iv_now.shift(-5).ffill()
    return S, iv_now, iv_exit, expiry


# -------- Public / low-cost dataset loaders (CSV exports) --------

def _flex_pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def load_dolthub_options_csv(csv_path: str, symbol: str=None, target_dte:int=30):
    """Load a CSV export from the DoltHub options database (e.g., table `option_chain`).
    Expected flexible columns (best-effort): date, expiration/expiry, strike, call/put (cp), iv/iv_mid/implied_volatility, underlying price (underlying/close/price).
    Filter by `symbol` if provided via a column like `act_symbol` or `symbol`.
    """
    import pandas as pd, numpy as np
    from datetime import timedelta
    df = pd.read_csv(csv_path)
    # Flexible column mapping
    c_date = _flex_pick(df, "date", "as_of_date", "quote_date")
    c_exp  = _flex_pick(df, "expiration", "expiry", "expiration_date")
    c_strk = _flex_pick(df, "strike", "strike_price", "k")
    c_iv   = _flex_pick(df, "iv", "iv_mid", "implied_volatility", "iv_mean")
    c_sym  = _flex_pick(df, "act_symbol", "symbol", "underlying_symbol")
    c_S    = _flex_pick(df, "underlying_price", "underlying", "S", "close")
    if c_date is None or c_exp is None or c_strk is None:
        raise ValueError("CSV missing required columns for DoltHub loader (date/expiration/strike).")
    if symbol and c_sym:
        df = df[df[c_sym] == symbol]
    df = df.rename(columns={c_date:"date", c_exp:"expiration", c_strk:"strike"})
    if c_iv: df = df.rename(columns={c_iv:"iv"})
    if c_S:  df = df.rename(columns={c_S:"S"})
    df["date"] = pd.to_datetime(df["date"]); df["expiration"] = pd.to_datetime(df["expiration"])
    # If no underlying column, infer S by grouping (e.g., mid of strikes around ATM) – fallback rough estimate
    if "S" not in df.columns:
        # Approximate S as weighted average of (strike near 0.5 delta) if available, else median strike of near-the-money
        df["S"] = df.groupby("date")["strike"].transform("median")
    # Pick ATM per date and nearest target DTE
    df["abs_moneyness"] = (pd.to_numeric(df["strike"], errors="coerce") - df["S"]).abs()
    atm = df.sort_values(["date","abs_moneyness"]).groupby(["date","expiration"], as_index=False).first()
    # choose expiration closest to target_dte (in calendar days)
    days = (atm["expiration"] - atm["date"]).dt.days
    atm["dte_err"] = (days - target_dte).abs()
    atm_dte = atm.sort_values(["date","dte_err"]).groupby("date", as_index=False).first().set_index("date").sort_index()
    if "dte_err" in atm_dte.columns:
        atm_dte = atm_dte.drop(columns=["dte_err"])
    # series
    S = atm_dte["S"].astype(float).asfreq("B").ffill()
    iv_now = (atm_dte["iv"] if "iv" in atm_dte.columns else pd.Series(0.2, index=S.index)).astype(float).asfreq("B").ffill()
    expiry = atm_dte["expiration"].asfreq("B").ffill()
    iv_exit = iv_now.shift(-5).ffill()
    return S, iv_now, iv_exit, expiry

def load_marketdata_csv(csv_path: str, symbol: str=None, target_dte:int=30):
    """Load a CSV exported from MarketData.app (options chain with IV/Greeks).
    Columns vary by export; we flex-map date/expiration/strike/iv/underlying.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    c_date = _flex_pick(df, "date","as_of","as_of_date","quote_date")
    c_exp  = _flex_pick(df, "expiration","expiry","exp")
    c_strk = _flex_pick(df, "strike","strike_price")
    c_iv   = _flex_pick(df, "implied_volatility","iv","iv_mid")
    c_S    = _flex_pick(df, "underlying_price","underlying_mid","S","close")
    c_sym  = _flex_pick(df, "underlying_symbol","symbol","act_symbol")
    if c_date is None or c_exp is None or c_strk is None:
        raise ValueError("CSV missing required columns for MarketData.app loader (date/expiration/strike).")
    if symbol and c_sym:
        df = df[df[c_sym] == symbol]
    df = df.rename(columns={c_date:"date", c_exp:"expiration", c_strk:"strike"})
    if c_iv: df = df.rename(columns={c_iv:"iv"})
    if c_S:  df = df.rename(columns={c_S:"S"})
    df["date"] = pd.to_datetime(df["date"]); df["expiration"] = pd.to_datetime(df["expiration"])
    if "S" not in df.columns:
        df["S"] = df.groupby("date")["strike"].transform("median")
    df["abs_moneyness"] = (pd.to_numeric(df["strike"], errors="coerce") - df["S"]).abs()
    atm = df.sort_values(["date","abs_moneyness"]).groupby(["date","expiration"], as_index=False).first()
    days = (atm["expiration"] - atm["date"]).dt.days
    atm["dte_err"] = (days - target_dte).abs()
    atm_dte = atm.sort_values(["date","dte_err"]).groupby("date", as_index=False).first().set_index("date").sort_index()
    if "dte_err" in atm_dte.columns:
        atm_dte = atm_dte.drop(columns=["dte_err"])
    S = atm_dte["S"].astype(float).asfreq("B").ffill()
    iv_now = (atm_dte["iv"] if "iv" in atm_dte.columns else pd.Series(0.2, index=S.index)).astype(float).asfreq("B").ffill()
    expiry = atm_dte["expiration"].asfreq("B").ffill()
    iv_exit = iv_now.shift(-5).ffill()
    return S, iv_now, iv_exit, expiry

def load_databento_csv(csv_path: str, symbol: str=None, target_dte:int=30):
    """Load a CSV downloaded via Databento (e.g., OPRA summary/quotes with IV calcs).
    Flexible mapping similar to others; if multiple underlyings present, filter by `symbol`.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    c_date = _flex_pick(df, "date","ts","as_of")
    c_exp  = _flex_pick(df, "expiration","expiry")
    c_strk = _flex_pick(df, "strike","k")
    c_iv   = _flex_pick(df, "iv","iv_mid","implied_volatility")
    c_S    = _flex_pick(df, "underlying_price","S","close","underlier")
    c_sym  = _flex_pick(df, "underlying_symbol","sym_root","act_symbol","symbol")
    if c_date is None or c_exp is None or c_strk is None:
        raise ValueError("CSV missing required columns for Databento loader (date/expiration/strike).")
    if symbol and c_sym:
        df = df[df[c_sym] == symbol]
    df = df.rename(columns={c_date:"date", c_exp:"expiration", c_strk:"strike"})
    if c_iv: df = df.rename(columns={c_iv:"iv"})
    if c_S:  df = df.rename(columns={c_S:"S"})
    df["date"] = pd.to_datetime(df["date"]); df["expiration"] = pd.to_datetime(df["expiration"])
    if "S" not in df.columns:
        df["S"] = df.groupby("date")["strike"].transform("median")
    df["abs_moneyness"] = (pd.to_numeric(df["strike"], errors="coerce") - df["S"]).abs()
    atm = df.sort_values(["date","abs_moneyness"]).groupby(["date","expiration"], as_index=False).first()
    days = (atm["expiration"] - atm["date"]).dt.days
    atm["dte_err"] = (days - target_dte).abs()
    atm_dte = atm.sort_values(["date","dte_err"]).groupby("date", as_index=False).first().set_index("date").sort_index()
    if "dte_err" in atm_dte.columns:
        atm_dte = atm_dte.drop(columns=["dte_err"])
    S = atm_dte["S"].astype(float).asfreq("B").ffill()
    iv_now = (atm_dte["iv"] if "iv" in atm_dte.columns else pd.Series(0.2, index=S.index)).astype(float).asfreq("B").ffill()
    expiry = atm_dte["expiration"].asfreq("B").ffill()
    iv_exit = iv_now.shift(-5).ffill()
    return S, iv_now, iv_exit, expiry
