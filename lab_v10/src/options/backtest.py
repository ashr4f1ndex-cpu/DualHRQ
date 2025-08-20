
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from .options import simulate_atm_straddle_roundtrip

@dataclass
class StraddleParams:
    r_annual: float = 0.01
    q_annual: float = 0.0
    bid_ask_bps: float = 12.5
    commission_per_contract: float = 0.65
    contracts: int = 1
    multiplier: int = 100
    threshold: float = 0.0
    dynamic_spread: bool = True
    style: str = "european"  # 'european' for SPX, 'american' for SPY
    session: str = "open"
    open_widen_bps: float = 5.0

def _spread_for_iv(iv: float, base_bps: float) -> float:
    if not np.isfinite(iv):
        return base_bps
    if iv < 0.15: return base_bps * 0.8
    if iv < 0.30: return base_bps * 1.0
    return base_bps * 1.4

def simulate_straddle_pnl(
    prices: "pd.Series",
    iv_now: "pd.Series",
    iv_future: "pd.Series",
    expiry: "pd.Series",
    signal: "pd.Series",
    params: Optional[StraddleParams] = None
) -> dict[str, "pd.Series"]:
    if params is None:
        params = StraddleParams()

    df = pd.DataFrame({
        "S": prices, "iv_now": iv_now, "iv_future": iv_future, "expiry": expiry, "signal": signal
    }).dropna()

    df["enter"] = (df["signal"] > params.threshold).astype(int)
    pnl = []; equity = []; entries = []; costs = []
    eq = 0.0
    dates = df.index.to_list()
    n = len(df)
    i = 0
    while i < n:
        if int(df.iloc[i]["enter"]) == 1:
            j = min(i + 5, n-1)  # default hold 5 days; can be externalized
            row_e = df.iloc[i]
            row_x = df.iloc[j]
            base_bps = params.bid_ask_bps
            if params.dynamic_spread:
                base_bps = _spread_for_iv(float(row_e["iv_now"]), base_bps)
            trade_pnl, trade_cost = simulate_atm_straddle_roundtrip(
                S_entry=float(row_e["S"]),
                iv_entry=float(row_e["iv_now"]),
                S_exit=float(row_x["S"]),
                iv_exit=float(row_x["iv_future"]),
                entry_ts=row_e.name.to_pydatetime(),
                expiry_ts=row_e["expiry"].to_pydatetime() if hasattr(row_e["expiry"], 'to_pydatetime') else row_e["expiry"],
                exit_ts=row_x.name.to_pydatetime(),
                r_annual=params.r_annual,
                q_annual=params.q_annual,
                bid_ask_bps=base_bps,
                commission_per_contract=params.commission_per_contract,
                contracts=params.contracts,
                multiplier=params.multiplier,
                style=params.style,
                dividends=None,
                session=params.session,
                open_widen_bps=params.open_widen_bps
            )
            for k in range(i, j):
                pnl.append(0.0); costs.append(0.0); eq += 0.0; equity.append(eq); entries.append(1 if k==i else 0)
            pnl.append(trade_pnl); costs.append(trade_cost); eq += trade_pnl; equity.append(eq); entries.append(0)
            i = j + 1
        else:
            pnl.append(0.0); costs.append(0.0); eq += 0.0; equity.append(eq); entries.append(0); i += 1
    pnl_s = pd.Series(pnl, index=df.index[:len(pnl)])
    eq_s = pd.Series(equity, index=df.index[:len(equity)])
    ent_s = pd.Series(entries, index=df.index[:len(entries)])
    c_s = pd.Series(costs, index=df.index[:len(costs)])
    return {"pnl": pnl_s, "equity": eq_s, "entry": ent_s, "costs": c_s}
