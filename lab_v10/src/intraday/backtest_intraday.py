
import pandas as pd
import numpy as np

def ssr_active(prev_close: float, low_today: float) -> bool:
    return (low_today <= prev_close * 0.9)

def simulate_intraday_backside_short(
    df: pd.DataFrame,
    signal: pd.Series,
    lh_level: pd.Series,
    stop_bps: float = 35,
    target_to_vwap: bool = True,
    slippage_bps: int = 5,
    apply_ssr: bool = True,
    apply_luld: bool = True,
    luld_pct: float = 0.10,
    luld_state_bars: int = 3,
    luld_pause_bars: int = 5,
    shock_prob: float = 0.0,
    shock_bps: int = 30,
    seed: int = None,
    borrow_availability: float = 1.0,
    borrow_rate_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series, int]:
    """
    Simulate short entries on the backside of a parabolic intraday move with
    optional enforcement of SSR (Short Sale Restriction) and LULD (Limit
    Up–Limit Down) price bands.

    Parameters
    ----------
    df : pd.DataFrame
        Intraday OHLCV data indexed by timestamp.  Must contain
        columns ``open``, ``high``, ``low``, ``close``, ``volume``.
    signal : pd.Series
        Boolean series aligned to ``df`` indicating trade entry points
        (fires on the bar immediately preceding entry).
    lh_level : pd.Series
        Series of pivot (lower–high) levels; the stop is placed above
        this level.
    stop_bps : float
        Stop distance in basis points above the pivot.
    target_to_vwap : bool
        If True, exit trades when the running VWAP is hit; otherwise
        exit at the session close.
    slippage_bps : int
        Base slippage applied to entry and exit prices (in basis points).
    apply_ssr : bool
        If True, apply the short sale restriction when a security
        declines 10 % below the previous close.  Once triggered the
        restriction remains for the rest of the day.  This is a
        simplified approximation; persistence into the next day must
        be handled by the caller.
    apply_luld : bool
        If True, enforce limit up–limit down price bands.  Trades are
        blocked when the price moves outside the band for a number of
        consecutive bars and paused for a fixed number of bars.
    luld_pct : float
        Percentage width of the LULD price band.  For example 0.10
        corresponds to ±10 % around the five‑minute reference price.
    luld_state_bars : int
        Number of consecutive bars at a band edge required to trigger a
        limit state (approx. 15 s if using one‑minute bars).  During
        the limit state trades are blocked and a trading pause is
        initiated.
    luld_pause_bars : int
        Number of bars to pause trading after a limit state is
        triggered (approx. 5 minutes if using one‑minute bars).
    shock_prob : float
        Probability of encountering an additional slippage shock on
        entry or exit.
    shock_bps : int
        Extra slippage (in basis points) applied when a shock occurs.
    seed : int, optional
        Random seed for reproducibility of slippage shocks.

    Returns
    -------
    pnl : pd.Series
        Per‑bar P&L (per share).  Length ``len(df)`` minus one.
    equity : pd.Series
        Cumulative equity curve.
    trades : int
        Total number of executed trades.
    """
    d = df.copy()
    # align signals and pivot levels
    d['signal'] = signal.reindex(d.index).fillna(False).astype(bool)
    d['lh_level'] = lh_level.reindex(d.index)
    # running VWAP for each bar
    d['vwap'] = (d['close'] * d['volume']).cumsum() / (d['volume'].cumsum().replace(0, np.nan))
    # rolling five‑minute reference price for LULD (min_periods=1 to seed early bars)
    if apply_luld:
        d['ref_price'] = d['close'].rolling(window=5, min_periods=1).mean()
    pnl: list[float] = []
    eq = 0.0
    equity: list[float] = []
    trades = 0
    # random generator for slippage shocks
    rng = np.random.default_rng(seed)
    # random generator for borrow availability
    rng_borrow = np.random.default_rng(seed + 1234 if seed is not None else None)
    # Precompute SSR activation per bar: a simplified approach that triggers
    # once the low of the day drops below 90 % of the prior day's close.
    ssr_flag = np.zeros(len(d), dtype=bool)
    if apply_ssr:
        # map each bar to the previous day's close.  For the first day
        # this may be NaN; we forward fill afterwards.
        day = pd.to_datetime(d.index.date)
        last_close_by_day = d['close'].groupby(day).last()
        prev_close_map = last_close_by_day.shift(1)
        prev_closes = prev_close_map.reindex(day).reset_index(drop=True)
        prev_closes = prev_closes.reindex_like(d['close']).fillna(method='ffill')
        # running minimum of low to detect SSR trigger intra‑day
        running_min_low = d['low'].cummin()
        ssr_trigger = running_min_low <= prev_closes * 0.9
        # once triggered, SSR remains active for the rest of the day
        active = False
        for i in range(len(d)):
            if not active and ssr_trigger.iloc[i]:
                active = True
            ssr_flag[i] = active
    # LULD state tracking
    limit_counter = 0
    pause_counter = 0
    for i in range(1, len(d)):
        # update LULD limit state if enabled
        if apply_luld:
            if pause_counter > 0:
                pause_counter -= 1
            else:
                # check if price outside band
                ref = d['ref_price'].iloc[i]
                band_hi = ref * (1 + luld_pct)
                band_lo = ref * (1 - luld_pct)
                price = d['close'].iloc[i]
                if price >= band_hi or price <= band_lo:
                    limit_counter += 1
                    if limit_counter >= luld_state_bars:
                        pause_counter = luld_pause_bars
                        limit_counter = 0
                else:
                    limit_counter = 0
        # decide whether we can place a trade on this bar
        can_trade = True
        if apply_luld and pause_counter > 0:
            can_trade = False
        if apply_ssr and ssr_flag[i]:
            can_trade = False
        # When the prior bar has signalled, attempt a trade on this bar
        if d['signal'].iloc[i - 1] and can_trade:
            trades += 1
            # check borrow availability; if unavailable, skip trade
            if rng_borrow.random() > borrow_availability:
                pnl.append(0.0)
                equity.append(eq)
                continue
            # compute slippage at entry (base + possible shock)
            slip_in = slippage_bps + (shock_bps if (rng.random() < shock_prob) else 0)
            entry_price = d['open'].iloc[i] * (1 - slip_in / 1e4)
            pivot = d['lh_level'].iloc[i - 1]
            # fallback stop above previous bar high if pivot missing
            raw_stop = pivot if pd.notna(pivot) else d['high'].iloc[i - 1]
            stop_price = raw_stop * (1 + stop_bps / 1e4)
            # choose target: vwap at bar i or session close
            target_price = d['vwap'].iloc[i] if target_to_vwap else d['close'].iloc[-1]
            # Determine exit price with slippage on exit
            slip_out = slippage_bps + (shock_bps if (rng.random() < shock_prob) else 0)
            exit_price = None
            # evaluate stops/targets in order
            if d['high'].iloc[i] >= stop_price:
                exit_price = stop_price * (1 + slip_out / 1e4)
            elif d['low'].iloc[i] <= target_price:
                exit_price = target_price * (1 + slip_out / 1e4)
            else:
                exit_price = d['close'].iloc[i] * (1 + slip_out / 1e4)
            trade_pnl = entry_price - exit_price
            # subtract borrow cost (basis points)
            if borrow_rate_bps > 0.0:
                borrow_cost = borrow_rate_bps / 1e4 * entry_price
                trade_pnl -= borrow_cost
            pnl.append(trade_pnl)
            eq += trade_pnl
        else:
            # either no signal or trade blocked; record zero pnl
            pnl.append(0.0)
            eq += 0.0
        equity.append(eq)
    # align index lengths (pnl/emits len-1, equity len-1)
    return (
        pd.Series(pnl, index=d.index[: len(pnl)]),
        pd.Series(equity, index=d.index[: len(equity)]),
        trades,
    )
