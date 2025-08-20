
import math
from typing import Optional

# ---- Black-Scholes-Merton with dividend yield q ----
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bsm_call_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float,float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        call = max(S - K, 0.0)
        put  = max(K - S, 0.0)
        return call, put
    d1 = (math.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    disc_r = math.exp(-r*T)
    disc_q = math.exp(-q*T)
    call = S*disc_q*_norm_cdf(d1) - K*disc_r*_norm_cdf(d2)
    put  = K*disc_r*_norm_cdf(-d2) - S*disc_q*_norm_cdf(-d1)
    return call, put

def bsm_straddle(S: float, T: float, r: float, q: float, sigma: float) -> float:
    call, put = bsm_call_put(S, S, T, r, q, sigma)
    return call + put

# ---- CRR Binomial for American options with discrete dividends ----
def crr_american(
    S: float, K: float, T: float, r: float, sigma: float, steps: int = 200,
    is_call: bool = True, dividends: Optional[list[tuple[float, float]]] = None
) -> float:
    """Cox-Ross-Rubinstein binomial tree with early exercise and discrete dividends.
    dividends: list of (t_div, amount) in years from now.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S-K,0.0) if is_call else max(K-S,0.0)
    dt = T/steps
    u = math.exp(sigma*math.sqrt(dt))
    d = 1.0/u
    disc = math.exp(-r*dt)
    p = (math.exp(r*dt) - d) / (u - d)
    # Adjust S for dividends by reducing price at ex-div times along the path approximately
    div_times = sorted(dividends or [])
    # Terminal payoffs
    values = [0.0]*(steps+1)
    for i in range(steps+1):
        # price at node
        ST = S*(u**i)*(d**(steps-i))
        # subtract dividends that occur before expiry (approx by PV at node time)
        for t_div, amt in div_times:
            if t_div <= T:
                ST = max(1e-8, ST - amt*math.exp(-r*(T - t_div)))
        payoff = max(ST - K, 0.0) if is_call else max(K - ST, 0.0)
        values[i] = payoff
    # Backward induction with early exercise
    for step in range(steps-1, -1, -1):
        for i in range(step+1):
            hold = disc*(p*values[i+1] + (1-p)*values[i])
            # underlying price at this node
            St = S*(u**i)*(d**(step-i))
            for t_div, amt in div_times:
                if t_div <= step*dt:
                    St = max(1e-8, St - amt*math.exp(-r*((step*dt) - t_div)))
            exercise = max(St - K, 0.0) if is_call else max(K - St, 0.0)
            values[i] = max(hold, exercise)
    return values[0]

def crr_straddle(S: float, T: float, r: float, sigma: float, steps:int=200, dividends:Optional[list[tuple[float,float]]]=None) -> float:
    call = crr_american(S, S, T, r, sigma, steps=steps, is_call=True, dividends=dividends)
    put  = crr_american(S, S, T, r, sigma, steps=steps, is_call=False, dividends=dividends)
    return call + put

# ---- Calendar-day DTE helper ----
def year_fraction_calendar_days(entry_ts, expiry_ts, basis: str = "ACT/365") -> float:
    seconds = (expiry_ts - entry_ts).total_seconds()
    if basis.upper() == "ACT/365":
        return seconds / (365.0*24*3600.0)
    elif basis.upper() == "ACT/360":
        return seconds / (360.0*24*3600.0)
    else:
        return seconds / (365.0*24*3600.0)
