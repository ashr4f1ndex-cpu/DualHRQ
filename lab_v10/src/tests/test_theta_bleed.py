
from ..options.options import simulate_atm_straddle_roundtrip


def test_theta_bleed_flat():
    # If S and IV constant, long straddle should decay
    S0 = 400.0
    iv = 0.20
    pnl = simulate_atm_straddle_roundtrip(S0, iv, S0, iv, dte_days=30, hold_days=5, r_annual=0.01, q_annual=0.0, bid_ask_bps=0.0, commission_per_contract=0.0, contracts=1)
    assert pnl <= 0.0 + 1e-6
