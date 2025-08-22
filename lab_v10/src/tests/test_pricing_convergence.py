

from ..options.pricing import bsm_call_put, crr_american


def test_binomial_converges_to_bsm_when_q0_and_no_early_exercise_value():
    S = 100
    K = 100
    T = 0.25
    r = 0.01
    q = 0.0
    sigma = 0.2
    call_bsm, put_bsm = bsm_call_put(S, K, T, r, q, sigma)
    call_bin = crr_american(S, K, T, r, sigma, steps=200, is_call=True, dividends=None)
    # American call on non-dividend stock ~ European; early exercise not optimal
    assert abs(call_bsm - call_bin) < 1.0  # loose bound to avoid numeric strictness
