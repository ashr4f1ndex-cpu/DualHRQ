
import pandas as pd
import numpy as np
from ..common.metrics import deflated_sharpe, sharpe

def test_deflated_sharpe_not_exceed_sample_sharpe():
    """The deflated Sharpe ratio should not exceed the sample Sharpe ratio on IID data."""
    np.random.seed(0)
    ret = pd.Series(np.random.normal(0.001, 0.01, 200))
    s = sharpe(ret)
    d = deflated_sharpe(ret, trials=10)
    assert d <= s + 1e-6
