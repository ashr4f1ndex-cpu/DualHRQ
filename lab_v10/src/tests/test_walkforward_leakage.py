
import pandas as pd
from ..options.walkforward import calendar_walkforward

def test_embargo_strict():
    idx = pd.date_range("2020-01-01","2024-12-31",freq="B")
    windows = list(calendar_walkforward(idx, train_years=2, test_years=1, embargo_days=5))
    for _tr_s, tr_e, te_s, _te_e in windows:
        assert tr_e < te_s  # embargo moves test start forward
