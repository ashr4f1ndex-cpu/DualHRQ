
import pandas as pd

from lab_v10.src.common.leakage_auditor import assert_no_leakage


def _toy(n=60, folds=3):
    t0 = pd.Timestamp("2025-01-01 09:30")
    idx = [t0 + pd.Timedelta(minutes=i) for i in range(n)]
    df = pd.DataFrame({"t": idx})
    df["fold"] = [i % folds for i in range(n)]
    return df

def test_leakage_guard_detects_when_disabled():
    df = _toy()
    try:
        assert_no_leakage(df, "t", "fold", pd.Timedelta(minutes=5),
                          purge=pd.Timedelta(seconds=0), embargo=pd.Timedelta(seconds=0))
    except AssertionError:
        return
    assert False, "Leakage should be detected when purge/embargo are zero"

def test_leakage_guard_passes_with_purge_embargo():
    df = _toy()
    assert_no_leakage(df, "t", "fold", pd.Timedelta(minutes=1),
                      purge=pd.Timedelta(minutes=2), embargo=pd.Timedelta(minutes=2))
