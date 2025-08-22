
import sys

import pandas as pd

from lab_v10.src.common.leakage_auditor import assert_no_leakage

t0 = pd.Timestamp("2025-01-01 09:30")
idx = [t0 + pd.Timedelta(minutes=i) for i in range(60)]
df = pd.DataFrame({"t": idx})
df["fold"] = [i % 3 for i in range(60)]
try:
    assert_no_leakage(df, "t", "fold", pd.Timedelta(minutes=1),
                      purge=pd.Timedelta(minutes=2), embargo=pd.Timedelta(minutes=2))
    print("LEAKAGE_SMOKE=OK")
    sys.exit(0)
except AssertionError as e:
    print("LEAKAGE_SMOKE=FAIL", e)
    sys.exit(1)
