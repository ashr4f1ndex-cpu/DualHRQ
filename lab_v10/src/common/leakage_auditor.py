
from __future__ import annotations

import pandas as pd


def assert_no_leakage(df: pd.DataFrame, time_col: str, fold_col: str,
                      label_horizon: pd.Timedelta,
                      purge: pd.Timedelta, embargo: pd.Timedelta) -> None:
    d = df.sort_values(time_col).copy()
    folds = d[fold_col].unique()
    for k in folds:
        val = d[d[fold_col] == k]
        trn = d[d[fold_col] != k]
        if val.empty or trn.empty:
            continue
        v_start, v_end = val[time_col].min(), val[time_col].max()
        ex_start = v_start - purge
        ex_end   = v_end + embargo
        viol = trn[(trn[time_col] >= ex_start) & (trn[time_col] <= ex_end)]
        if not viol.empty:
            raise AssertionError(f"Leakage: training within purge/embargo around fold {k}")
        trn_label_end = trn[time_col] + label_horizon
        if ((trn_label_end >= v_start) & (trn[time_col] <= v_end)).any():
            raise AssertionError(f"Leakage: training label window crosses validation period for fold {k}")
