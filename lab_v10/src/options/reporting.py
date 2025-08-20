
import json
import pandas as pd
from ..common.metrics import summarize

def write_reports(outdir: str, fold_stats: list, equity: pd.Series, pnl: pd.Series, trials: int=1, costs: pd.Series=None, baselines: dict=None):
    overall = summarize(pnl, equity, label="overall", trials=trials)
    summary = {
        "folds": fold_stats,
        "overall": overall,
    }
    if baselines:
        summary["baselines"] = baselines
    if costs is not None:
        summary["costs_sum"] = float(costs.sum())
    import os
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pnl.to_csv(f"{outdir}/pnl.csv")
    equity.to_csv(f"{outdir}/equity_curve.csv")
    if costs is not None:
        costs.to_csv(f"{outdir}/costs.csv")
