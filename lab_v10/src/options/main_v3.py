
import argparse

import pandas as pd

from ..common.metrics import summarize
from .backtest import StraddleParams, simulate_straddle_pnl
from .baselines.strategies import always_long_signal, iv_slope_long_signal
from .data.options_chain_adapter import load_chain_series
from .data.pipeline_adapter import build_features_and_labels
from .data.vendor_loaders import (
    load_cboe_datashop,
    load_databento_csv,
    load_dolthub_options_csv,
    load_marketdata_csv,
    load_optionmetrics_ivydb,
)
from .hrm_adapter import HRMModel
from .reporting import write_reports
from .walkforward import calendar_walkforward


def run(start: str, end: str, outdir: str="reports/options", trials: int=5, style: str="european", vendor: str=None, csv: str=None, underlying_csv: str=None, symbol: str=None):
    if vendor == 'cboe':
        if not csv:
            raise SystemExit("--csv required for vendor=cboe")
        S, iv_now, iv_future, expiry = load_cboe_datashop(csv)
    elif vendor == 'ivydb':
        if not csv or not underlying_csv:
            raise SystemExit("--csv and --underlying_csv required for vendor=ivydb")
        S, iv_now, iv_future, expiry = load_optionmetrics_ivydb(csv, underlying_csv)
    elif vendor == 'dolthub':
        if not csv:
            raise SystemExit("--csv required for vendor=dolthub (CSV export of DoltHub query)")
        S, iv_now, iv_future, expiry = load_dolthub_options_csv(csv, symbol=symbol)
    elif vendor == 'marketdata':
        if not csv:
            raise SystemExit("--csv required for vendor=marketdata (CSV export)")
        S, iv_now, iv_future, expiry = load_marketdata_csv(csv, symbol=symbol)
    elif vendor == 'databento':
        if not csv:
            raise SystemExit("--csv required for vendor=databento (CSV export)")
        S, iv_now, iv_future, expiry = load_databento_csv(csv, symbol=symbol)
    else:
        S, iv_now, iv_future, expiry = load_chain_series(None, start, end)
    features, labels = build_features_and_labels(S, iv_now, horizon=5)
    idx = (features.index.intersection(labels.index).intersection(S.index)
           .intersection(iv_now.index).intersection(iv_future.index)
           .intersection(expiry.index))
    S, iv_now, iv_future, expiry = S.loc[idx], iv_now.loc[idx], iv_future.loc[idx], expiry.loc[idx]
    features, labels = features.loc[idx], labels.loc[idx]

    model = HRMModel(reg_lambda=1e-2)
    all_equity = pd.Series(0.0, index=idx)
    all_pnl = pd.Series(0.0, index=idx)
    all_costs = pd.Series(0.0, index=idx)
    fold_stats = []

    params = StraddleParams(style=style)

    for (tr_s, tr_e, te_s, te_e) in calendar_walkforward(idx, train_years=4, test_years=1, embargo_days=5):
        tr_mask = (idx >= tr_s) & (idx <= tr_e)
        te_mask = (idx >= te_s) & (idx <= te_e)
        X_tr, y_tr = features[tr_mask], labels[tr_mask]
        X_te = features[te_mask]

        if len(X_tr) < 50 or len(X_te) < 5:
            continue
        model.fit(X_tr, y_tr)
        signal_hrm = pd.Series(model.predict(X_te), index=X_te.index)

        # Baselines on the same test window
        sig_always = always_long_signal(X_te.index)
        sig_ivslp = iv_slope_long_signal(iv_now.loc[X_te.index])

        # Backtests
        for name, sig in [
            ("HRM", signal_hrm), ("AlwaysLong", sig_always), ("IVSlope", sig_ivslp)
        ]:
            bt = simulate_straddle_pnl(
                S[te_mask], iv_now[te_mask], iv_future[te_mask],
                expiry[te_mask], sig, params
            )
            pnl, equity, costs = bt["pnl"], bt["equity"], bt["costs"]
            if name == "HRM":
                all_pnl.loc[pnl.index] += pnl
                # rebuild equity cumulatively on global index
                all_equity.loc[equity.index] = (
                    all_equity.loc[equity.index].add(pnl, fill_value=0).cumsum()
                )
                all_costs.loc[costs.index] += costs
                fold_stats.append(summarize(pnl, equity, label=f"{te_s.date()}_{te_e.date()}", trials=trials))

    # Baseline summaries over full period (rebuilt by running once more on full idx test mask concatenation is non-trivial here)
    # For brevity, compute on overlapping HRM test indices where pnl != 0
    test_idx = all_pnl.index[all_pnl != 0]
    baselines = {}
    if len(test_idx) > 0:
        te_mask_all = idx.isin(test_idx)
        for name, sig in [("AlwaysLong", always_long_signal(idx[te_mask_all])), ("IVSlope", iv_slope_long_signal(iv_now.loc[idx[te_mask_all]]))]:
            bt = simulate_straddle_pnl(S[te_mask_all], iv_now[te_mask_all], iv_future[te_mask_all], expiry[te_mask_all], sig, params)
            baselines[name] = summarize(bt["pnl"], bt["equity"], label=name, trials=trials)

    write_reports(outdir, fold_stats, all_equity.fillna(method='ffill').fillna(0), all_pnl.fillna(0), trials=trials, costs=all_costs.fillna(0), baselines=baselines)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2016-01-01")
    ap.add_argument("--end", type=str, default="2024-12-31")
    ap.add_argument("--outdir", type=str, default="reports/options")
    ap.add_argument("--vendor", type=str, default=None, choices=[None, 'cboe', 'ivydb', 'dolthub', 'marketdata', 'databento'])
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--underlying_csv", type=str, default=None)
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--style", type=str, default="european")  # 'european' (SPX) or 'american' (SPY)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()
    run(args.start, args.end, args.outdir, trials=args.trials, style=args.style)
