
# Dual-Book Trading Lab (Options + Intraday)

An end-to-end research framework for **Systematic Options (ATM straddles)** and **Intraday Momentum-Reversal (backside shorts)**, with realistic option PnL, walk-forward validation, and portfolio orchestration.

## Quick Start

```bash
# 1) Run the options walk-forward lab
python -m src.options.main_v3 --start 2016-01-01 --end 2024-12-31 --outdir reports/options

# 2) Combine into a portfolio (options + intraday placeholder)
python -m src.portfolio.main_portfolio --start 2016-01-01 --end 2024-12-31 --alloc_options 0.7
```

Outputs:
- `reports/options/summary.json`, `pnl.csv`, `equity_curve.csv`
- `reports/portfolio_pnl.csv`, `reports/portfolio_equity.csv`

## Where to plug real data

- **Options chain IV & prices**: `src/options/data/options_chain_adapter.py`
  - Replace the stub with a loader for Cboe DataShop / OptionMetrics (IvyDB).
  - Provide `S` (underlying close), `iv_entry` (mid IV at entry DTE), `iv_exit` (same contract IV at exit).

- **HRM features**: `src/options/data/pipeline_adapter.py`
  - Replace with your production HRM feature/label pipeline.

- **Intraday scanner/backtest**: `src/intraday/*` 
  - Feed 1-minute OHLCV; `scanner.detect_parabolic_backside` returns boolean triggers.
  - `backtest_intraday.simulate_intraday_backside_short` will mark trades using a simple VWAP target and stop.

## Why this is credible

- **Walk-forward with embargo** to reduce leakage across folds.
- **Black–Scholes–Merton** pricing with dividend yield (q) for theta/gamma-aware straddle PnL.
- **Dynamic spreads** widening in high IV regimes.
- **Risk hooks**: vol-target sizing and drawdown guard.
- **Sanity tests**: theta-bleed on flat paths; embargo window check.

> This repository is a research framework. Before deployment, plug in real options chain data, add more stringent execution models, and validate with additional baselines.

## Final Iteration Upgrades (v10)
The **v10** release implements the full deep‑research specification.  Key upgrades include:

* **HRM‑27M architecture** – dual‑scale transformer with 4‑layer H‑module and 6‑layer L‑module, rotary positional embeddings, FiLM conditioning, optional cross‑attention, and an **Adaptive Computation Time (ACT)** halting mechanism.  Dynamic halting reduces unnecessary computation while allowing deeper reasoning when needed.

* **Deflated Sharpe Ratio** – reported via the Bailey/López de Prado correction【277152836813280†L142-L150】, alongside standard Sharpe, Sortino, CVaR and EVaR measures.  The model now includes differentiable surrogates for Sharpe, Sortino, CVaR and EVaR, enabling risk‑aware finetuning.

* **Intraday microstructure realism** – Rule 201 SSR gate persists from a 10 % drop through the remainder of the day and the following day【838491366880600†L220-L289】.  Limit Up–Limit Down (LULD) price bands are enforced with dynamic doubling in the final 25 minutes of trading【733935163568087†L33-L71】.  Borrow availability and borrow rates can be configured, and trades are blocked if borrow is unavailable.

* **Slippage shocks and dynamic spreads** – slippage shocks model rare liquidity events; spreads widen during high IV regimes and session opens.  Position sizing and turnover penalties remain differentiable via `diff_exec.py`.

* **Optional portfolio losses** – smooth Sharpe, Sortino, CVaR and EVaR objectives can be applied during training via the `portfolio.losses` config.  A differentiable execution surrogate converts raw signals into PnL and computes tail‑risk penalties.

* **Expanded multi‑task training** – uncertainty‑weighted multi‑task losses for daily volatility gap and intraday downside triggers.  ACT halting losses and ponder penalties discourage over‑thinking.

* **Improved MLOps** – pinned dependencies, deterministic seeding, faster CI and reproducible tests.  New tests verify ACT halting, cross‑attention, intraday gating and portfolio losses.

## Vendor data loaders
Use `--vendor` to parse real data:
- `--vendor cboe --csv path/to/cboe_datashop_eod.csv`
- `--vendor ivydb --csv path/to/Option_Price.csv --underlying_csv path/to/Underlying_Price.csv`

The loaders pick **ATM** contracts and the expiration closest to **30 calendar days** (configurable inside loaders). The **expiry** series is passed to pricing so time-to-expiry is **calendar-day** accurate.

## Dynamic portfolio allocation
Run the portfolio with dynamic allocation based on **rolling Sharpe** and a **drawdown gate**:
```bash
python -m src.portfolio.main_portfolio --start 2016-01-01 --end 2024-12-31 --dynamic
```
Outputs `reports/portfolio_weights.csv` with time-varying weights.

## Intraday strategy from video (final integration)
- `src/intraday/scanner_video.py` detects **parabolic stretch + lower-high breakdown** (VWAP/ATR/volume based).
- `src/intraday/backtest_intraday.py` simulates **short entries** at next bar open, **stops** above LH, **targets** VWAP, with SSR/slippage realism.
- `src/intraday/aggregate_daily.py` rolls intraday signals into **daily features** that your options **HRM** can consume.

## Running Book B and combining with options
```bash
# Portfolio with synthetic minute data (replace with your minute CSV via --minute_csv)
python -m src.portfolio.main_portfolio --start 2019-01-01 --end 2024-12-31 --dynamic
```
This creates:
- `reports/intraday_pnl_daily.csv` (Book B PnL)
- `reports/intraday_daily_features.csv` (features feed for Book A)
- `reports/portfolio_*` (combined equity and weights)

## Public/low-cost data integration (NEW in v5)
You can prototype with public data before buying anything:
- **DoltHub CSV**: export from the `options` database and run:
```bash
python -m src.options.main_v3 --vendor dolthub --csv data/dolthub_option_chain.csv --symbol TSLA --style american
```
- **MarketData.app CSV** (trial): export chain snapshots with IV:
```bash
python -m src.options.main_v3 --vendor marketdata --csv data/marketdata_chain.csv --symbol SPY --style american
```
- **Databento CSV**: download OPRA/quotes with IV, then:
```bash
python -m src.options.main_v3 --vendor databento --csv data/databento_opra.csv --symbol SPX --style european
```
All loaders pick the **ATM** contract per day and the **expiration closest to 30 calendar days**, yielding the (S, iv_entry, iv_exit, expiry) series used by the backtester. Replace the IV-exit proxy with same-contract IV if you have a contract-stable dataset.
