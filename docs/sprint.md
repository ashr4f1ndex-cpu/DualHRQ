# 10‑Day Sprint Plan

This sprint plan outlines the key tasks required to bring the dual‑book
trading lab from a skeleton implementation to a research‑grade
deliverable.  Tasks are sequenced to minimise integration risk and
include clear acceptance criteria.

## Day 1: Kickoff and Data Pipeline

* **Define scope:** Review the HRM specification and prioritise
  outstanding features (intraday features, SSR/LULD fidelity, HRM
  integration, documentation).  Assign owners to each feature area.
* **Set up data pipeline:** Verify access to historical intraday and
  options data, ensuring it can be loaded without leakage.  Prepare
  derived features tables.
* **Acceptance:** Data loads without errors; CPCV folds can be
  enumerated; baseline tests pass.

## Day 2–3: Feature Engineering

* **Intraday features:** Implement vectorised, leakage‑safe intraday
  features such as realised volatility (volx), parabolic spike
  detectors, time‑to‑VWAP and additional pivot logic.  Write unit
  tests to validate behaviour on synthetic data.
* **Options features:** Extend the options feature module to compute
  realised volatility windows, implied vol skew, curvature and, if
  necessary, Black–Scholes greeks and regime indicators.  Ensure
  functions avoid look‑ahead by shifting windows.
* **Acceptance:** All new features have docstrings, tests pass and
  coverage includes edge cases.

## Day 4–5: HRM Wiring and Adapter

* **HRM core:** Review the HRM network and trainer.  Integrate
  optional heteroscedastic head‑A and binary head‑B outputs.  Expose
  configuration knobs for ACT and cross‑attention in the adapter.
* **Adapter integration:** Modify `hrm_adapter.py` to support the
  complete HRM and to toggle between simplified and full models via
  configuration.  Ensure tokenisation and scaling are consistent.
* **Acceptance:** The HRM fits on a small synthetic dataset and
  produces outputs with the expected shapes.  Parameter count is
  within ±0.5 M of 27 M.

## Day 6: Backtesting Integration

* **Intraday simulator:** Replace the simplified simulator with the
  strict version that enforces SSR persistence, uptick rules and
  LULD band doubling【515680207816213†L213-L218】【457522738594840†L70-L71】.  Add random slippage shocks
  and borrow‑rate costs.  Write tests to verify SSR persists into
  the next day and that trades are blocked under the uptick rule.
* **Portfolio logic:** Integrate differentiable portfolio loss
  surrogates into the training loop.  Confirm Sharpe/Sortino/CVaR
  objectives produce gradients and align with CVaR backtest metrics.
* **Acceptance:** Backtests run end‑to‑end without errors and key
  metrics (returns, number of trades) are sensible on toy data.

## Day 7: Evaluation and MLOps

* **Cross‑validation:** Implement combinatorial purged CV splitting
  (`walkforward_cpcv.py`) and evaluate the HRM on held‑out folds.
* **Metrics:** Compute deflated Sharpe, Sortino and CVaR across CV
  folds; compare against ridge baselines; track uncertainty estimates.
* **Reproducibility:** Finalise CI workflow with determinism gates and
  leakage tests.  Set up experiment tracking (e.g. via MLflow or
  Weights & Biases) and ensure results are logged.
* **Acceptance:** Evaluation script runs from the command line and
  outputs reproducible metrics.  CI passes on the main branch.

## Day 8–9: Documentation and Sources

* **Risk register:** Expand the risk register with detection
  methods, mitigation playbooks and fallback strategies; cite
  authoritative sources for SSR and LULD rules【515680207816213†L213-L218】【457522738594840†L70-L71】.
* **Latency/memory:** Estimate per‑module parameter counts, memory
  footprints and latency budgets.  Document throughput and mixed
  precision considerations.
* **Curated sources:** Compile a curated list of papers, regulatory
  documents and industry articles covering options backtesting,
  microstructure regulations, portfolio optimisation and CV
  techniques.  Note licensing and relevance for each source.
* **Acceptance:** Documentation is comprehensive and readable; all
  references include tether IDs or links; the coverage report marks
  docs as complete.

## Day 10: Hardening and Release

* **Integration:** Replace the HRM ridge adapter in the main lab with
  the complete HRM, exposing configuration switches in `main_v3.py`
  and `main_portfolio.py`.  Verify integration through smoke tests.
* **Final testing:** Run the full suite of unit tests, backtests and
  performance benchmarks.  Address any regressions in latency,
  determinism or accuracy.
* **Packaging:** Update the change manifest and spec coverage CSV.
  Produce a final zip archive with all new and modified files and
  ensure instructions for reproduction are clear.

* **Acceptance:** All planned features are implemented; tests pass;
  documentation is complete; and stakeholders sign off on the
  release.