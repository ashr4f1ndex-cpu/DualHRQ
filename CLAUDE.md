# DualHRQ • Claude Guide
## Rules
- No import-time I/O; network only inside explicit functions/tests.
- CI: actions/upload-artifact@v4; actions/cache@v4 with cache-hit skip.
- CPCV with purge/embargo; leakage smoke tests required.
- HRM params ∈ [26.5M, 27.5M].
- Intraday sim: SSR next-day persistence; uptick-only; LULD last-25-min doubling.
## Workflow
- ≤25 files per PR; conventional commits.
- Merge gates: CI green, CPCV OK, param budget OK, reflection-auditor OK.