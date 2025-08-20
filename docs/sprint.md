# 10-Day Sprint Plan

## Days 1-2: Data Infrastructure
- [ ] CPCV pipeline with purge/embargo
- [ ] Leakage smoke tests
- [ ] Data alignment validation
- [ ] Corporate actions handling

## Days 3-5: HRM Implementation  
- [ ] 27M parameter architecture
- [ ] ACT halting mechanism
- [ ] Multi-task heads (vol-gap + triggers)
- [ ] Parameter budget validation

## Days 6-7: Backtesting Engine
- [ ] SSR/LULD constraint enforcement
- [ ] Options straddle simulator  
- [ ] Intraday short mechanics
- [ ] Portfolio combiner

## Days 8-9: Hyperparameters & Training
- [ ] Loss aggregation tuning
- [ ] DEQ-style training loop
- [ ] Uncertainty weighting
- [ ] Model selection criteria

## Day 10: Deployment & Gates
- [ ] Docker containerization
- [ ] CI/CD pipeline validation
- [ ] Acceptance gate testing
- [ ] Production readiness review

## Acceptance Gates per Task
- Data: Zero leakage violations
- HRM: Param count ∈ [26.5M, 27.5M]  
- Backtest: SSR/LULD compliance
- Training: Deflated Sharpe > 1.0
- Deploy: <100ms inference latency
