# ADR-001: Dynamic Conditioning System for HRM Generalization

**Status:** Proposed  
**Date:** 2025-08-24  
**Deciders:** Core Engineering Team  

## Problem & Goals

**Current Problem:**
- HRM model has static puzzle_id conditioning enabling memorization rather than generalization
- 46.68M parameters exceed 26.5M-27.5M budget by 74%
- Missing pattern recognition for regime-based learning
- No validation framework for leakage detection

**Goals:**
1. **Generalization**: Replace static puzzle_id with dynamic regime-based conditioning
2. **Auditability**: All conditioning paths explainable and testable  
3. **Parameter Budget**: Strict 26.5M-27.5M constraint with CI enforcement
4. **Strict Simulation**: Zero SSR/LULD violations in regulatory compliance

## Alternatives Considered

### A) FiLM + Regime Clustering
- **Architecture**: Market Features → Regime Classifier → FiLM Parameters → HRM
- **Pros**: Clean separation, interpretable, low latency (~20ms), efficient (~0.2M params)
- **Cons**: Potential regime collapse, limited expressiveness, manual regime definition

### B) FiLM + RAG (Retrieval-Augmented Generation)  
- **Architecture**: Market Features → Pattern Retrieval → Context Fusion → FiLM → HRM
- **Pros**: Highly expressive, natural pattern discovery, handles rare regimes
- **Cons**: Higher latency (~80ms), complex failure modes, maintenance overhead

### C) Hybrid (FiLM + Regime + RAG)
- **Architecture**: Market Features → {Regime + Pattern Retrieval} → Fusion → FiLM → HRM
- **Pros**: Best of both, fail-open capability, comprehensive coverage
- **Cons**: Higher complexity, more parameters (~0.3M), orchestration complexity

## Final Choice: Hybrid (Option C)

**Rationale:**
- **Generalization**: Combines interpretable regimes with discovered patterns per JSON Doc A
- **Robustness**: Fail-open to regime-only meets latency SLO requirements from JSON Doc B
- **Parameter Budget**: 0.3M conditioning + 26.2M HRM = 26.5M total
- **Auditability**: Regime states provide baseline interpretability for compliance

## Architecture

### Component Structure
```
src/models/conditioning/
├── __init__.py              # Unified conditioning interface
├── film.py                  # FiLM layer (≤0.1M params)  
├── regime.py                # Regime classifier (≤0.1M params)
└── rag.py                   # Pattern retrieval (≤0.1M params)

src/models/
├── pattern_library.py       # Pattern storage & retrieval
├── hrm_integration.py       # HRM adapter layer  
└── adaptive_budget.py       # Parameter budget management
```

### Data Flow
```
Market Data → Feature Engineering → {
  Regime Features: TSRV, BPV, Amihud, SSR/LULD state
  Pattern Features: Technical indicators, microstructure
} → {
  Regime Classifier → regime_logits
  Pattern Retrieval → pattern_context  
} → Context Fusion → FiLM Parameters → HRM L-module
```

### Parameter Budget Strategy
- **HRM Core**: 26.2M (H=384d/4L, L=512d/4L)
- **Conditioning**: ≤0.3M total
- **Target**: 26.5M ≤ total ≤ 27.5M
- **CI Gate**: `tools/param_count.py` fails build if exceeded

## Validation Framework

### Leakage Prevention  
- **MI Tests**: `MI(features, puzzle_id) < 0.1 bits` in `tests/conditioning/test_leakage_mi.py`
- **Shuffle Tests**: `>50% performance drop` in `tests/conditioning/test_shuffle_codes.py`
- **Static ID Detection**: `tools/check_no_static_ids.py` with allowlist

### Time Series Validation
- **CPCV**: 2-week purge + 1-week embargo
- **Determinism**: Exact reproduction in `tests/conditioning/test_determinism.py`
- **Strict-Sim**: Zero SSR/LULD violations in `tests/sim/test_ssr_luld_replay.py`

### Statistical Validity
- **Reality Check**: Bootstrap p-values
- **SPA Test**: Superior Predictive Ability p < 0.10
- **DSR**: Data Snooping-Robust reporting

## Latency SLO + Fail-Open

**Primary Path**: Regime + RAG → <100ms  
**Fail-Open Path**: Regime-only → <80ms when RAG unavailable  
**Implementation**: Circuit breaker pattern with automatic fallback

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Parameter budget exceeded | Build fails | CI gate `test_param_gate.py` |
| Regime collapse | Poor generalization | Multi-scale detection + validation |
| RAG latency overrun | SLO violation | Fail-open to regime-only |
| Data leakage | Invalid results | Automated MI/shuffle in CI |
| SSR/LULD violations | Regulatory breach | Strict simulation + replay |

## Rollout Plan

### Phase 0: Baseline (Weeks 1-2)
- Add parameter counter + CI gate
- Prove absence of static IDs  
- Determinism smoke tests
- Strict-sim replay foundation

### Phase 1: Conditioning + Budget (Weeks 3-8)  
- FiLM + Regime + RAG implementation
- Parameter budget compliance
- Feature flag controls
- Basic integration tests

### Phase 2: Pattern + HRM (Weeks 9-14)
- Advanced pattern detection
- HRM integration layer
- Walk-forward validation
- Performance optimization

### Phase 3: Enhancements (Weeks 15-20)
- Complete validation suite
- Statistical testing framework
- Ablation studies
- Performance certification

### Phase 4: Agents + Infra (Weeks 21-26)
- Agent orchestration system
- Production infrastructure
- Monitoring & alerting
- Deployment automation

## Feature Flags & Promotion Criteria

**Flags:**
```yaml
conditioning.enable_regime: false      # Regime classifier
conditioning.enable_rag: false        # RAG system
conditioning.enable_hybrid: false     # Full hybrid mode
conditioning.fail_open: true          # Always enable fail-open
```

**Promotion Gates:**
- Phase 0 → 1: Parameter/ID/determinism gates pass
- Phase 1 → 2: Conditioning system functional, budget compliant
- Phase 2 → 3: Pattern library operational, HRM integrated  
- Phase 3 → 4: Statistical validation complete, performance certified
- Phase 4 → Prod: Agent system deployed, monitoring operational

## CI Gates Mapped to Test Files

**Required Gates (Fail Build):**
```bash
tests/conditioning/test_param_gate.py          # Parameter budget: 26.5M ≤ params ≤ 27.5M
tests/conditioning/test_leakage_mi.py          # MI(features, puzzle_id) < 0.1 bits  
tests/conditioning/test_shuffle_codes.py       # Shuffle performance drop >50%
tests/conditioning/test_determinism.py         # Exact reproducibility
tests/sim/test_ssr_luld_replay.py             # Zero regulatory violations
```

**Tools:**
```bash
tools/param_count.py                           # Model parameter counting
tools/check_no_static_ids.py                   # Static ID detection
```