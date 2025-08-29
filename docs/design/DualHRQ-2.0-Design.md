# DualHRQ 2.0 System Design

**Version:** 2.0  
**Date:** 2025-08-24  
**Status:** Design Phase  

## Executive Summary

DualHRQ 2.0 replaces static puzzle_id conditioning with a dynamic regime-based system that enables generalization while maintaining strict parameter budget (26.5M-27.5M) and regulatory compliance (SSR/LULD). The hybrid conditioning approach combines regime classification with pattern retrieval for robust performance across market conditions.

## System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Market Data   │───▶│  Feature Engine  │───▶│   Conditioning  │
│                 │    │                  │    │     System      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐           ▼
│  Pattern Store  │◀──▶│   HRM Network    │    ┌─────────────────┐
│                 │    │   (26.2M params) │◀───│ FiLM Modulation │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │  Trading Signals │
                        └──────────────────┘
```

### Core Components

#### 1. Feature Engineering Pipeline
**Location**: `src/features/`

**Regime Features:**
- **TSRV** (Time-Scaled Realized Volatility): Multi-scale volatility estimation
- **BPV** (Bipower Variation): Jump-robust volatility measure
- **Amihud Illiquidity**: Price impact per unit volume
- **SSR/LULD State**: Regulatory circuit breaker status

**Pattern Features:**
- Technical indicators (momentum, mean reversion)
- Microstructure signals (order flow, tick dynamics)
- Cross-sectional rankings (NO absolute identifiers)

#### 2. Conditioning System
**Location**: `src/models/conditioning/`

**Components:**
```python
class ConditioningSystem:
    regime_classifier: RegimeClassifier     # ≤0.1M params
    pattern_retrieval: PatternRAG           # ≤0.1M params  
    film_modulation: FiLMLayer              # ≤0.1M params
    fusion_layer: ContextFusion             # Minimal params
```

**Data Flow:**
```
Market Features → {
    Regime Classification → regime_logits [B, R]
    Pattern Retrieval → pattern_context [B, P, D]
} → Context Fusion → FiLM Parameters {gamma, beta} → HRM L-module
```

#### 3. HRM Integration
**Location**: `src/models/hrm_integration.py`

**Parameter Allocation:**
- **H-Module**: 384d × 4L = ~8.2M parameters
- **L-Module**: 512d × 4L = ~17.8M parameters  
- **Heads**: ~0.2M parameters
- **Total HRM**: ~26.2M parameters
- **Conditioning**: ≤0.3M parameters
- **Grand Total**: ≤26.5M parameters

#### 4. Pattern Library
**Location**: `src/models/pattern_library.py`

**Storage Architecture:**
```
PatternLibrary:
├── pattern_detector: MultiScaleDetector
├── pattern_codebook: VectorQuantization
├── pattern_index: FAISS/ANN Search
└── pattern_lifecycle: ExpirationManager
```

**Operations:**
- **Detection**: Real-time pattern identification
- **Storage**: Efficient vector storage with metadata
- **Retrieval**: Sub-80ms pattern lookup with similarity search
- **Lifecycle**: Automated pattern expiration and quality scoring

#### 5. Validation Framework
**Location**: `src/validation/`

**Time Series Validation:**
```python
@dataclass
class CPCVConfig:
    n_splits: int = 5
    purge_days: int = 14      # 2-week purge
    embargo_days: int = 7     # 1-week embargo
    test_ratio: float = 0.2
```

**Statistical Tests:**
- **Reality Check**: Bootstrap significance testing
- **SPA**: Superior Predictive Ability vs baseline
- **DSR**: Data Snooping-Robust metrics with multiple testing correction

## Data Flows & Integration

### Training Pipeline
```
Historical Data → Feature Engineering → {
    Regime Labeling (unsupervised clustering)
    Pattern Discovery (autoencoder + clustering)
} → Model Training → Validation → Model Registry
```

### Inference Pipeline  
```
Real-time Data → Feature Computation → {
    Regime Classification (20ms)
    Pattern Retrieval (60ms, fail-open to 0ms)
} → Context Fusion (5ms) → HRM Forward (10ms) → Signals (5ms)
```

### Replay & Strict-Sim Pipeline
```
Historical Events → SSR/LULD Rule Engine → {
    Circuit Breaker State Updates
    Trading Restriction Enforcement  
    Violation Detection & Reporting
} → Compliance Validation
```

## Parameter Budget Management

### Budget Allocation
| Component | Target Size | Max Size | Enforcement |
|-----------|-------------|----------|-------------|
| H-Module | 8.0M | 8.5M | CI gate |
| L-Module | 17.5M | 18.0M | CI gate |
| Heads | 0.2M | 0.3M | CI gate |
| Conditioning | 0.3M | 0.5M | CI gate |
| **Total** | **26.0M** | **27.3M** | **CI gate** |

### Enforcement Strategy
```bash
# Automated parameter counting
tools/param_count.py --config config/hrm_27m.yaml --strict

# CI integration  
pytest tests/conditioning/test_param_gate.py --fail-fast
```

## Latency Requirements & Fail-Open

### SLO Requirements
- **Primary Path**: <100ms end-to-end
- **Fail-Open Path**: <80ms when RAG disabled
- **Pattern Retrieval**: <60ms or fail-open
- **Circuit Breaker**: <10ms response to disable RAG

### Fail-Open Implementation
```python
class CircuitBreaker:
    def __call__(self, timeout_ms: int = 60):
        try:
            return self.rag_system.retrieve(timeout=timeout_ms)
        except TimeoutException:
            logger.warning("RAG timeout, failing open to regime-only")
            return self.regime_only_fallback()
```

## Reproducibility & Determinism

### Seed Control Strategy
```python
@dataclass  
class ReproducibilityConfig:
    torch_seed: int = 42
    numpy_seed: int = 42
    python_seed: int = 42
    cudnn_deterministic: bool = True
    data_version: str = "v2.1.0"
    model_checkpoint: str = "hrm-2.0-baseline"
```

### Version Control
- **Data Snapshots**: Immutable versioned datasets
- **Model Checkpoints**: Reproducible model states  
- **Config Versioning**: All hyperparameters tracked
- **Environment Pinning**: Exact dependency versions

## Security & Compliance

### Data Protection
- **No PII**: All features anonymized and aggregated
- **Access Control**: Role-based access to sensitive components
- **Audit Logging**: All model decisions logged for compliance

### Regulatory Compliance
- **SSR Rules**: 10% decline triggers with uptick-only execution
- **LULD Rules**: 5% circuit breakers with last-25min band doubling  
- **Immutable Rule Tables**: SSR/LULD logic cannot be modified in production
- **Compliance Reporting**: Daily violation reports with zero tolerance

## Monitoring & Observability

### Key Metrics
```yaml
Performance:
  - latency_p50, latency_p95, latency_p99
  - throughput_rps
  - error_rate_percentage

Model Health:  
  - parameter_count_daily
  - conditioning_utilization
  - regime_stability_score
  - pattern_library_size

Business:
  - sharpe_ratio_daily
  - max_drawdown_daily  
  - trading_volume_daily
  - pnl_attribution
```

### Alerting Strategy
- **Parameter Budget**: Alert if >27.0M, page if >27.5M
- **Latency SLO**: Alert if p95 >80ms, page if p95 >100ms  
- **Data Leakage**: Page immediately if MI > 0.1 bits
- **Regulatory**: Page immediately on any SSR/LULD violation

## Deployment Architecture

### Environment Strategy
```
Development → Staging → Canary → Production
     ↓           ↓         ↓         ↓
   Feature    1-month   5% traffic  Full rollout
    flags     replay    monitoring   monitoring
```

### Infrastructure Requirements
- **Compute**: 8 GPU instances (inference), 32 CPU cores (features)
- **Storage**: 500GB pattern library, 1TB historical data  
- **Network**: <10ms latency to exchanges, 1Gbps bandwidth
- **Monitoring**: Prometheus + Grafana, ELK stack, Jaeger tracing

### Rollback Strategy
- **Immediate**: Circuit breaker disables new system in <30s
- **Gradual**: Feature flags reduce traffic to new system  
- **Full**: Automated rollback to previous stable version

## Testing Strategy

### Test Categories
1. **Unit Tests**: Component-level functionality
2. **Integration Tests**: End-to-end pipeline validation
3. **Performance Tests**: Latency and throughput validation  
4. **Regression Tests**: Model performance vs baseline
5. **Compliance Tests**: Regulatory rule validation

### Test-Driven Development
```python
# Example test-first approach
def test_conditioning_parameter_budget():
    model = ConditioningSystem(config)
    param_count = count_parameters(model)
    assert param_count <= 300_000  # 0.3M parameter limit

def test_regime_classification_latency():
    classifier = RegimeClassifier()
    start = time.time()  
    result = classifier.classify(sample_features)
    latency_ms = (time.time() - start) * 1000
    assert latency_ms < 20  # 20ms SLO
```

## Risk Assessment

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|---------|------------|
| Parameter budget overrun | Medium | High | Automated CI gates |
| Regime collapse | Medium | Medium | Multi-scale validation |
| RAG latency violation | High | Medium | Circuit breaker fail-open |
| Data leakage | Low | Critical | Automated MI testing |

### Operational Risks  
| Risk | Probability | Impact | Mitigation |
|------|-------------|---------|------------|
| Model drift | High | Medium | Daily performance monitoring |
| Infrastructure failure | Medium | High | Multi-AZ deployment |
| Regulatory violation | Low | Critical | Strict simulation testing |
| Performance degradation | Medium | Medium | Continuous benchmarking |

## Success Criteria

### Technical Metrics
- [ ] Parameter count: 26.5M ≤ params ≤ 27.5M
- [ ] Latency: p95 < 80ms fail-open, p95 < 100ms full system
- [ ] Data leakage: MI(features, puzzle_id) < 0.1 bits
- [ ] Reproducibility: Bit-exact reproduction with seeds
- [ ] Compliance: Zero SSR/LULD violations in 1-month replay

### Performance Metrics  
- [ ] Sharpe ratio: >10% improvement vs baseline
- [ ] Maximum drawdown: <15% in validation period
- [ ] SPA test: p < 0.10 vs baseline with multiple testing correction
- [ ] Reality Check: Bootstrap p < 0.05
- [ ] Regime stability: <5% regime switches per day

### Operational Metrics
- [ ] Availability: >99.9% uptime
- [ ] Kill switch: <30s response time
- [ ] Deployment: Zero-downtime automated deployment
- [ ] Monitoring: Full observability stack operational
- [ ] Documentation: Complete technical documentation

This system design provides the comprehensive architecture foundation required for successful implementation of DualHRQ 2.0 with dynamic conditioning.