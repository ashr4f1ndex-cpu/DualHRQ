# DR-001: DualHRQ 2.0 Conditioning System Design Review

**Review Date:** TBD  
**Reviewers:** Senior Engineering Team  
**Design Version:** v1.0  
**Review Type:** Architecture Shred Session  

## Purpose

This design review is intended to "front-load the pain" by having Senior Engineers systematically shred the proposed conditioning system design before development begins. All critical flaws must be identified and resolved in this phase.

## Design Review Checklist

### 1. Parameter Budget Validation ⚠️ CRITICAL

**Question**: Does param count provably land in 26.5–27.5M?

**Evidence Required:**
- [ ] Mathematical proof of parameter calculation
- [ ] Working `tools/param_count.py` script with exact counts
- [ ] Config file producing target dimensions (H=384d/4L, L=512d/4L)
- [ ] Margin analysis: what's the buffer for unexpected parameter growth?

**Specific Concerns:**
- Current config shows 46.68M parameters - how confident are we that 26.5M is achievable?
- What happens if conditioning system needs more than 0.3M parameters?
- Are there hidden parameters in batch norm, embeddings, or other components?

**Required Artifacts:**
```bash
tools/param_count.py --config config/hrm_27m.yaml
# Expected output: 26,500,000 ≤ params ≤ 27,500,000
```

### 2. ID Proxy Detection ⚠️ CRITICAL

**Question**: Are any ID proxies possible that could enable memorization?

**Evidence Required:**
- [ ] Complete feature audit: no features correlated with puzzle_id
- [ ] Join analysis: no implicit IDs through data joins
- [ ] Temporal analysis: no time-based ID proxies
- [ ] Cross-sectional analysis: no ranking-based ID proxies

**Specific Concerns:**
- Could regime classification learn to distinguish specific puzzles?
- Are there subtle ID proxies in technical indicators?
- Does pattern retrieval create memorization paths?
- What about data preprocessing that might leak IDs?

**Required Tests:**
```bash
tools/check_no_static_ids.py --strict
pytest tests/conditioning/test_leakage_mi.py  # MI < 0.1 bits
pytest tests/conditioning/test_shuffle_codes.py  # >50% drop
```

### 3. Fail-Open Path Verification ⚠️ CRITICAL

**Question**: What breaks if RAG is disabled? Is fail-open path verified?

**Evidence Required:**
- [ ] Fail-open path tested under load
- [ ] Latency guarantees with regime-only: <80ms p95
- [ ] Performance degradation quantified: how much worse is regime-only?
- [ ] Circuit breaker response time: <10ms to disable RAG

**Specific Concerns:**
- Does regime-only conditioning provide sufficient signal?
- What's the performance impact of disabling RAG?
- Can the system gracefully handle RAG timeouts?
- Is the circuit breaker itself a failure point?

**Required Tests:**
```python
def test_fail_open_latency():
    with mock_rag_timeout():
        start = time.time()
        result = conditioning_system.forward(features)
        latency = (time.time() - start) * 1000
        assert latency < 80  # ms
```

### 4. Reproducibility Requirements 🔍 HIGH

**Question**: Reproducibility knobs (seeds, versions, data snapshots) documented?

**Evidence Required:**
- [ ] Comprehensive seed control across torch/numpy/random
- [ ] Data versioning strategy with immutable snapshots
- [ ] Model checkpoint reproducibility
- [ ] Environment determinism (CUDA, system dependencies)

**Specific Concerns:**
- How do we ensure bit-exact reproduction across environments?
- What happens when dependencies are updated?
- Are there any non-deterministic operations in the pipeline?
- How is data drift detected and handled?

**Required Implementation:**
```python
@dataclass
class ReproducibilityConfig:
    torch_seed: int
    numpy_seed: int
    python_seed: int
    data_version: str  # Immutable snapshot ID
    model_checkpoint: str
    environment_hash: str  # Dependency fingerprint
```

### 5. Strict-Sim Rule Compliance 🔍 HIGH

**Question**: Strict-sim rule tables (SSR/LULD) immutable + tested?

**Evidence Required:**
- [ ] SSR rules: 10% decline → uptick-only execution
- [ ] LULD rules: 5% bands with last-25min doubling
- [ ] Rule immutability: no runtime modification possible
- [ ] Historical replay: zero violations in 1-month test

**Specific Concerns:**
- Are SSR/LULD rules implemented correctly?
- How do we prevent accidental rule modification?
- What's the testing coverage for edge cases?
- How are new rule changes incorporated safely?

**Required Tests:**
```bash
pytest tests/sim/test_ssr_luld_replay.py --historical-data=1month
# Expected: 0 violations
```

### 6. CI Gate Completeness 🔍 HIGH

**Question**: CI gates are binary and mapped to exact test files?

**Evidence Required:**
- [ ] All gates have corresponding test files
- [ ] Gates fail the build definitively (no warnings)
- [ ] Test files are executable and well-defined
- [ ] Gate thresholds are justified and documented

**Required CI Gates:**
```yaml
# Must all pass for merge
gates:
  - tests/conditioning/test_param_gate.py          # 26.5M ≤ params ≤ 27.5M
  - tests/conditioning/test_leakage_mi.py          # MI < 0.1 bits
  - tests/conditioning/test_shuffle_codes.py       # >50% performance drop  
  - tests/conditioning/test_determinism.py         # Bit-exact reproduction
  - tests/sim/test_ssr_luld_replay.py             # Zero violations
```

## Additional Technical Concerns

### Architecture Scalability
- [ ] How does the system handle 10K+ patterns in the pattern library?
- [ ] What's the memory usage under peak load?
- [ ] How does pattern retrieval scale with library size?
- [ ] Are there any O(n²) algorithms that could become bottlenecks?

### Error Handling & Edge Cases
- [ ] What happens when no regime is classified?
- [ ] How are malformed patterns handled?
- [ ] What's the behavior during market halt conditions?
- [ ] How are network partitions and timeouts handled?

### Security & Compliance
- [ ] Are there any data exfiltration risks in the pattern library?
- [ ] How is sensitive market data protected?
- [ ] What audit logs are maintained for compliance?
- [ ] Are there any potential manipulation vectors?

### Monitoring & Alerting
- [ ] How quickly can we detect system degradation?
- [ ] What metrics indicate imminent failure?
- [ ] Are alert thresholds calibrated correctly?
- [ ] Is there sufficient observability for debugging?

## Review Questions for Design Team

### Technical Deep Dive
1. **Parameter Budget**: Walk through the exact parameter calculation. Show your work.
2. **ID Leakage**: Demonstrate that no features are correlated with puzzle_id. Show the tests.
3. **Latency SLO**: Prove that fail-open path meets <80ms requirement under load.
4. **Reproducibility**: Show bit-exact reproduction across different environments.
5. **Compliance**: Demonstrate zero SSR/LULD violations in historical replay.

### Design Trade-offs
1. **Why hybrid over pure regime or pure RAG?** What's the quantitative justification?
2. **Why 0.3M parameter limit for conditioning?** What's the sensitivity analysis?
3. **Why 2-week purge + 1-week embargo?** What's the leakage analysis?
4. **Why fail-open instead of fail-closed?** What are the risk trade-offs?

### Implementation Concerns
1. **What's the biggest technical risk?** How are you mitigating it?
2. **What assumptions are you making about data quality?** How robust is the design?
3. **What happens if performance is worse than baseline?** What's the rollback plan?
4. **How confident are you in the 6-month timeline?** What could extend it?

## Review Outcome Classification

### ✅ APPROVED - Proceed to implementation
- All critical concerns addressed
- Parameter budget mathematically verified  
- ID leakage prevention demonstrated
- Fail-open path validated
- CI gates operational

### ⚠️ APPROVED WITH CONDITIONS - Address concerns before development
- Minor technical concerns identified
- Specific conditions must be met
- Re-review required after changes

### ❌ REJECTED - Major redesign required
- Critical flaws identified
- Architecture fundamentally unsound
- Return to design phase

## Required Deliverables Before Development

1. **Working parameter counting script**: `tools/param_count.py`
2. **ID detection tool**: `tools/check_no_static_ids.py` 
3. **Failing test stubs**: All CI gate tests failing appropriately
4. **Module stubs**: Import structure that doesn't break CI
5. **Updated design doc**: Addressing all review concerns

## Sign-off

- [ ] **Lead Architect**: Design fundamentally sound
- [ ] **Senior ML Engineer**: Parameter budget achievable, no leakage vectors
- [ ] **Senior Backend Engineer**: Latency SLO realistic, fail-open robust
- [ ] **DevOps/SRE**: Monitoring sufficient, deployment strategy sound
- [ ] **Compliance/Risk**: Regulatory requirements met, risk acceptable

---

**Next Steps**: Address all concerns marked as CRITICAL before proceeding. Schedule follow-up review if needed.