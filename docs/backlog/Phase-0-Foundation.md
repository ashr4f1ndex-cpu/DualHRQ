# Phase 0: Foundation & Guards - Sprint Backlog

**Duration:** Weeks 1-2  
**Goal:** Establish baseline protections and prevent regressions  
**Board Columns:** Design → Tests → Impl → Review → CI → Merge  

## P0 Tickets (Critical - Block everything else)

### DRQ-001: Add Parameter Counter + CI Gate
**Priority:** P0  
**Story Points:** 8  
**Assignee:** Core ML Team  

**Description:** Create automated parameter counting tool and CI gate to prevent parameter budget violations.

**Acceptance Criteria:**
- [ ] `tools/param_count.py` accurately counts all model parameters
- [ ] Script works with current config: `config/default_hrm27m.yaml`
- [ ] CI fails build if parameters > 27.5M
- [ ] Tool reports exact breakdown by component (H-module, L-module, heads, conditioning)

**Tasks:**
1. **[TESTS]** Write `tests/conditioning/test_param_gate.py` (TDD first)
   ```python
   def test_param_count_current_config():
       # Should show 46.68M parameters (current broken state)
       
   def test_param_count_target_config():  
       # Should show ~26.7M parameters (target state)
       
   def test_ci_gate_blocks_over_budget():
       # Should fail CI if >27.5M parameters
   ```
   
2. **[IMPL]** Implement `tools/param_count.py`
   ```python
   def count_hrm_parameters(config_path: str) -> Dict[str, int]:
       """Count parameters by component"""
   
   def verify_budget_compliance(param_count: int, max_params: int = 27_500_000) -> bool:
       """Verify parameter budget compliance"""
   ```
   
3. **[CI]** Add to CI pipeline: fail build if budget exceeded
4. **[REVIEW]** Code review + documentation

**Dependencies:** None  
**Blocker For:** All other tickets  

---

### DRQ-002: Prove Absence of Static IDs  
**Priority:** P0  
**Story Points:** 5  
**Assignee:** Core ML Team

**Description:** Create automated detection tool to prevent puzzle_id usage and verify no ID leakage.

**Acceptance Criteria:**
- [ ] `tools/check_no_static_ids.py` scans codebase for puzzle_id usage
- [ ] Tool uses allowlist for legitimate ID usage (logging, debugging only)
- [ ] CI fails if static IDs detected outside allowlist
- [ ] Mutual information tests verify no ID correlation

**Tasks:**
1. **[TESTS]** Write failing tests first
   ```python
   # tests/conditioning/test_leakage_mi.py
   def test_no_puzzle_id_in_features():
       # MI(features, puzzle_id) < 0.1 bits
       
   def test_no_puzzle_id_in_conditioning():
       # Conditioning output uncorrelated with puzzle_id
   ```
   
   ```python  
   # tests/conditioning/test_shuffle_codes.py
   def test_shuffle_degradation():
       # Performance drops >50% on shuffled labels
   ```

2. **[IMPL]** Implement ID detection tool
   ```python
   def scan_for_static_ids(codebase_path: str, allowlist: List[str]) -> List[str]:
       """Scan for prohibited static ID usage"""
       
   def compute_mutual_information(features: np.ndarray, ids: np.ndarray) -> float:
       """Compute MI between features and IDs"""
   ```

3. **[CI]** Wire to CI pipeline
4. **[REVIEW]** Validate detection accuracy

**Dependencies:** None  
**Blocker For:** All conditioning system work  

---

### DRQ-003: Determinism Smoke Tests
**Priority:** P0  
**Story Points:** 5  
**Assignee:** Core ML Team

**Description:** Establish deterministic reproduction baseline with exact seed control.

**Acceptance Criteria:**
- [ ] `tests/conditioning/test_determinism.py` passes with bit-exact reproduction
- [ ] All random seeds controlled (torch, numpy, python, CUDA)
- [ ] Same input → identical output across multiple runs
- [ ] Seed configuration documented and enforced

**Tasks:**
1. **[TESTS]** Write determinism test first
   ```python
   def test_exact_reproduction():
       # Same seeds → identical outputs
       
   def test_different_seeds_different_outputs():
       # Different seeds → different outputs
       
   def test_cuda_determinism():
       # GPU runs produce identical results
   ```

2. **[IMPL]** Implement seed management
   ```python
   def set_all_seeds(seed: int = 42):
       """Set all random seeds for reproducibility"""
       
   def validate_determinism(model_func, data, n_runs=3) -> bool:
       """Validate deterministic behavior"""
   ```

3. **[CI]** Add to CI pipeline  
4. **[REVIEW]** Test across different environments

**Dependencies:** None  
**Blocker For:** All model training and validation  

---

### DRQ-004: Strict-Sim Replay Test Foundation
**Priority:** P0  
**Story Points:** 8  
**Assignee:** Trading Systems Team

**Description:** Implement SSR/LULD compliance testing with historical replay validation.

**Acceptance Criteria:**
- [ ] `tests/sim/test_ssr_luld_replay.py` validates regulatory compliance
- [ ] SSR rules: 10% decline → uptick-only execution
- [ ] LULD rules: 5% bands, last-25min doubling  
- [ ] Zero violations in historical replay test
- [ ] Immutable rule tables (no runtime modification)

**Tasks:**
1. **[TESTS]** Write compliance tests first
   ```python
   def test_ssr_trigger_detection():
       # 10% decline correctly triggers SSR
       
   def test_luld_band_calculation():
       # 5% bands calculated correctly
       
   def test_historical_replay_zero_violations():
       # Historical data replay shows 0 violations
   ```

2. **[IMPL]** Implement compliance engine
   ```python
   class SSRLULDEngine:
       def check_ssr_trigger(self, price_history: pd.Series) -> bool:
       def check_luld_bands(self, price: float, reference_price: float) -> bool:
       def validate_compliance(self, trades: List[Trade]) -> ComplianceReport:
   ```

3. **[CI]** Add to CI pipeline
4. **[REVIEW]** Validate against known compliance scenarios

**Dependencies:** Market data access  
**Blocker For:** Trading system integration  

---

## P1 Tickets (High Priority - Enable Phase 1)

### DRQ-005: Create Module Skeletons
**Priority:** P1  
**Story Points:** 3  
**Assignee:** Core ML Team

**Description:** Create import structure to prevent CI failures during development.

**Acceptance Criteria:**
- [ ] All modules importable without errors
- [ ] Basic class stubs with pass-through behavior  
- [ ] No functionality implemented (just structure)
- [ ] Import tests pass in CI

**Tasks:**
1. **[TESTS]** Write import tests first
   ```python
   def test_conditioning_imports():
       from src.models.conditioning import FiLMLayer, RegimeClassifier, PatternRAG
       
   def test_core_model_imports():
       from src.models import pattern_library, hrm_integration, adaptive_budget
   ```

2. **[IMPL]** Create module stubs
   ```python
   # src/models/conditioning/__init__.py
   # src/models/conditioning/film.py  
   # src/models/conditioning/regime.py
   # src/models/conditioning/rag.py
   # src/models/pattern_library.py
   # src/models/hrm_integration.py  
   # src/models/adaptive_budget.py
   ```

3. **[CI]** Verify imports work
4. **[REVIEW]** Code review for structure

**Dependencies:** None  
**Blocker For:** Phase 1 development  

---

### DRQ-006: FiLM Shared Head Implementation (≤0.3M params)
**Priority:** P1  
**Story Points:** 13  
**Assignee:** Core ML Team

**Description:** Implement FiLM conditioning layer with strict parameter budget and feature flags.

**Acceptance Criteria:**
- [ ] FiLM layer ≤0.1M parameters  
- [ ] Latency <5ms for typical batch
- [ ] Feature flag controls (default OFF)
- [ ] Integration with HRM L-module
- [ ] Parameter count CI gate passes

**Tasks:**
1. **[TESTS]** Write tests first (expand stubs)
   ```python
   def test_film_parameter_budget():
       film = FiLMLayer(context_dim=256, target_dim=512)
       assert count_parameters(film) <= 100_000
       
   def test_film_latency():
       # <5ms latency requirement
       
   def test_film_modulation_effect():
       # Verify conditioning actually modifies tokens
   ```

2. **[IMPL]** Implement FiLM layer
   ```python
   class FiLMLayer(nn.Module):
       def __init__(self, context_dim: int, target_dim: int):
       def forward(self, l_tokens: Tensor, context: Tensor) -> Tensor:
   ```

3. **[CONFIG]** Add feature flags
   ```yaml
   conditioning:
     enable_film: false  # Default OFF
   ```

4. **[REVIEW]** Code review + performance testing

**Dependencies:** DRQ-001 (parameter gate), DRQ-005 (module stubs)  
**Blocker For:** Regime and RAG implementation  

---

### DRQ-007: Regime Features v1 Implementation  
**Priority:** P1  
**Story Points:** 21  
**Assignee:** Data/Features Team

**Description:** Implement core regime features (TSRV, BPV, Amihud, SSR/LULD state) with <10ms computation SLO.

**Acceptance Criteria:**
- [ ] All regime features implemented and tested
- [ ] Computation time <10ms per symbol
- [ ] No correlation with puzzle_id (MI < 0.01)  
- [ ] Feature quality validation passes
- [ ] Unit tests for all feature computations

**Tasks:**
1. **[TESTS]** Write feature tests first
   ```python
   def test_tsrv_computation():
       # Test TSRV calculation accuracy
       
   def test_bpv_jump_detection():  
       # Test BPV distinguishes jumps from continuous moves
       
   def test_regime_features_no_id_correlation():
       # MI(features, puzzle_id) < 0.01
       
   def test_feature_computation_latency():
       # <10ms SLO per symbol
   ```

2. **[IMPL]** Implement feature computations
   ```python
   def compute_tsrv(returns: pd.Series, windows: List[int]) -> Dict[str, float]:
   def compute_bpv(returns: pd.Series, window: int = 30) -> float:
   def compute_amihud(returns: pd.Series, volume: pd.Series) -> float:
   def compute_regulatory_features(price_history: pd.Series) -> np.ndarray:
   ```

3. **[INTEGRATION]** Real-time feature pipeline
4. **[REVIEW]** Performance and accuracy validation

**Dependencies:** DRQ-002 (ID leakage detection)  
**Blocker For:** Regime classifier implementation  

---

### DRQ-008: Regime Classifier (≤0.1M params) + Unit Tests
**Priority:** P1  
**Story Points:** 13  
**Assignee:** Core ML Team

**Description:** Implement regime classification with stability validation and parameter budget compliance.

**Acceptance Criteria:**
- [ ] Regime classifier ≤0.1M parameters
- [ ] Latency <20ms per classification
- [ ] Regime stability: <5% transitions per day
- [ ] 8 distinct regime states identified
- [ ] Unit tests for all functionality

**Tasks:**
1. **[TESTS]** Write classifier tests first
   ```python
   def test_regime_classifier_budget():
       classifier = RegimeClassifier(input_dim=20, num_regimes=8)
       assert count_parameters(classifier) <= 100_000
       
   def test_regime_stability():
       # <5% regime transitions per day
       
   def test_regime_classification_latency():
       # <20ms SLO
   ```

2. **[IMPL]** Implement regime classifier  
   ```python
   class RegimeClassifier(nn.Module):
       def __init__(self, input_dim: int, num_regimes: int = 8):
       def forward(self, features: Tensor) -> Tensor:
       def get_regime_names(self) -> List[str]:
   ```

3. **[TRAINING]** Train regime classifier on historical data
4. **[REVIEW]** Validate regime interpretability

**Dependencies:** DRQ-006 (FiLM layer), DRQ-007 (regime features)  
**Blocker For:** Hybrid conditioning system  

---

## Sprint Planning

### Week 1: Critical Foundation
**Focus:** Parameter gates and ID detection (cannot proceed without these)

**Monday-Wednesday:**
- DRQ-001: Parameter counter + CI gate  
- DRQ-002: Static ID detection tool

**Thursday-Friday:**  
- DRQ-003: Determinism smoke tests
- DRQ-004: Strict-sim replay foundation

**Sprint Goal:** All P0 tickets complete, CI gates operational

### Week 2: Module Structure + Core Features  
**Focus:** Enable Phase 1 development

**Monday-Tuesday:**
- DRQ-005: Module skeletons (prevent import errors)
- DRQ-006: FiLM layer implementation

**Wednesday-Friday:**
- DRQ-007: Regime features v1
- DRQ-008: Regime classifier

**Sprint Goal:** Core conditioning components ready, feature flags operational

## Definition of Done

**For All Tickets:**
- [ ] Tests written first (TDD)
- [ ] Implementation passes all tests
- [ ] Code review completed (2 approvals)
- [ ] CI pipeline passes (all gates green)
- [ ] Documentation updated
- [ ] Feature flags configured (default OFF)

**For P0 Tickets (Additional):**
- [ ] CI gate prevents regression
- [ ] Tool/test runs in <30 seconds  
- [ ] Zero false positives/negatives
- [ ] Works across all environments

**For P1 Tickets (Additional):**
- [ ] Parameter budget verified
- [ ] Performance benchmarks met
- [ ] Integration tests pass
- [ ] Ready for Phase 1 development

## Risk Mitigation

**Technical Risks:**
- Parameter counting inaccuracy → Validate against manual calculation
- ID detection false positives → Comprehensive allowlist + validation
- Determinism breaks in CI → Test across multiple environments
- Feature computation too slow → Profile and optimize critical path

**Process Risks:**  
- Scope creep in foundation → Strict adherence to acceptance criteria
- Incomplete test coverage → TDD mandatory, coverage >95%
- CI gates too restrictive → Careful threshold calibration
- Team blocked on P0 → Daily standups, immediate escalation

**Rollback Plan:**
- All changes behind feature flags
- Can disable new functionality immediately  
- Automated rollback in CI pipeline
- Previous working state always available