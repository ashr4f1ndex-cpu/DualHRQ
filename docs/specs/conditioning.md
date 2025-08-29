# Conditioning System Specification

**Team:** Core ML  
**Sprint:** Phase 1 (Weeks 3-8)  
**Parameter Budget:** ≤0.3M total  
**Latency Budget:** <100ms primary, <80ms fail-open  

## Scope

Implement dynamic conditioning system to replace static puzzle_id with regime-based + pattern-based conditioning for HRM generalization.

## Components

### 1. FiLM Layer (`src/models/conditioning/film.py`)
**Budget:** ≤0.1M parameters  
**Latency:** <5ms  

**API:**
```python
class FiLMLayer(nn.Module):
    def __init__(self, context_dim: int, target_dim: int):
        # context_dim: conditioning context size
        # target_dim: L-module dimension (512)
        
    def forward(self, l_tokens: Tensor, context: Tensor) -> Tensor:
        # l_tokens: [B, T, target_dim] 
        # context: [B, context_dim]
        # returns: [B, T, target_dim] modulated tokens
```

**Tests to Write First:**
```python
def test_film_parameter_budget():
    film = FiLMLayer(context_dim=256, target_dim=512)
    assert count_parameters(film) <= 100_000  # 0.1M limit

def test_film_latency():
    # Target: <5ms for typical batch
    
def test_film_modulation_effect():
    # Verify modulation actually changes L-tokens
```

### 2. Regime Classifier (`src/models/conditioning/regime.py`)  
**Budget:** ≤0.1M parameters  
**Latency:** <20ms  

**Input Features:**
- TSRV (Time-Scaled Realized Volatility): float32
- BPV (Bipower Variation): float32  
- Amihud Illiquidity: float32
- SSR/LULD State: categorical (4 states)
- Volume Profile: float32[10] (decile buckets)
- Spread Dynamics: float32[3] (bid/ask/mid features)

**API:**
```python
class RegimeClassifier(nn.Module):
    def __init__(self, input_dim: int, num_regimes: int = 8):
        
    def forward(self, features: Tensor) -> Tensor:
        # features: [B, input_dim]
        # returns: [B, num_regimes] logits
        
    def get_regime_names(self) -> List[str]:
        return ["low_vol", "high_vol", "trending", "mean_reverting", 
                "ssr_active", "luld_active", "illiquid", "normal"]
```

**Tests to Write First:**
```python
def test_regime_classifier_budget():
    classifier = RegimeClassifier(input_dim=20, num_regimes=8)
    assert count_parameters(classifier) <= 100_000

def test_regime_stability():
    # Regime shouldn't change >5% per day
    
def test_regime_no_id_correlation():
    # Verify regimes not correlated with puzzle_id
```

### 3. Pattern RAG System (`src/models/conditioning/rag.py`)
**Budget:** ≤0.1M parameters  
**Latency:** <60ms (with fail-open to 0ms)  

**API:**
```python
class PatternRAG(nn.Module):
    def __init__(self, pattern_dim: int, context_dim: int):
        
    def forward(self, query_features: Tensor, 
                timeout_ms: int = 60) -> Optional[Tensor]:
        # query_features: [B, feature_dim]  
        # returns: [B, context_dim] or None (timeout/fail-open)
        
    def update_pattern_library(self, patterns: List[Pattern]):
        # Online pattern library updates
```

**Tests to Write First:**
```python
def test_rag_timeout_handling():
    # Verify proper fail-open behavior
    
def test_rag_latency_slo():
    # Must return result or None within 60ms
    
def test_pattern_retrieval_quality():
    # Verify retrieved patterns are relevant
```

### 4. Unified Interface (`src/models/conditioning/__init__.py`)

**API:**
```python
class ConditioningSystem(nn.Module):
    def __init__(self, config: ConditioningConfig):
        self.regime_classifier: RegimeClassifier
        self.pattern_rag: PatternRAG  
        self.film_layer: FiLMLayer
        self.enable_regime: bool
        self.enable_rag: bool
        
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        # features: {"regime_features": [...], "pattern_features": [...]}
        # returns: [B, context_dim] unified conditioning context
```

## Configuration

```yaml
conditioning:
  enable_regime: false        # Feature flag
  enable_rag: false          # Feature flag  
  enable_hybrid: false       # Feature flag
  fail_open: true            # Always enabled
  
  regime:
    num_regimes: 8
    stability_threshold: 0.05
    
  rag:
    pattern_dim: 128
    timeout_ms: 60
    max_patterns: 10000
    
  film:
    context_dim: 256
    target_dim: 512
```

## Integration Points

### Input: Feature Pipeline
```python
# Expected input format
features = {
    "regime_features": torch.tensor([...]),  # [B, 20] 
    "pattern_features": torch.tensor([...]), # [B, 128]
}
```

### Output: HRM Integration
```python
# Output format for HRM L-module
conditioning_context = conditioning_system(features)  # [B, 256]
film_params = film_layer.get_params(conditioning_context)  # gamma, beta
modulated_l_tokens = film_layer(l_tokens, conditioning_context)  # [B, T, 512]
```

## Validation Requirements

### Parameter Budget Validation
```python
def test_total_conditioning_budget():
    system = ConditioningSystem(config)
    total_params = count_parameters(system)
    assert total_params <= 300_000  # 0.3M hard limit
```

### Leakage Prevention  
```python
def test_mutual_information():
    # MI(conditioning_output, puzzle_id) < 0.1 bits
    
def test_shuffle_robustness():  
    # Performance drops >50% on shuffled labels
```

### Latency Validation
```python
def test_end_to_end_latency():
    # Primary path: <100ms
    # Fail-open path: <80ms
```

## Rollout Strategy

### Week 3-4: Core Implementation
- Implement FiLM, Regime, RAG components
- Parameter budget compliance
- Basic unit tests

### Week 5-6: Integration  
- Unified conditioning interface
- Feature flag integration
- Integration testing

### Week 7-8: Validation
- Leakage testing
- Latency validation  
- Performance benchmarking

## Success Criteria

- [ ] All components ≤ parameter budgets
- [ ] Latency SLOs met (primary + fail-open)
- [ ] MI(output, puzzle_id) < 0.1 bits
- [ ] Feature flags control all paths
- [ ] Integration tests pass
- [ ] No CI breakage