# Regime Features Specification

**Team:** Data/Features  
**Sprint:** Phase 1 (Weeks 3-5)  
**Dependencies:** Market data pipeline  
**SLO:** <10ms feature computation  

## Scope

Implement regime-based feature engineering that captures market microstructure and regulatory state without puzzle_id dependencies.

## Feature Categories

### 1. Volatility Features

#### TSRV (Time-Scaled Realized Volatility)
**Definition:** Multi-scale realized volatility estimation  
**Computation:**
```python
def compute_tsrv(returns: pd.Series, windows: List[int]) -> Dict[str, float]:
    """
    windows: [5, 15, 30, 60] minutes
    returns: intraday returns at 1-minute frequency
    """
    tsrv_features = {}
    for w in windows:
        rv = (returns.rolling(w).std() * np.sqrt(252 * 390)).iloc[-1]
        tsrv_features[f'tsrv_{w}m'] = float(rv)
    return tsrv_features
```

**Output Shape:** `[B, 4]` (one per window)  
**Range:** `[0.0, 2.0]` (0% to 200% annualized volatility)

#### BPV (Bipower Variation)  
**Definition:** Jump-robust volatility measure
**Computation:**
```python
def compute_bpv(returns: pd.Series, window: int = 30) -> float:
    """
    Bipower variation to separate continuous vs jump volatility
    """
    abs_returns = returns.abs().rolling(window)
    bpv = (abs_returns.shift(1) * abs_returns).mean() * (np.pi / 2)
    return float(bpv.iloc[-1])
```

**Output Shape:** `[B, 1]`  
**Range:** `[0.0, 1.0]` (normalized bipower variation)

### 2. Liquidity Features

#### Amihud Illiquidity
**Definition:** Price impact per unit volume
**Computation:**
```python
def compute_amihud(returns: pd.Series, volume: pd.Series, 
                   window: int = 60) -> float:
    """
    Amihud = |return| / dollar_volume
    """
    dollar_volume = volume * returns.abs()  # Simplified
    illiquidity = (returns.abs() / dollar_volume).rolling(window).mean()
    return float(illiquidity.iloc[-1])
```

**Output Shape:** `[B, 1]`  
**Range:** `[0.0, 10.0]` (log-scaled illiquidity measure)

#### Volume Profile
**Definition:** Intraday volume distribution patterns
**Computation:**
```python
def compute_volume_profile(volume: pd.Series, 
                          price: pd.Series) -> np.ndarray:
    """
    Volume distribution across price deciles
    """
    price_deciles = pd.qcut(price, q=10, labels=False)
    profile = np.zeros(10)
    for i in range(10):
        profile[i] = volume[price_deciles == i].sum()
    return profile / profile.sum()  # Normalize
```

**Output Shape:** `[B, 10]` (decile buckets)  
**Range:** `[0.0, 1.0]` (probability distribution)

### 3. Spread Dynamics

#### Bid-Ask Spread Features
**Computation:**
```python
def compute_spread_dynamics(bid: pd.Series, ask: pd.Series, 
                           mid: pd.Series) -> Dict[str, float]:
    """
    Microstructure spread-based features
    """
    spread = ask - bid
    relative_spread = spread / mid
    
    return {
        'spread_mean': float(spread.tail(30).mean()),
        'spread_std': float(spread.tail(30).std()), 
        'relative_spread': float(relative_spread.tail(30).mean())
    }
```

**Output Shape:** `[B, 3]`  
**Range:** Various (dollar amounts and percentages)

### 4. Regulatory State Features

#### SSR/LULD State
**Definition:** Circuit breaker and short sale restriction status
**Computation:**
```python
@dataclass
class RegulatoryState:
    ssr_active: bool          # Short Sale Restriction active
    luld_active: bool         # Limit Up/Limit Down active  
    circuit_breaker: bool     # Market-wide circuit breaker
    last_band_active: bool    # Last 25 minutes of trading
    
def compute_regulatory_features(price_history: pd.Series,
                               timestamp: pd.Timestamp) -> np.ndarray:
    """
    Encode regulatory state as one-hot features
    """
    state = RegulatoryState(
        ssr_active=check_ssr_trigger(price_history),
        luld_active=check_luld_bands(price_history), 
        circuit_breaker=check_circuit_breaker(),
        last_band_active=check_last_25_minutes(timestamp)
    )
    return np.array([state.ssr_active, state.luld_active, 
                    state.circuit_breaker, state.last_band_active])
```

**Output Shape:** `[B, 4]` (binary features)  
**Range:** `{0, 1}` (boolean flags)

## Feature Engineering Pipeline

### Real-time Computation
```python
class RegimeFeatureEngine:
    def __init__(self, config: RegimeConfig):
        self.tsrv_windows = config.tsrv_windows
        self.lookback_minutes = config.lookback_minutes
        
    def compute_features(self, market_data: MarketData) -> RegimeFeatures:
        """
        Compute all regime features for current timestamp
        SLO: <10ms computation time
        """
        features = RegimeFeatures()
        
        # Volatility features (3ms)
        features.tsrv = compute_tsrv(market_data.returns, self.tsrv_windows)
        features.bpv = compute_bpv(market_data.returns)
        
        # Liquidity features (4ms)  
        features.amihud = compute_amihud(market_data.returns, market_data.volume)
        features.volume_profile = compute_volume_profile(market_data.volume, 
                                                        market_data.price)
        
        # Spread features (2ms)
        features.spread_dynamics = compute_spread_dynamics(market_data.bid,
                                                          market_data.ask,
                                                          market_data.mid)
        
        # Regulatory features (1ms)
        features.regulatory_state = compute_regulatory_features(market_data.price_history,
                                                               market_data.timestamp)
        
        return features
```

### Batch Computation  
```python
def compute_regime_features_batch(market_data_batch: List[MarketData]) -> torch.Tensor:
    """
    Vectorized feature computation for training
    """
    batch_features = []
    for data in market_data_batch:
        features = feature_engine.compute_features(data)
        feature_vector = features.to_tensor()  # [20,] dimensional
        batch_features.append(feature_vector)
    
    return torch.stack(batch_features)  # [B, 20]
```

## Data Validation

### Feature Quality Checks
```python
def validate_regime_features(features: torch.Tensor) -> bool:
    """
    Validate feature quality and detect anomalies
    """
    # Check for NaN/inf values
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()
    
    # Check feature ranges
    assert (features[:, :4] >= 0).all()  # TSRV non-negative
    assert (features[:, 4] >= 0).all()   # BPV non-negative  
    assert (features[:, 5] >= 0).all()   # Amihud non-negative
    assert (features[:, 6:16] >= 0).all() and (features[:, 6:16] <= 1).all()  # Volume profile
    assert (features[:, 16:19] >= 0).all()  # Spread features
    assert features[:, 19:].int().eq(features[:, 19:]).all()  # Regulatory binary
    
    return True
```

### Leakage Prevention
```python
def test_no_puzzle_id_correlation():
    """
    Verify regime features are not correlated with puzzle_id
    """
    features = compute_regime_features_batch(validation_data)
    puzzle_ids = [d.puzzle_id for d in validation_data]
    
    # Mutual information test
    for i in range(features.shape[1]):
        mi = mutual_info_regression(features[:, i:i+1], puzzle_ids)
        assert mi[0] < 0.01, f"Feature {i} correlated with puzzle_id: MI={mi[0]}"
```

## Configuration

```yaml
regime_features:
  tsrv_windows: [5, 15, 30, 60]  # minutes
  lookback_minutes: 120           # 2 hours of history
  
  bpv:
    window_minutes: 30
    
  amihud:
    window_minutes: 60
    
  volume_profile:
    price_deciles: 10
    
  spread:
    lookback_minutes: 30
    
  regulatory:
    ssr_decline_threshold: 0.10    # 10% decline triggers SSR
    luld_band_percent: 0.05        # 5% LULD bands
    last_minutes_threshold: 25     # Last 25 minutes special handling
```

## Performance Requirements

### Latency SLOs
- **Real-time computation**: <10ms per symbol
- **Batch computation**: <1ms per sample (vectorized)
- **Feature validation**: <0.1ms per sample

### Memory Usage
- **Feature cache**: <100MB per symbol (2 hours history)
- **Computation buffers**: <10MB per symbol

## Testing Strategy

### Unit Tests (Write First)
```python
def test_tsrv_computation():
    # Test TSRV calculation accuracy
    
def test_bpv_jump_detection():
    # Test BPV distinguishes jumps from continuous moves
    
def test_amihud_scaling():
    # Test Amihud illiquidity scaling properties
    
def test_volume_profile_normalization():
    # Test volume profile sums to 1.0
    
def test_regulatory_state_accuracy():
    # Test SSR/LULD state detection
```

### Integration Tests
```python  
def test_feature_pipeline_latency():
    # End-to-end latency under SLO
    
def test_batch_consistency():
    # Batch vs individual computation consistency
    
def test_missing_data_handling():
    # Graceful degradation with incomplete data
```

## Rollout Plan

### Week 3: Core Implementation
- TSRV, BPV, Amihud computation
- Basic unit tests
- Performance profiling

### Week 4: Microstructure Features
- Volume profile computation  
- Spread dynamics
- Regulatory state detection

### Week 5: Integration & Validation
- Pipeline integration
- Leakage testing
- Performance optimization

## Success Criteria

- [ ] All features computable within <10ms SLO
- [ ] No correlation with puzzle_id (MI < 0.01)
- [ ] Feature quality validation passes
- [ ] Integration with conditioning system
- [ ] Performance benchmarks met