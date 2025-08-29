# HRM Integration Specification

**Team:** Core ML  
**Sprint:** Phase 2 (Weeks 9-14)  
**Parameter Budget:** Maintain 26.5M-27.5M total  
**Dependencies:** Conditioning system, Pattern library  

## Scope

Implement HRM adapter layer for dynamic conditioning integration, walk-forward validation, and parameter budget management.

## Architecture Overview

```
Conditioning System → HRM Adapter → HRM Network → Trading Signals
       ↓                  ↓             ↓              ↓
   FiLM params        Integration    H/L modules    Head outputs
   Regime/RAG         Walk-forward   Parameter      Signal gen
   Context            Validation     Management     Performance
```

## Core Components

### 1. HRM Adapter Layer (`src/models/hrm_integration.py`)

#### Main Integration Interface
```python
class HRMAdapter:
    def __init__(self, hrm_config: HRMConfig, conditioning_config: ConditioningConfig):
        self.hrm_net = HRMNet(hrm_config)
        self.conditioning_system = ConditioningSystem(conditioning_config)
        self.parameter_manager = AdaptiveParameterManager()
        
    def forward(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor,
                market_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Integrated forward pass with dynamic conditioning
        
        Args:
            h_tokens: [B, T_h, D_h] daily tokens
            l_tokens: [B, T_l, D_l] intraday tokens  
            market_features: regime + pattern features
            
        Returns:
            predictions: (outA, outB) from HRM heads
            metadata: conditioning info, attention weights, etc.
        """
        # Generate conditioning context
        conditioning_context = self.conditioning_system(market_features)
        
        # Apply FiLM conditioning to L-tokens
        conditioned_l_tokens = self.conditioning_system.film_layer(
            l_tokens, conditioning_context
        )
        
        # Forward through HRM with conditioned tokens
        (h_final, l_final), segments = self.hrm_net(h_tokens, conditioned_l_tokens)
        
        # Extract predictions from final segment
        outA, outB, q_logits = segments[-1]
        
        # Collect metadata for analysis
        metadata = {
            'conditioning_context': conditioning_context,
            'regime_logits': market_features.get('regime_logits'),
            'pattern_context': market_features.get('pattern_context'),
            'q_logits': q_logits,
            'segments_count': len(segments)
        }
        
        return (outA, outB), metadata
```

#### Parameter Budget Manager
```python
class AdaptiveParameterManager:
    def __init__(self, target_params: int = 26_500_000, max_params: int = 27_500_000):
        self.target_params = target_params
        self.max_params = max_params
        self.current_config = None
        
    def optimize_config(self, base_config: HRMConfig) -> HRMConfig:
        """
        Optimize HRM config to meet parameter budget
        """
        # Start with target configuration
        optimized_config = HRMConfig(
            h_layers=4, h_dim=384, h_heads=8, h_ffn_mult=3.0, h_dropout=0.1,
            l_layers=4, l_dim=512, l_heads=8, l_ffn_mult=3.0, l_dropout=0.1,
            segments_N=base_config.segments_N,
            l_inner_T=base_config.l_inner_T,
            act_enable=base_config.act_enable,
            act_max_segments=base_config.act_max_segments,
            ponder_cost=base_config.ponder_cost,
            use_cross_attn=False  # Disable to save parameters
        )
        
        # Verify parameter count
        param_count = self._count_parameters(optimized_config)
        
        if param_count > self.max_params:
            # Reduce dimensions if still over budget
            optimized_config = self._reduce_dimensions(optimized_config)
            
        return optimized_config
        
    def _count_parameters(self, config: HRMConfig) -> int:
        """Count total parameters for given config"""
        # H-module parameters
        h_params = (
            config.h_layers * (
                config.h_dim * config.h_dim * 3 +  # QKV projections
                config.h_dim * config.h_dim +      # Output projection  
                config.h_dim * 2 +                 # RMSNorm weights
                config.h_dim * int(config.h_dim * config.h_ffn_mult) * 2 +  # GLU
                int(config.h_dim * config.h_ffn_mult) * config.h_dim  # GLU output
            ) + config.h_dim  # Final norm
        )
        
        # L-module parameters  
        l_params = (
            config.l_layers * (
                config.l_dim * config.l_dim * 3 +  # QKV projections
                config.l_dim * config.l_dim +      # Output projection
                config.l_dim * 2 +                 # RMSNorm weights  
                config.l_dim * int(config.l_dim * config.l_ffn_mult) * 2 +  # GLU
                int(config.l_dim * config.l_ffn_mult) * config.l_dim  # GLU output
            ) + config.l_dim  # Final norm
        )
        
        # Head parameters
        head_params = config.h_dim + config.l_dim + config.h_dim * 2  # headA + headB + q_head
        
        # Cross attention (if enabled)
        cross_attn_params = 0
        if config.use_cross_attn:
            cross_attn_params = (
                config.l_dim * config.l_dim +      # Q projection
                config.h_dim * config.l_dim * 2 +  # K, V projections  
                config.l_dim * config.l_dim        # Output projection
            )
            
        return h_params + l_params + head_params + cross_attn_params
```

### 2. Walk-Forward Validation System

#### Walk-Forward Cross Validation
```python
class WalkForwardValidator:
    def __init__(self, config: WalkForwardConfig):
        self.train_window_days = config.train_window_days  # 252 (1 year)
        self.test_window_days = config.test_window_days    # 21 (1 month)
        self.step_size_days = config.step_size_days        # 21 (monthly steps)
        self.purge_days = config.purge_days                # 14 (2 weeks)
        self.embargo_days = config.embargo_days            # 7 (1 week)
        
    def create_splits(self, dates: pd.DatetimeIndex) -> List[WalkForwardSplit]:
        """
        Create walk-forward splits with purge and embargo
        """
        splits = []
        start_date = dates.min()
        end_date = dates.max()
        
        current_date = start_date + pd.Timedelta(days=self.train_window_days)
        
        while current_date + pd.Timedelta(days=self.test_window_days) <= end_date:
            # Define train period
            train_start = current_date - pd.Timedelta(days=self.train_window_days)
            train_end = current_date
            
            # Define test period with purge and embargo
            test_start = current_date + pd.Timedelta(days=self.purge_days)
            test_end = test_start + pd.Timedelta(days=self.test_window_days)
            
            # Create split
            split = WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_start=current_date,
                purge_end=test_start
            )
            splits.append(split)
            
            # Move to next period
            current_date += pd.Timedelta(days=self.step_size_days)
            
        return splits
        
    def validate_model(self, model: HRMAdapter, 
                      data: pd.DataFrame) -> WalkForwardResults:
        """
        Perform walk-forward validation
        """
        splits = self.create_splits(data.index)
        results = []
        
        for i, split in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)}: "
                       f"{split.train_start} to {split.test_end}")
            
            # Extract train and test data
            train_data = data[split.train_start:split.train_end]
            test_data = data[split.test_start:split.test_end]
            
            # Train model on train data
            trained_model = self._train_on_split(model, train_data)
            
            # Evaluate on test data  
            test_metrics = self._evaluate_on_split(trained_model, test_data)
            
            split_result = WalkForwardSplitResult(
                split=split,
                train_metrics=self._evaluate_on_split(trained_model, train_data),
                test_metrics=test_metrics,
                model_checkpoint=self._save_model_checkpoint(trained_model, i)
            )
            results.append(split_result)
            
        return WalkForwardResults(splits=results)
```

#### Leakage Detection in Walk-Forward
```python
class LeakageDetector:
    def detect_temporal_leakage(self, predictions: np.ndarray, 
                               targets: np.ndarray,
                               dates: pd.DatetimeIndex) -> Dict[str, float]:
        """
        Detect temporal information leakage
        """
        leakage_metrics = {}
        
        # Test 1: Future information in current predictions
        # Predictions shouldn't be correlated with future targets
        for lag in [1, 5, 10, 20]:  # Days ahead
            if len(targets) > lag:
                future_correlation = np.corrcoef(
                    predictions[:-lag], targets[lag:]
                )[0, 1]
                leakage_metrics[f'future_corr_{lag}d'] = abs(future_correlation)
                
        # Test 2: Prediction consistency across time
        # Similar market conditions should yield similar predictions
        consistency_score = self._compute_prediction_consistency(
            predictions, targets, dates
        )
        leakage_metrics['prediction_consistency'] = consistency_score
        
        # Test 3: Data snooping detection
        # Model performance shouldn't be suspiciously good
        is_leakage = self._statistical_leakage_test(predictions, targets)
        leakage_metrics['statistical_leakage_pvalue'] = is_leakage
        
        return leakage_metrics
        
    def _statistical_leakage_test(self, predictions: np.ndarray, 
                                 targets: np.ndarray) -> float:
        """Statistical test for data snooping"""
        from scipy import stats
        
        # Compute correlation
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        # Test against expected correlation under null hypothesis
        # (random predictions should have ~0 correlation)
        n = len(predictions)
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        return p_value
```

### 3. Performance Monitoring & Logging

#### Model Performance Tracker
```python
class HRMPerformanceTracker:
    def __init__(self, log_dir: str = "logs/hrm_performance/"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        
    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'phase': 'training',
            **metrics
        }
        self.metrics_history.append(log_entry)
        
        # Write to file
        with open(self.log_dir / 'training_metrics.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_conditioning_usage(self, conditioning_metadata: Dict[str, Any]):
        """Log conditioning system usage patterns"""
        usage_entry = {
            'timestamp': datetime.now().isoformat(),
            'regime_distribution': conditioning_metadata.get('regime_logits', []),
            'pattern_count': conditioning_metadata.get('pattern_count', 0),
            'rag_latency_ms': conditioning_metadata.get('rag_latency_ms', 0),
            'fail_open_triggered': conditioning_metadata.get('fail_open_triggered', False)
        }
        
        with open(self.log_dir / 'conditioning_usage.jsonl', 'a') as f:
            f.write(json.dumps(usage_entry) + '\n')
            
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {'error': 'No metrics logged'}
            
        df = pd.DataFrame(self.metrics_history)
        
        return {
            'summary': {
                'total_epochs': df['epoch'].max() if 'epoch' in df else 0,
                'final_train_loss': df[df['phase'] == 'training']['loss'].iloc[-1] if 'loss' in df else None,
                'final_val_loss': df[df['phase'] == 'validation']['loss'].iloc[-1] if 'loss' in df else None,
            },
            'conditioning_stats': self._analyze_conditioning_usage(),
            'performance_trends': self._analyze_performance_trends(df),
            'anomaly_detection': self._detect_performance_anomalies(df)
        }
```

## Configuration

```yaml
hrm_integration:
  parameter_management:
    target_params: 26500000      # 26.5M target
    max_params: 27500000         # 27.5M hard limit
    auto_optimize: true
    
  walk_forward:
    train_window_days: 252       # 1 year training
    test_window_days: 21         # 1 month testing  
    step_size_days: 21           # Monthly updates
    purge_days: 14               # 2-week purge
    embargo_days: 7              # 1-week embargo
    
  conditioning_integration:
    enable_film: true
    enable_cross_attention: false # Disabled to save parameters
    film_context_dim: 256
    
  performance_tracking:
    log_interval: 100            # Log every 100 steps
    save_checkpoint_interval: 1000
    performance_report_interval: 10000
    
  leakage_detection:
    future_correlation_lags: [1, 5, 10, 20]  # Days
    max_future_correlation: 0.05  # 5% threshold
    statistical_significance: 0.01  # p < 0.01
```

## Testing Strategy

### Unit Tests (Write First)
```python
def test_parameter_budget_compliance():
    adapter = HRMAdapter(hrm_config, conditioning_config)
    total_params = count_parameters(adapter)
    assert 26_500_000 <= total_params <= 27_500_000

def test_walk_forward_split_creation():
    validator = WalkForwardValidator(config)
    splits = validator.create_splits(date_index)
    # Verify no overlap between train/test, proper purge/embargo
    
def test_conditioning_integration():
    adapter = HRMAdapter(hrm_config, conditioning_config)
    # Test that conditioning actually modifies L-tokens
    
def test_leakage_detection():
    detector = LeakageDetector()
    # Test detection of synthetic leakage scenarios
```

### Integration Tests
```python
def test_end_to_end_training():
    # Test complete training loop with walk-forward validation
    
def test_performance_monitoring():
    # Test logging and reporting functionality
    
def test_model_checkpointing():
    # Test save/load of model states during validation
```

## Performance Requirements

### Computational Efficiency
- **Training time**: <2 hours per walk-forward split
- **Validation time**: <30 minutes per test period
- **Memory usage**: <16GB GPU memory during training

### Model Performance
- **Parameter efficiency**: 26.5M-27.5M total parameters
- **Generalization**: Out-of-sample Sharpe ratio >1.5
- **Stability**: <20% performance variance across splits

## Integration Points

### Input: Market Data Pipeline
```python
# Expected input format from data pipeline
market_data = {
    'h_tokens': torch.tensor([...]),      # [B, T_h, D_h]
    'l_tokens': torch.tensor([...]),      # [B, T_l, D_l] 
    'market_features': {
        'regime_features': torch.tensor([...]),    # [B, 20]
        'pattern_features': torch.tensor([...])    # [B, 128]
    }
}
```

### Output: Trading System
```python
# Output format for trading system
predictions, metadata = hrm_adapter(
    market_data['h_tokens'], 
    market_data['l_tokens'],
    market_data['market_features']
)

# predictions: (outA, outB) - daily and intraday predictions
# metadata: conditioning info, attention weights, confidence scores
```

## Success Criteria

- [ ] HRM adapter integrates conditioning without parameter budget violation
- [ ] Walk-forward validation shows no temporal leakage  
- [ ] Model performance stable across validation splits
- [ ] Conditioning system usage properly logged and monitored
- [ ] Integration tests pass for full pipeline
- [ ] Performance metrics meet target thresholds