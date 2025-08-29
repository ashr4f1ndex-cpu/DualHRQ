# Validation Framework Specification

**Team:** ML/Data Science  
**Sprint:** Phase 3 (Weeks 15-20)  
**Dependencies:** HRM Integration, Pattern Library  
**SLO:** Statistical tests complete within 4 hours  

## Scope

Implement comprehensive validation framework including CPCV+embargo, statistical significance testing (RC/SPA/DSR), mutual information leakage detection, and deterministic reproducibility validation.

## Architecture Overview

```
Data → CPCV Splits → Model Training → Predictions → Statistical Tests → Validation Report
  ↓                    ↓               ↓              ↓               ↓
Leakage              Walk-Forward     Performance    RC/SPA/DSR     Pass/Fail
Detection            Validation       Metrics        Bootstrap       Decision
```

## Core Components

### 1. Combinatorial Purged Cross Validation (CPCV)

#### CPCV Implementation
```python
class CombinatorialPurgedCV:
    def __init__(self, n_splits: int = 5, purge_days: int = 14, 
                 embargo_days: int = 7, test_ratio: float = 0.2):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days  
        self.test_ratio = test_ratio
        
    def split(self, X: pd.DataFrame, y: pd.Series, 
             groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial purged cross-validation splits
        """
        dates = X.index.to_pydatetime()
        n_samples = len(X)
        test_size = int(n_samples * self.test_ratio)
        
        # Generate all possible test periods
        test_periods = self._generate_test_periods(dates, test_size)
        
        # Select n_splits periods using combinatorial approach
        selected_periods = self._select_test_periods(test_periods, self.n_splits)
        
        for test_period in selected_periods:
            train_idx, test_idx = self._create_purged_split(
                dates, test_period, n_samples
            )
            yield train_idx, test_idx
            
    def _generate_test_periods(self, dates: List[datetime], 
                              test_size: int) -> List[Tuple[datetime, datetime]]:
        """Generate all possible test periods"""
        periods = []
        for i in range(len(dates) - test_size + 1):
            start_date = dates[i]
            end_date = dates[i + test_size - 1]
            periods.append((start_date, end_date))
        return periods
        
    def _create_purged_split(self, dates: List[datetime], 
                           test_period: Tuple[datetime, datetime],
                           n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create train/test split with purge and embargo"""
        test_start, test_end = test_period
        
        # Define purge period (before test)
        purge_start = test_start - timedelta(days=self.purge_days)
        
        # Define embargo period (after test)  
        embargo_end = test_end + timedelta(days=self.embargo_days)
        
        train_idx = []
        test_idx = []
        
        for i, date in enumerate(dates):
            if test_start <= date <= test_end:
                test_idx.append(i)
            elif date < purge_start or date > embargo_end:
                train_idx.append(i)
            # Skip samples in purge/embargo periods
                
        return np.array(train_idx), np.array(test_idx)
```

#### Purged Group Time Series Split
```python
class PurgedGroupTimeSeriesSplit:
    """
    Time series cross-validation with purging for grouped data
    Prevents leakage when samples are not independent (e.g., same day)
    """
    def __init__(self, n_splits: int = 5, purge_days: int = 14):
        self.n_splits = n_splits
        self.purge_days = purge_days
        
    def split(self, X: pd.DataFrame, y: pd.Series, 
             groups: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged group time series splits
        
        Args:
            groups: Series indicating group membership (e.g., date)
        """
        unique_groups = sorted(groups.unique())
        n_groups = len(unique_groups)
        test_size = n_groups // self.n_splits
        
        for i in range(self.n_splits):
            # Define test groups
            test_start_idx = i * test_size
            test_end_idx = (i + 1) * test_size
            test_groups = unique_groups[test_start_idx:test_end_idx]
            
            # Define purge groups (before and after test)
            purge_before = set()
            purge_after = set()
            
            for group in test_groups:
                group_date = pd.to_datetime(group)
                
                # Add groups within purge window
                for other_group in unique_groups:
                    other_date = pd.to_datetime(other_group)
                    days_diff = abs((other_date - group_date).days)
                    
                    if days_diff <= self.purge_days:
                        if other_date < group_date:
                            purge_before.add(other_group)
                        elif other_date > group_date:
                            purge_after.add(other_group)
            
            # Create train/test indices
            test_idx = groups.isin(test_groups)
            train_idx = ~groups.isin(test_groups | purge_before | purge_after)
            
            yield np.where(train_idx)[0], np.where(test_idx)[0]
```

### 2. Statistical Significance Testing

#### Reality Check Bootstrap
```python
class RealityCheckBootstrap:
    """
    Reality Check test for data snooping bias
    Tests if model performance is significantly better than random
    """
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
    def test(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> Dict[str, float]:
        """
        Perform Reality Check test
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark strategy returns
            
        Returns:
            Dict with test statistics and p-values
        """
        n_samples = len(returns)
        
        # Compute test statistic (Sharpe ratio difference)
        strategy_sharpe = self._compute_sharpe(returns)
        benchmark_sharpe = self._compute_sharpe(benchmark_returns)
        observed_diff = strategy_sharpe - benchmark_sharpe
        
        # Bootstrap null distribution
        bootstrap_diffs = []
        
        for _ in range(self.n_bootstrap):
            # Resample returns under null hypothesis (no skill)
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_strategy_returns = returns[boot_indices]
            boot_benchmark_returns = benchmark_returns[boot_indices]
            
            # Add noise to break any spurious patterns
            boot_strategy_returns += np.random.normal(0, 0.001, n_samples)
            
            boot_strategy_sharpe = self._compute_sharpe(boot_strategy_returns)
            boot_benchmark_sharpe = self._compute_sharpe(boot_benchmark_returns)
            boot_diff = boot_strategy_sharpe - boot_benchmark_sharpe
            
            bootstrap_diffs.append(boot_diff)
            
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Compute p-value
        p_value = np.mean(bootstrap_diffs >= observed_diff)
        
        # Compute confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        return {
            'observed_sharpe_diff': observed_diff,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < (1 - self.confidence_level),
            'n_bootstrap': self.n_bootstrap
        }
        
    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """Compute annualized Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
```

#### Superior Predictive Ability (SPA) Test
```python
class SPATest:
    """
    Superior Predictive Ability test with multiple testing correction
    Tests if model significantly outperforms multiple benchmarks
    """
    def __init__(self, n_bootstrap: int = 1000, block_length: int = 20):
        self.n_bootstrap = n_bootstrap
        self.block_length = block_length  # For stationary bootstrap
        
    def test(self, model_returns: np.ndarray, 
            benchmark_returns_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        Perform SPA test against multiple benchmarks
        """
        n_models = len(benchmark_returns_list)
        n_samples = len(model_returns)
        
        # Compute performance differences
        performance_diffs = []
        for benchmark_returns in benchmark_returns_list:
            diff = model_returns - benchmark_returns
            performance_diffs.append(diff)
            
        performance_diffs = np.column_stack(performance_diffs)  # [T, K]
        
        # Compute test statistic (max of mean differences)
        mean_diffs = np.mean(performance_diffs, axis=0)
        max_diff = np.max(mean_diffs)
        
        # Stationary bootstrap for null distribution
        bootstrap_max_stats = []
        
        for _ in range(self.n_bootstrap):
            boot_diffs = self._stationary_bootstrap(performance_diffs)
            boot_mean_diffs = np.mean(boot_diffs, axis=0)
            
            # Center at zero under null hypothesis
            boot_mean_diffs_centered = boot_mean_diffs - mean_diffs
            boot_max_stat = np.max(boot_mean_diffs_centered)
            bootstrap_max_stats.append(boot_max_stat)
            
        bootstrap_max_stats = np.array(bootstrap_max_stats)
        
        # Compute p-value
        p_value = np.mean(bootstrap_max_stats >= max_diff)
        
        # Individual model p-values (not corrected)
        individual_p_values = []
        for i in range(n_models):
            individual_stats = [boot_diffs[i] for boot_diffs in 
                              [np.mean(self._stationary_bootstrap(performance_diffs), axis=0) 
                               for _ in range(self.n_bootstrap)]]
            individual_p = np.mean(np.array(individual_stats) >= mean_diffs[i])
            individual_p_values.append(individual_p)
            
        return {
            'spa_statistic': max_diff,
            'spa_p_value': p_value,
            'individual_p_values': individual_p_values,
            'mean_performance_diffs': mean_diffs,
            'is_significant': p_value < 0.10,  # Standard threshold for SPA
            'n_benchmarks': n_models
        }
        
    def _stationary_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """Generate stationary bootstrap sample"""
        n_samples, n_series = data.shape
        boot_sample = np.zeros_like(data)
        
        i = 0
        while i < n_samples:
            # Random starting point
            start = np.random.randint(0, n_samples)
            
            # Random block length (geometric distribution)
            block_len = min(
                np.random.geometric(1 / self.block_length),
                n_samples - i
            )
            
            # Copy block
            for j in range(block_len):
                boot_sample[i] = data[(start + j) % n_samples]
                i += 1
                if i >= n_samples:
                    break
                    
        return boot_sample
```

#### Data Snooping-Robust (DSR) Statistics
```python
class DSRStatistics:
    """
    Data Snooping-Robust statistics with multiple testing correction
    """
    def __init__(self, fdr_method: str = 'benjamini_hochberg'):
        self.fdr_method = fdr_method
        
    def compute_dsr_metrics(self, returns_dict: Dict[str, np.ndarray], 
                           alpha: float = 0.05) -> Dict[str, Any]:
        """
        Compute DSR-corrected performance metrics
        
        Args:
            returns_dict: Dictionary of {model_name: returns_array}
            alpha: Significance level for multiple testing
        """
        model_names = list(returns_dict.keys())
        n_models = len(model_names)
        
        # Compute raw performance metrics
        raw_metrics = {}
        for name, returns in returns_dict.items():
            raw_metrics[name] = {
                'sharpe_ratio': self._compute_sharpe(returns),
                'max_drawdown': self._compute_max_drawdown(returns),
                'calmar_ratio': self._compute_calmar_ratio(returns),
                'sortino_ratio': self._compute_sortino_ratio(returns)
            }
            
        # Compute p-values for each metric using bootstrap
        p_values = {}
        for metric_name in ['sharpe_ratio', 'calmar_ratio', 'sortino_ratio']:
            metric_p_values = []
            for model_name in model_names:
                p_val = self._bootstrap_test_metric(
                    returns_dict[model_name], metric_name
                )
                metric_p_values.append(p_val)
            p_values[metric_name] = metric_p_values
            
        # Apply multiple testing correction
        corrected_results = {}
        for metric_name, p_vals in p_values.items():
            corrected_p_vals = self._fdr_correction(p_vals, alpha)
            
            corrected_results[metric_name] = {
                'raw_p_values': p_vals,
                'corrected_p_values': corrected_p_vals,
                'significant_models': [
                    model_names[i] for i, p in enumerate(corrected_p_vals) 
                    if p < alpha
                ],
                'family_wise_error_rate': min(corrected_p_vals) * n_models
            }
            
        return {
            'raw_metrics': raw_metrics,
            'corrected_results': corrected_results,
            'n_models_tested': n_models,
            'significance_level': alpha
        }
        
    def _fdr_correction(self, p_values: List[float], alpha: float) -> List[float]:
        """Apply False Discovery Rate correction"""
        p_values = np.array(p_values)
        n = len(p_values)
        
        if self.fdr_method == 'benjamini_hochberg':
            # Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values)
            sorted_p_values = p_values[sorted_indices]
            
            corrected_p_values = np.zeros(n)
            for i in range(n-1, -1, -1):
                if i == n-1:
                    corrected_p_values[sorted_indices[i]] = sorted_p_values[i]
                else:
                    corrected_p_values[sorted_indices[i]] = min(
                        sorted_p_values[i] * n / (i + 1),
                        corrected_p_values[sorted_indices[i+1]]
                    )
                    
            return corrected_p_values.tolist()
        else:
            raise ValueError(f"Unknown FDR method: {self.fdr_method}")
```

### 3. Leakage Detection Framework

#### Mutual Information Tests
```python
class MutualInformationTester:
    """Test for information leakage using mutual information"""
    
    def __init__(self, mi_threshold: float = 0.1):
        self.mi_threshold = mi_threshold  # bits
        
    def test_feature_leakage(self, features: np.ndarray, 
                           puzzle_ids: np.ndarray) -> Dict[str, float]:
        """
        Test if features leak information about puzzle_id
        """
        from sklearn.feature_selection import mutual_info_classif
        
        # Convert puzzle_ids to integers if needed
        unique_ids = np.unique(puzzle_ids)
        id_to_int = {id_val: i for i, id_val in enumerate(unique_ids)}
        puzzle_ids_int = np.array([id_to_int[id_val] for id_val in puzzle_ids])
        
        # Compute mutual information for each feature
        mi_scores = mutual_info_classif(features, puzzle_ids_int, random_state=42)
        
        results = {
            'mi_scores': mi_scores.tolist(),
            'max_mi_score': float(np.max(mi_scores)),
            'mean_mi_score': float(np.mean(mi_scores)),
            'features_above_threshold': int(np.sum(mi_scores > self.mi_threshold)),
            'leakage_detected': bool(np.any(mi_scores > self.mi_threshold)),
            'leakage_threshold': self.mi_threshold
        }
        
        return results
        
    def test_prediction_leakage(self, predictions: np.ndarray,
                              puzzle_ids: np.ndarray) -> Dict[str, float]:
        """
        Test if predictions leak puzzle_id information
        """
        # Use predictions as features
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            
        return self.test_feature_leakage(predictions, puzzle_ids)
```

#### Shuffle Tests
```python
class ShuffleTest:
    """Test model robustness to label shuffling"""
    
    def __init__(self, n_shuffles: int = 10):
        self.n_shuffles = n_shuffles
        
    def test_label_shuffling(self, model_func: Callable, 
                           X: np.ndarray, y: np.ndarray,
                           train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, float]:
        """
        Test model performance degradation under label shuffling
        
        Args:
            model_func: Function that trains and evaluates model
            X, y: Features and labels
            train_idx, test_idx: Train/test split indices
        """
        # Original performance
        original_score = model_func(X, y, train_idx, test_idx)
        
        # Shuffled performance  
        shuffled_scores = []
        for i in range(self.n_shuffles):
            # Shuffle labels within training set only
            y_shuffled = y.copy()
            y_shuffled[train_idx] = np.random.permutation(y_shuffled[train_idx])
            
            shuffled_score = model_func(X, y_shuffled, train_idx, test_idx)
            shuffled_scores.append(shuffled_score)
            
        shuffled_scores = np.array(shuffled_scores)
        
        # Compute degradation metrics
        mean_shuffled_score = np.mean(shuffled_scores)
        performance_drop = original_score - mean_shuffled_score
        relative_drop = performance_drop / abs(original_score) if original_score != 0 else 0
        
        return {
            'original_score': float(original_score),
            'mean_shuffled_score': float(mean_shuffled_score),
            'shuffled_scores': shuffled_scores.tolist(),
            'performance_drop': float(performance_drop),
            'relative_performance_drop': float(relative_drop),
            'degradation_sufficient': bool(relative_drop > 0.5),  # >50% drop expected
            'n_shuffles': self.n_shuffles
        }
```

### 4. Deterministic Reproducibility

#### Reproducibility Validator
```python
class ReproducibilityValidator:
    """Validate deterministic reproducibility across runs"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    def validate_reproducibility(self, model_func: Callable,
                               data: Dict[str, Any],
                               n_runs: int = 3) -> Dict[str, Any]:
        """
        Validate that model produces identical results across runs
        """
        results = []
        
        for run_id in range(n_runs):
            # Set all random seeds
            self._set_seeds(42 + run_id)
            
            # Run model
            result = model_func(data)
            results.append(result)
            
        # Check consistency across runs
        consistency_results = self._check_consistency(results)
        
        return {
            'n_runs': n_runs,
            'all_results': results,
            'consistency_check': consistency_results,
            'is_reproducible': consistency_results['is_consistent'],
            'tolerance': self.tolerance
        }
        
    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Additional determinism for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _check_consistency(self, results: List[Any]) -> Dict[str, Any]:
        """Check consistency of results across runs"""
        if len(results) < 2:
            return {'is_consistent': True, 'message': 'Only one run'}
            
        # Compare first result with all others
        base_result = results[0]
        
        for i, result in enumerate(results[1:], 1):
            if not self._results_equal(base_result, result):
                return {
                    'is_consistent': False,
                    'failed_at_run': i,
                    'message': f'Run {i} differs from run 0'
                }
                
        return {'is_consistent': True, 'message': 'All runs identical'}
        
    def _results_equal(self, result1: Any, result2: Any) -> bool:
        """Check if two results are equal within tolerance"""
        if isinstance(result1, np.ndarray) and isinstance(result2, np.ndarray):
            return np.allclose(result1, result2, atol=self.tolerance)
        elif isinstance(result1, torch.Tensor) and isinstance(result2, torch.Tensor):
            return torch.allclose(result1, result2, atol=self.tolerance)
        elif isinstance(result1, dict) and isinstance(result2, dict):
            if set(result1.keys()) != set(result2.keys()):
                return False
            return all(self._results_equal(result1[k], result2[k]) 
                      for k in result1.keys())
        else:
            return result1 == result2
```

## Configuration

```yaml
validation:
  cpcv:
    n_splits: 5
    purge_days: 14           # 2-week purge
    embargo_days: 7          # 1-week embargo  
    test_ratio: 0.2
    
  statistical_tests:
    reality_check:
      n_bootstrap: 1000
      confidence_level: 0.95
      
    spa_test:
      n_bootstrap: 1000
      block_length: 20
      significance_threshold: 0.10
      
    dsr_statistics:
      fdr_method: "benjamini_hochberg"
      significance_level: 0.05
      
  leakage_detection:
    mi_threshold: 0.1        # bits
    shuffle_tests:
      n_shuffles: 10
      min_degradation: 0.5   # 50% performance drop required
      
  reproducibility:
    n_validation_runs: 3
    tolerance: 1e-6
    require_identical: true
```

## Testing Strategy

### Unit Tests (Write First)
```python
def test_cpcv_no_leakage():
    # Test CPCV prevents temporal leakage
    
def test_reality_check_bootstrap():
    # Test reality check with known random data
    
def test_spa_multiple_testing():
    # Test SPA with multiple benchmarks
    
def test_mutual_information_detection():
    # Test MI detection with synthetic leakage
    
def test_reproducibility_validation():
    # Test deterministic reproduction
```

### Integration Tests
```python
def test_full_validation_pipeline():
    # Test complete validation framework
    
def test_statistical_power():
    # Test statistical tests have sufficient power
    
def test_validation_performance():
    # Test validation completes within SLO
```

## Performance Requirements

### Computational Efficiency
- **CPCV validation**: <2 hours for 5-fold CV
- **Statistical tests**: <1 hour for RC/SPA/DSR
- **Leakage detection**: <30 minutes for MI/shuffle tests
- **Total validation time**: <4 hours end-to-end

### Statistical Power
- **Minimum effect size detection**: 0.1 Sharpe ratio units
- **Type I error rate**: <5% false positive rate
- **Type II error rate**: <20% false negative rate  

## Success Criteria

- [ ] CPCV shows no temporal leakage (purge + embargo working)
- [ ] Statistical tests achieve target power and error rates
- [ ] MI tests detect leakage when present (sensitivity >95%)
- [ ] Shuffle tests show >50% performance degradation
- [ ] Reproducibility tests achieve bit-exact results
- [ ] Full validation pipeline completes within 4-hour SLO
- [ ] Integration with HRM adapter functional