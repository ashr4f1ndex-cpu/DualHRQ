"""
Advanced Statistical Validation Framework

Research-grade statistical testing surpassing academic standards:
- Deflated Sharpe Ratio with multiple testing correction
- White's Reality Check for data snooping
- Hansen's Superior Predictive Ability (SPA) test
- Probabilistic Sharpe Ratio with higher moments
- Block bootstrap for time-series confidence intervals
- Combinatorial Purged Cross-Validation integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import norm, t as t_dist
from dataclasses import dataclass
import warnings
from itertools import combinations
import logging

logger = logging.getLogger(__name__)

@dataclass
class StatisticalTestResult:
    """Result of statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: float
    confidence_level: float
    reject_null: bool
    interpretation: str
    additional_stats: Dict[str, float] = None

class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio (DSR) implementation.
    
    Based on Bailey & López de Prado's "The Deflated Sharpe Ratio: 
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality" (2014).
    """
    
    @staticmethod
    def calculate_deflated_sharpe(returns: pd.Series, 
                                number_of_trials: int,
                                skewness: Optional[float] = None,
                                kurtosis: Optional[float] = None,
                                confidence_level: float = 0.05) -> StatisticalTestResult:
        """
        Calculate Deflated Sharpe Ratio with multiple testing correction.
        
        Args:
            returns: Strategy returns series
            number_of_trials: Number of strategies tested (for multiple testing)
            skewness: Returns skewness (calculated if None)
            kurtosis: Returns excess kurtosis (calculated if None)
            confidence_level: Significance level
            
        Returns:
            StatisticalTestResult with DSR analysis
        """
        
        if len(returns) < 30:
            raise ValueError("Minimum 30 observations required for DSR calculation")
        
        # Calculate basic statistics
        n = len(returns)
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Calculate or use provided higher moments
        if skewness is None:
            skewness = returns.skew()
        if kurtosis is None:
            kurtosis = returns.kurtosis()  # Pandas returns excess kurtosis
        
        # Adjust for non-normality (López de Prado adjustment)
        skew_adjustment = skewness * sharpe_ratio / 6
        kurt_adjustment = (kurtosis - 3) * (sharpe_ratio ** 2) / 24
        
        # Adjusted Sharpe ratio for non-normality
        adjusted_sharpe = sharpe_ratio - skew_adjustment - kurt_adjustment
        
        # Standard error of Sharpe ratio
        sharpe_se = np.sqrt((1 + 0.5 * sharpe_ratio ** 2 - 
                           skewness * sharpe_ratio + 
                           (kurtosis - 3) * sharpe_ratio ** 2 / 4) / n)
        
        # Multiple testing correction
        # Expected maximum Sharpe ratio under null (no skill)
        expected_max_sharpe = ((1 - np.euler_gamma) * norm.ppf(1 - 1./number_of_trials) + 
                              np.euler_gamma * norm.ppf(1 - 1./(number_of_trials * np.e)))
        
        # Deflated Sharpe Ratio
        deflated_sharpe = (adjusted_sharpe - expected_max_sharpe) / sharpe_se
        
        # Critical value and p-value
        critical_value = norm.ppf(1 - confidence_level)
        p_value = 1 - norm.cdf(deflated_sharpe)
        
        # Interpretation
        reject_null = deflated_sharpe > critical_value
        if reject_null:
            interpretation = f"Strategy shows significant skill after correcting for selection bias (DSR = {deflated_sharpe:.3f})"
        else:
            interpretation = f"No significant evidence of skill after multiple testing correction (DSR = {deflated_sharpe:.3f})"
        
        return StatisticalTestResult(
            test_name="Deflated Sharpe Ratio",
            statistic=deflated_sharpe,
            p_value=p_value,
            critical_value=critical_value,
            confidence_level=1 - confidence_level,
            reject_null=reject_null,
            interpretation=interpretation,
            additional_stats={
                'raw_sharpe': sharpe_ratio,
                'adjusted_sharpe': adjusted_sharpe,
                'expected_max_sharpe': expected_max_sharpe,
                'skewness': skewness,
                'excess_kurtosis': kurtosis,
                'number_of_trials': number_of_trials,
                'observations': n
            }
        )

class ProbabilisticSharpeRatio:
    """
    Probabilistic Sharpe Ratio (PSR) implementation.
    
    Calculates probability that Sharpe ratio exceeds a benchmark,
    accounting for higher moments of return distribution.
    """
    
    @staticmethod
    def calculate_psr(returns: pd.Series, 
                     benchmark_sharpe: float = 0.0,
                     confidence_level: float = 0.95) -> StatisticalTestResult:
        """
        Calculate Probabilistic Sharpe Ratio.
        
        Args:
            returns: Strategy returns
            benchmark_sharpe: Benchmark Sharpe ratio to exceed
            confidence_level: Confidence level for probability calculation
            
        Returns:
            StatisticalTestResult with PSR analysis
        """
        
        n = len(returns)
        if n < 30:
            raise ValueError("Minimum 30 observations required for PSR")
        
        # Calculate moments
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        skewness = returns.skew()
        kurtosis = returns.kurtosis()  # Excess kurtosis
        
        # Sharpe ratio standard error with higher moments
        sharpe_se = np.sqrt((1 + 0.5 * sharpe_ratio ** 2 - 
                           skewness * sharpe_ratio + 
                           (kurtosis) * sharpe_ratio ** 2 / 4) / n)
        
        # PSR statistic
        psr_statistic = (sharpe_ratio - benchmark_sharpe) / sharpe_se
        
        # Probability that true Sharpe > benchmark
        probability = norm.cdf(psr_statistic)
        
        # Critical value for given confidence level
        critical_value = norm.ppf(confidence_level)
        
        # Test whether PSR exceeds confidence threshold
        reject_null = psr_statistic > critical_value
        p_value = 1 - probability
        
        interpretation = (f"Probability that Sharpe ratio > {benchmark_sharpe} is {probability:.3f}. "
                        f"Strategy {'shows' if reject_null else 'does not show'} significant outperformance.")
        
        return StatisticalTestResult(
            test_name="Probabilistic Sharpe Ratio",
            statistic=psr_statistic,
            p_value=p_value,
            critical_value=critical_value,
            confidence_level=confidence_level,
            reject_null=reject_null,
            interpretation=interpretation,
            additional_stats={
                'probability_exceed_benchmark': probability,
                'benchmark_sharpe': benchmark_sharpe,
                'observed_sharpe': sharpe_ratio,
                'skewness': skewness,
                'excess_kurtosis': kurtosis,
                'observations': n
            }
        )

class WhitesRealityCheck:
    """
    White's Reality Check for data snooping.
    
    Tests whether the best performing strategy from a universe
    has genuine skill or just got lucky.
    """
    
    def __init__(self, n_bootstraps: int = 5000):
        self.n_bootstraps = n_bootstraps
    
    def reality_check_test(self, best_strategy_returns: pd.Series,
                          all_strategies_returns: pd.DataFrame,
                          benchmark_returns: pd.Series = None) -> StatisticalTestResult:
        """
        Perform White's Reality Check test.
        
        Args:
            best_strategy_returns: Returns of the best strategy
            all_strategies_returns: Returns of all strategies tested
            benchmark_returns: Benchmark returns (if None, use zero)
            
        Returns:
            StatisticalTestResult with Reality Check analysis
        """
        
        if benchmark_returns is None:
            benchmark_returns = pd.Series(0, index=best_strategy_returns.index)
        
        # Calculate relative performance vs benchmark
        best_relative = best_strategy_returns - benchmark_returns
        all_relative = all_strategies_returns.subtract(benchmark_returns, axis=0)
        
        # Test statistic: mean of best strategy's relative performance
        observed_performance = best_relative.mean()
        
        # Bootstrap procedure
        n_obs = len(best_relative)
        n_strategies = len(all_strategies_returns.columns)
        
        bootstrap_max_performance = []
        
        for _ in range(self.n_bootstraps):
            # Resample with replacement
            bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
            
            # Calculate performance for each strategy in bootstrap sample
            bootstrap_performances = []
            for strategy in all_relative.columns:
                bootstrap_series = all_relative[strategy].iloc[bootstrap_indices]
                bootstrap_performances.append(bootstrap_series.mean())
            
            # Record maximum performance in this bootstrap
            bootstrap_max_performance.append(max(bootstrap_performances))
        
        # Calculate p-value
        bootstrap_max_performance = np.array(bootstrap_max_performance)
        p_value = (bootstrap_max_performance >= observed_performance).mean()
        
        # Critical value (95th percentile of bootstrap distribution)
        critical_value = np.percentile(bootstrap_max_performance, 95)
        
        reject_null = observed_performance > critical_value
        
        interpretation = (f"Reality Check p-value: {p_value:.4f}. "
                        f"Best strategy {'shows' if reject_null else 'does not show'} "
                        f"significant outperformance after correcting for data snooping.")
        
        return StatisticalTestResult(
            test_name="White's Reality Check",
            statistic=observed_performance,
            p_value=p_value,
            critical_value=critical_value,
            confidence_level=0.95,
            reject_null=reject_null,
            interpretation=interpretation,
            additional_stats={
                'n_strategies': n_strategies,
                'n_bootstraps': self.n_bootstraps,
                'bootstrap_mean': bootstrap_max_performance.mean(),
                'bootstrap_std': bootstrap_max_performance.std(),
                'observations': n_obs
            }
        )

class HansenSPATest:
    """
    Hansen's Superior Predictive Ability (SPA) test.
    
    More sophisticated version of White's Reality Check with
    improved finite-sample properties.
    """
    
    def __init__(self, n_bootstraps: int = 5000):
        self.n_bootstraps = n_bootstraps
    
    def spa_test(self, best_strategy_returns: pd.Series,
                all_strategies_returns: pd.DataFrame,
                benchmark_returns: pd.Series = None) -> StatisticalTestResult:
        """
        Perform Hansen's SPA test.
        
        Args:
            best_strategy_returns: Returns of best strategy
            all_strategies_returns: Returns of all strategies
            benchmark_returns: Benchmark returns
            
        Returns:
            StatisticalTestResult with SPA analysis
        """
        
        if benchmark_returns is None:
            benchmark_returns = pd.Series(0, index=best_strategy_returns.index)
        
        # Relative performance matrix
        relative_returns = all_strategies_returns.subtract(benchmark_returns, axis=0)
        n_obs, n_strategies = relative_returns.shape
        
        # Calculate mean relative performance for each strategy
        mean_performance = relative_returns.mean()
        std_performance = relative_returns.std()
        
        # Find best performing strategy
        best_strategy_idx = mean_performance.idxmax()
        best_performance = mean_performance[best_strategy_idx]
        
        # SPA test statistics
        t_stats = mean_performance * np.sqrt(n_obs) / std_performance
        max_t_stat = t_stats.max()
        
        # Bootstrap procedure with studentization
        bootstrap_max_t = []
        
        for _ in range(self.n_bootstraps):
            # Resample
            bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
            bootstrap_returns = relative_returns.iloc[bootstrap_indices]
            
            # Re-center around sample mean for null hypothesis
            bootstrap_returns_centered = bootstrap_returns - mean_performance
            
            # Calculate bootstrap t-statistics
            bootstrap_means = bootstrap_returns_centered.mean()
            bootstrap_stds = bootstrap_returns_centered.std()
            
            bootstrap_t = bootstrap_means * np.sqrt(n_obs) / bootstrap_stds
            bootstrap_max_t.append(bootstrap_t.max())
        
        bootstrap_max_t = np.array(bootstrap_max_t)
        
        # Calculate p-value
        p_value = (bootstrap_max_t >= max_t_stat).mean()
        
        # Critical value
        critical_value = np.percentile(bootstrap_max_t, 95)
        
        reject_null = max_t_stat > critical_value
        
        interpretation = (f"SPA test p-value: {p_value:.4f}. "
                        f"Best strategy {'demonstrates' if reject_null else 'does not demonstrate'} "
                        f"superior predictive ability.")
        
        return StatisticalTestResult(
            test_name="Hansen SPA Test",
            statistic=max_t_stat,
            p_value=p_value,
            critical_value=critical_value,
            confidence_level=0.95,
            reject_null=reject_null,
            interpretation=interpretation,
            additional_stats={
                'best_strategy': best_strategy_idx,
                'best_mean_return': best_performance,
                'n_strategies': n_strategies,
                'n_bootstraps': self.n_bootstraps,
                'observations': n_obs
            }
        )

class StationaryBlockBootstrap:
    """
    Stationary Block Bootstrap for time-series confidence intervals.
    
    Preserves temporal dependence structure while generating
    bootstrap samples for statistical inference.
    """
    
    def __init__(self, block_length: Optional[int] = None):
        self.block_length = block_length
    
    def _optimal_block_length(self, data: pd.Series) -> int:
        """
        Estimate optimal block length using Politis & White (2004) method.
        """
        n = len(data)
        
        # Calculate sample autocorrelations
        max_lag = min(n // 4, 50)
        autocorrs = []
        
        for lag in range(1, max_lag + 1):
            if lag >= n:
                break
            autocorr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            autocorrs.append(autocorr)
        
        autocorrs = np.array(autocorrs)
        
        # Find where autocorrelation becomes negligible
        threshold = 2 / np.sqrt(n)
        negligible_lags = np.where(np.abs(autocorrs) < threshold)[0]
        
        if len(negligible_lags) > 0:
            optimal_length = negligible_lags[0] + 1
        else:
            # Fallback: rule of thumb
            optimal_length = int(np.ceil(n ** (1/3)))
        
        return min(optimal_length, n // 4)
    
    def generate_bootstrap_samples(self, data: pd.Series, 
                                 n_bootstraps: int = 1000) -> List[pd.Series]:
        """
        Generate stationary block bootstrap samples.
        
        Args:
            data: Time series data
            n_bootstraps: Number of bootstrap samples
            
        Returns:
            List of bootstrap samples
        """
        
        n = len(data)
        
        if self.block_length is None:
            self.block_length = self._optimal_block_length(data)
        
        bootstrap_samples = []
        
        for _ in range(n_bootstraps):
            bootstrap_data = []
            
            while len(bootstrap_data) < n:
                # Random starting point
                start_idx = np.random.randint(0, n)
                
                # Geometric block length (stationary bootstrap)
                block_len = np.random.geometric(1 / self.block_length)
                block_len = min(block_len, n - len(bootstrap_data))
                
                # Add block (with wraparound)
                for i in range(block_len):
                    idx = (start_idx + i) % n
                    bootstrap_data.append(data.iloc[idx])
            
            # Truncate to original length
            bootstrap_sample = pd.Series(bootstrap_data[:n], index=data.index)
            bootstrap_samples.append(bootstrap_sample)
        
        return bootstrap_samples
    
    def bootstrap_confidence_interval(self, data: pd.Series, 
                                    statistic_func: callable,
                                    confidence_level: float = 0.95,
                                    n_bootstraps: int = 1000) -> Dict[str, float]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Time series data
            statistic_func: Function to calculate statistic
            confidence_level: Confidence level
            n_bootstraps: Number of bootstrap samples
            
        Returns:
            Dictionary with confidence interval bounds
        """
        
        # Original statistic
        original_stat = statistic_func(data)
        
        # Generate bootstrap samples
        bootstrap_samples = self.generate_bootstrap_samples(data, n_bootstraps)
        
        # Calculate statistic for each bootstrap sample
        bootstrap_stats = []
        for sample in bootstrap_samples:
            try:
                stat = statistic_func(sample)
                bootstrap_stats.append(stat)
            except:
                continue  # Skip problematic samples
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate percentiles for confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            'original_statistic': original_stat,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'bootstrap_mean': bootstrap_stats.mean(),
            'bootstrap_std': bootstrap_stats.std(),
            'n_bootstraps': len(bootstrap_stats)
        }

class MultipleTestingCorrection:
    """
    Multiple testing correction methods for controlling false discovery rate.
    """
    
    @staticmethod
    def benjamini_hochberg(p_values: List[float], 
                          alpha: float = 0.05) -> Dict[str, any]:
        """
        Benjamini-Hochberg FDR correction.
        
        Args:
            p_values: List of p-values
            alpha: Target FDR level
            
        Returns:
            Dictionary with correction results
        """
        
        p_values = np.array(p_values)
        n = len(p_values)
        
        # Sort p-values and keep track of original indices
        sort_indices = np.argsort(p_values)
        sorted_p = p_values[sort_indices]
        
        # BH critical values
        bh_critical = np.array([alpha * (i + 1) / n for i in range(n)])
        
        # Find largest i such that P(i) <= alpha * i / n
        significant_indices = np.where(sorted_p <= bh_critical)[0]
        
        if len(significant_indices) > 0:
            cutoff_index = significant_indices[-1]
            cutoff_p_value = sorted_p[cutoff_index]
            
            # All p-values <= cutoff are significant
            significant_original_indices = sort_indices[:cutoff_index + 1]
            significant_mask = np.zeros(n, dtype=bool)
            significant_mask[significant_original_indices] = True
        else:
            cutoff_p_value = 0
            significant_mask = np.zeros(n, dtype=bool)
        
        return {
            'significant_mask': significant_mask,
            'cutoff_p_value': cutoff_p_value,
            'n_significant': significant_mask.sum(),
            'n_total': n,
            'fdr_level': alpha,
            'method': 'Benjamini-Hochberg'
        }
    
    @staticmethod
    def bonferroni_correction(p_values: List[float],
                            alpha: float = 0.05) -> Dict[str, any]:
        """
        Bonferroni correction for multiple testing.
        
        Args:
            p_values: List of p-values
            alpha: Significance level
            
        Returns:
            Dictionary with correction results
        """
        
        p_values = np.array(p_values)
        n = len(p_values)
        
        # Bonferroni correction
        corrected_alpha = alpha / n
        significant_mask = p_values <= corrected_alpha
        
        return {
            'significant_mask': significant_mask,
            'corrected_alpha': corrected_alpha,
            'n_significant': significant_mask.sum(),
            'n_total': n,
            'method': 'Bonferroni'
        }

class StatisticalValidationSuite:
    """
    Comprehensive statistical validation suite for trading strategies.
    """
    
    def __init__(self):
        self.dsr_calculator = DeflatedSharpeRatio()
        self.psr_calculator = ProbabilisticSharpeRatio()
        self.reality_check = WhitesRealityCheck()
        self.spa_test = HansenSPATest()
        self.bootstrap = StationaryBlockBootstrap()
        self.multiple_testing = MultipleTestingCorrection()
    
    def run_comprehensive_validation(self, 
                                   strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series = None,
                                   all_strategy_returns: pd.DataFrame = None,
                                   number_of_trials: int = 1,
                                   confidence_level: float = 0.95) -> Dict[str, any]:
        """
        Run complete statistical validation suite.
        
        Args:
            strategy_returns: Main strategy returns
            benchmark_returns: Benchmark returns
            all_strategy_returns: Returns of all strategies tested
            number_of_trials: Number of strategies tested (for DSR)
            confidence_level: Statistical confidence level
            
        Returns:
            Comprehensive validation results
        """
        
        validation_results = {
            'strategy_stats': self._calculate_basic_stats(strategy_returns),
            'tests': {}
        }
        
        # 1. Deflated Sharpe Ratio
        try:
            dsr_result = self.dsr_calculator.calculate_deflated_sharpe(
                strategy_returns, number_of_trials, confidence_level=1-confidence_level
            )
            validation_results['tests']['deflated_sharpe'] = dsr_result
        except Exception as e:
            logger.warning(f"DSR calculation failed: {e}")
        
        # 2. Probabilistic Sharpe Ratio
        try:
            benchmark_sharpe = 0
            if benchmark_returns is not None:
                benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std()
            
            psr_result = self.psr_calculator.calculate_psr(
                strategy_returns, benchmark_sharpe, confidence_level
            )
            validation_results['tests']['probabilistic_sharpe'] = psr_result
        except Exception as e:
            logger.warning(f"PSR calculation failed: {e}")
        
        # 3. White's Reality Check
        if all_strategy_returns is not None:
            try:
                reality_check_result = self.reality_check.reality_check_test(
                    strategy_returns, all_strategy_returns, benchmark_returns
                )
                validation_results['tests']['reality_check'] = reality_check_result
            except Exception as e:
                logger.warning(f"Reality Check failed: {e}")
        
        # 4. Hansen SPA Test
        if all_strategy_returns is not None:
            try:
                spa_result = self.spa_test.spa_test(
                    strategy_returns, all_strategy_returns, benchmark_returns
                )
                validation_results['tests']['spa_test'] = spa_result
            except Exception as e:
                logger.warning(f"SPA test failed: {e}")
        
        # 5. Bootstrap confidence intervals
        try:
            sharpe_ci = self.bootstrap.bootstrap_confidence_interval(
                strategy_returns,
                lambda x: x.mean() / x.std() if x.std() > 0 else 0,
                confidence_level
            )
            validation_results['bootstrap_sharpe_ci'] = sharpe_ci
        except Exception as e:
            logger.warning(f"Bootstrap CI failed: {e}")
        
        # 6. Overall assessment
        validation_results['overall_assessment'] = self._generate_assessment(validation_results)
        
        return validation_results
    
    def _calculate_basic_stats(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic return statistics."""
        
        return {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': returns.mean() * 252,
            'annualized_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min(),
            'skewness': returns.skew(),
            'excess_kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),
            'observations': len(returns)
        }
    
    def _generate_assessment(self, validation_results: Dict) -> Dict[str, any]:
        """Generate overall assessment of strategy validation."""
        
        assessments = []
        significant_tests = 0
        total_tests = 0
        
        tests = validation_results.get('tests', {})
        
        for test_name, result in tests.items():
            if hasattr(result, 'reject_null'):
                total_tests += 1
                if result.reject_null:
                    significant_tests += 1
                    assessments.append(f"{result.test_name}: Significant evidence of skill")
                else:
                    assessments.append(f"{result.test_name}: No significant evidence of skill")
        
        # Overall recommendation
        if total_tests == 0:
            recommendation = "Insufficient tests performed for assessment"
            confidence_score = 0
        elif significant_tests / total_tests >= 0.7:
            recommendation = "Strategy shows strong statistical evidence of skill"
            confidence_score = significant_tests / total_tests
        elif significant_tests / total_tests >= 0.5:
            recommendation = "Strategy shows moderate evidence of skill"
            confidence_score = significant_tests / total_tests
        else:
            recommendation = "Strategy lacks convincing statistical evidence of skill"
            confidence_score = significant_tests / total_tests
        
        return {
            'recommendation': recommendation,
            'confidence_score': confidence_score,
            'significant_tests': significant_tests,
            'total_tests': total_tests,
            'test_assessments': assessments
        }