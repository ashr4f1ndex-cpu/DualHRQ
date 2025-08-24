"""
Statistical Validity Testing for DualHRQ Trading Systems.

Implements advanced statistical tests to validate that backtest results
are not due to data mining, overfitting, or statistical artifacts:

- Reality Check (RC): Tests null hypothesis that strategy has no skill
- Superior Predictive Ability (SPA): Adjusts for multiple testing bias  
- Data Snooping-Robust (DSR): Controls for extensive data mining

Critical for ensuring statistical significance of trading strategy results.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, chi2
from sklearn.model_selection import KFold
from sklearn.utils import resample
import torch

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of statistical validity tests."""
    REALITY_CHECK = "reality_check"
    SPA = "spa"  # Superior Predictive Ability
    DSR = "dsr"  # Data Snooping-Robust
    WHITE_BOOTSTRAP = "white_bootstrap"
    BLOCK_BOOTSTRAP = "block_bootstrap"


@dataclass
class StatisticalTest:
    """Statistical test configuration."""
    test_type: TestType
    n_bootstrap: int = 10000
    block_size: Optional[int] = None  # For block bootstrap
    significance_level: float = 0.05
    enable_studentization: bool = True
    min_observations: int = 100


@dataclass
class TestResult:
    """Results from statistical validity test."""
    test_type: TestType
    test_statistic: float
    p_value: float
    critical_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    bootstrap_distribution: np.ndarray
    interpretation: str
    metadata: Dict[str, Any]


@dataclass
class MultipleTestingResult:
    """Results from multiple testing correction."""
    n_tests: int
    raw_p_values: List[float]
    adjusted_p_values: List[float]
    significant_tests: List[int]
    family_wise_error_rate: float
    method: str


class RealityCheckTest:
    """
    Reality Check test for evaluating trading strategy performance.
    
    Tests the null hypothesis that the best strategy has no superior
    predictive ability compared to a benchmark.
    """
    
    def __init__(self, config: StatisticalTest):
        self.config = config
        self.bootstrap_stats = []
        
    def test_strategy_performance(self, strategy_returns: np.ndarray,
                                benchmark_returns: np.ndarray,
                                alternative_strategies: Optional[List[np.ndarray]] = None) -> TestResult:
        """
        Perform Reality Check test on strategy performance.
        
        Args:
            strategy_returns: Returns of strategy being tested
            benchmark_returns: Returns of benchmark (e.g., market index)
            alternative_strategies: Returns of alternative strategies
            
        Returns:
            TestResult with Reality Check statistics
        """
        if len(strategy_returns) < self.config.min_observations:
            raise ValueError(f"Insufficient data: {len(strategy_returns)} < {self.config.min_observations}")
        
        # Calculate excess returns relative to benchmark
        excess_returns = strategy_returns - benchmark_returns
        
        # Include alternative strategies if provided
        all_strategies = [excess_returns]
        if alternative_strategies:
            for alt_returns in alternative_strategies:
                alt_excess = alt_returns - benchmark_returns
                all_strategies.append(alt_excess)
        
        # Test statistic: maximum Sharpe ratio among all strategies
        test_statistic = self._calculate_test_statistic(all_strategies)
        
        # Bootstrap procedure
        bootstrap_stats = self._bootstrap_reality_check(all_strategies)
        
        # Calculate p-value
        p_value = np.mean(bootstrap_stats >= test_statistic)
        
        # Critical value at specified significance level
        critical_value = np.percentile(bootstrap_stats, (1 - self.config.significance_level) * 100)
        
        # Statistical significance
        is_significant = test_statistic > critical_value
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_stats, self.config.significance_level/2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - self.config.significance_level/2) * 100)
        
        interpretation = self._interpret_reality_check(is_significant, p_value, test_statistic)
        
        return TestResult(
            test_type=TestType.REALITY_CHECK,
            test_statistic=test_statistic,
            p_value=p_value,
            critical_value=critical_value,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            bootstrap_distribution=bootstrap_stats,
            interpretation=interpretation,
            metadata={
                'n_strategies': len(all_strategies),
                'n_observations': len(strategy_returns),
                'n_bootstrap': self.config.n_bootstrap,
                'excess_returns_mean': np.mean(excess_returns),
                'excess_returns_std': np.std(excess_returns)
            }
        )
    
    def _calculate_test_statistic(self, strategies: List[np.ndarray]) -> float:
        """Calculate test statistic (maximum Sharpe ratio)."""
        max_sharpe = -np.inf
        
        for returns in strategies:
            if np.std(returns) > 1e-8:  # Avoid division by zero
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                max_sharpe = max(max_sharpe, sharpe)
        
        return max_sharpe
    
    def _bootstrap_reality_check(self, strategies: List[np.ndarray]) -> np.ndarray:
        """Perform bootstrap resampling for Reality Check."""
        n_obs = len(strategies[0])
        bootstrap_stats = np.zeros(self.config.n_bootstrap)
        
        for i in range(self.config.n_bootstrap):
            # Resample indices
            boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)
            
            # Bootstrap each strategy
            boot_strategies = []
            for returns in strategies:
                boot_returns = returns[boot_indices]
                # Center the bootstrap sample under null hypothesis
                boot_returns = boot_returns - np.mean(boot_returns)
                boot_strategies.append(boot_returns)
            
            # Calculate bootstrap test statistic
            bootstrap_stats[i] = self._calculate_test_statistic(boot_strategies)
        
        return bootstrap_stats
    
    def _interpret_reality_check(self, is_significant: bool, p_value: float, 
                                test_statistic: float) -> str:
        """Interpret Reality Check test results."""
        if is_significant:
            return (f"REJECT null hypothesis (p={p_value:.4f}). "
                   f"Strategy shows statistically significant outperformance "
                   f"with Sharpe ratio {test_statistic:.2f}.")
        else:
            return (f"FAIL TO REJECT null hypothesis (p={p_value:.4f}). "
                   f"No evidence of superior predictive ability. "
                   f"Results may be due to data mining.")


class SPATest:
    """
    Superior Predictive Ability (SPA) test.
    
    More powerful than Reality Check, adjusts for multiple testing
    and provides better control of Type I errors.
    """
    
    def __init__(self, config: StatisticalTest):
        self.config = config
        
    def test_superior_ability(self, strategy_returns: np.ndarray,
                            benchmark_returns: np.ndarray,
                            alternative_strategies: List[np.ndarray]) -> TestResult:
        """
        Perform SPA test for superior predictive ability.
        
        Args:
            strategy_returns: Returns of best strategy
            benchmark_returns: Benchmark returns
            alternative_strategies: All alternative strategy returns
            
        Returns:
            TestResult with SPA test statistics
        """
        # Prepare relative performance measures
        all_strategies = [strategy_returns] + alternative_strategies
        relative_performance = self._calculate_relative_performance(all_strategies, benchmark_returns)
        
        # SPA test statistic
        test_statistic = self._calculate_spa_statistic(relative_performance)
        
        # Bootstrap under null hypothesis
        bootstrap_stats = self._bootstrap_spa(relative_performance)
        
        # Calculate p-value using studentized bootstrap
        if self.config.enable_studentization:
            p_value = self._studentized_p_value(test_statistic, bootstrap_stats, relative_performance)
        else:
            p_value = np.mean(bootstrap_stats >= test_statistic)
        
        # Critical value
        critical_value = np.percentile(bootstrap_stats, (1 - self.config.significance_level) * 100)
        
        is_significant = test_statistic > critical_value
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_stats, self.config.significance_level/2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - self.config.significance_level/2) * 100)
        
        interpretation = self._interpret_spa(is_significant, p_value, test_statistic)
        
        return TestResult(
            test_type=TestType.SPA,
            test_statistic=test_statistic,
            p_value=p_value,
            critical_value=critical_value,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            bootstrap_distribution=bootstrap_stats,
            interpretation=interpretation,
            metadata={
                'n_strategies': len(all_strategies),
                'n_observations': len(strategy_returns),
                'studentized': self.config.enable_studentization,
                'best_strategy_sharpe': np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            }
        )
    
    def _calculate_relative_performance(self, strategies: List[np.ndarray],
                                      benchmark: np.ndarray) -> np.ndarray:
        """Calculate relative performance matrix."""
        n_strategies = len(strategies)
        n_obs = len(strategies[0])
        
        performance = np.zeros((n_obs, n_strategies))
        
        for i, returns in enumerate(strategies):
            performance[:, i] = returns - benchmark
        
        return performance
    
    def _calculate_spa_statistic(self, performance: np.ndarray) -> float:
        """Calculate SPA test statistic."""
        # Mean performance of each strategy
        mean_performance = np.mean(performance, axis=0)
        
        # Standard errors
        std_errors = np.std(performance, axis=0) / np.sqrt(len(performance))
        
        # T-statistics for each strategy
        t_stats = np.where(std_errors > 1e-8, mean_performance / std_errors, 0)
        
        # SPA statistic is maximum t-statistic
        return np.max(t_stats)
    
    def _bootstrap_spa(self, performance: np.ndarray) -> np.ndarray:
        """Bootstrap procedure for SPA test."""
        n_obs, n_strategies = performance.shape
        bootstrap_stats = np.zeros(self.config.n_bootstrap)
        
        # Center performance under null hypothesis
        centered_performance = performance - np.mean(performance, axis=0)
        
        for i in range(self.config.n_bootstrap):
            # Bootstrap resample
            boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)
            boot_performance = centered_performance[boot_indices]
            
            # Calculate bootstrap SPA statistic
            bootstrap_stats[i] = self._calculate_spa_statistic(boot_performance)
        
        return bootstrap_stats
    
    def _studentized_p_value(self, test_stat: float, bootstrap_stats: np.ndarray,
                           performance: np.ndarray) -> float:
        """Calculate studentized p-value for better finite sample properties."""
        # This is a simplified version - full implementation would use
        # variance estimation and studentization
        return np.mean(bootstrap_stats >= test_stat)
    
    def _interpret_spa(self, is_significant: bool, p_value: float, 
                      test_statistic: float) -> str:
        """Interpret SPA test results."""
        if is_significant:
            return (f"SIGNIFICANT superior predictive ability (p={p_value:.4f}). "
                   f"Best strategy outperforms with t-statistic {test_statistic:.2f}, "
                   f"accounting for multiple testing bias.")
        else:
            return (f"NO EVIDENCE of superior predictive ability (p={p_value:.4f}). "
                   f"Results consistent with data mining across multiple strategies.")


class DSRTest:
    """
    Data Snooping-Robust (DSR) test for extensive data mining control.
    
    Controls for the scenario where researcher has tested many strategies
    and reports only the best performing one.
    """
    
    def __init__(self, config: StatisticalTest):
        self.config = config
        
    def test_data_snooping_robust(self, best_strategy_returns: np.ndarray,
                                benchmark_returns: np.ndarray,
                                n_strategies_tested: int,
                                selection_process_info: Optional[Dict] = None) -> TestResult:
        """
        Perform DSR test accounting for extensive strategy search.
        
        Args:
            best_strategy_returns: Returns of best strategy found
            benchmark_returns: Benchmark returns
            n_strategies_tested: Total number of strategies tested
            selection_process_info: Additional info about selection process
            
        Returns:
            TestResult with DSR statistics
        """
        excess_returns = best_strategy_returns - benchmark_returns
        
        # DSR test statistic with multiple testing adjustment
        test_statistic = self._calculate_dsr_statistic(excess_returns, n_strategies_tested)
        
        # Bootstrap with data snooping adjustment
        bootstrap_stats = self._bootstrap_dsr(excess_returns, n_strategies_tested)
        
        # Adjusted p-value for multiple testing
        p_value = self._calculate_adjusted_p_value(test_statistic, bootstrap_stats, n_strategies_tested)
        
        # Conservative critical value
        critical_value = np.percentile(bootstrap_stats, (1 - self.config.significance_level) * 100)
        
        is_significant = test_statistic > critical_value
        
        # Wide confidence interval reflecting uncertainty
        ci_lower = np.percentile(bootstrap_stats, self.config.significance_level/4 * 100)  # More conservative
        ci_upper = np.percentile(bootstrap_stats, (1 - self.config.significance_level/4) * 100)
        
        interpretation = self._interpret_dsr(is_significant, p_value, n_strategies_tested)
        
        return TestResult(
            test_type=TestType.DSR,
            test_statistic=test_statistic,
            p_value=p_value,
            critical_value=critical_value,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            bootstrap_distribution=bootstrap_stats,
            interpretation=interpretation,
            metadata={
                'n_strategies_tested': n_strategies_tested,
                'selection_bias_adjustment': True,
                'excess_return_mean': np.mean(excess_returns),
                'excess_return_sharpe': np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            }
        )
    
    def _calculate_dsr_statistic(self, excess_returns: np.ndarray, n_strategies: int) -> float:
        """Calculate DSR test statistic with multiple testing penalty."""
        # Basic Sharpe ratio
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Penalty for multiple testing (Bonferroni-style adjustment)
        penalty = np.log(n_strategies) / np.sqrt(len(excess_returns))
        
        # Adjusted test statistic
        return sharpe - penalty
    
    def _bootstrap_dsr(self, excess_returns: np.ndarray, n_strategies: int) -> np.ndarray:
        """Bootstrap for DSR test with selection bias simulation."""
        n_obs = len(excess_returns)
        bootstrap_stats = np.zeros(self.config.n_bootstrap)
        
        # Center under null hypothesis
        centered_returns = excess_returns - np.mean(excess_returns)
        
        for i in range(self.config.n_bootstrap):
            # Bootstrap resample
            boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)
            boot_returns = centered_returns[boot_indices]
            
            # Simulate selection of best strategy from multiple candidates
            max_sharpe = -np.inf
            for _ in range(min(n_strategies, 100)):  # Limit for computational efficiency
                # Generate alternative strategy (random permutation)
                perm_returns = np.random.permutation(boot_returns)
                if np.std(perm_returns) > 1e-8:
                    sharpe = np.mean(perm_returns) / np.std(perm_returns) * np.sqrt(252)
                    max_sharpe = max(max_sharpe, sharpe)
            
            # Apply DSR penalty
            penalty = np.log(n_strategies) / np.sqrt(n_obs)
            bootstrap_stats[i] = max_sharpe - penalty
        
        return bootstrap_stats
    
    def _calculate_adjusted_p_value(self, test_stat: float, bootstrap_stats: np.ndarray,
                                  n_strategies: int) -> float:
        """Calculate p-value adjusted for data snooping."""
        raw_p = np.mean(bootstrap_stats >= test_stat)
        
        # Bonferroni correction
        adjusted_p = min(raw_p * n_strategies, 1.0)
        
        return adjusted_p
    
    def _interpret_dsr(self, is_significant: bool, p_value: float, 
                      n_strategies: int) -> str:
        """Interpret DSR test results."""
        if is_significant:
            return (f"ROBUST significance after data snooping adjustment (p={p_value:.4f}). "
                   f"Performance remains significant even after testing {n_strategies} strategies. "
                   f"Strong evidence against pure data mining.")
        else:
            return (f"NOT SIGNIFICANT after data snooping correction (p={p_value:.4f}). "
                   f"Results likely due to selection bias from testing {n_strategies} strategies. "
                   f"Substantial data mining concerns.")


class StatisticalValidityFramework:
    """
    Comprehensive framework for statistical validity testing.
    
    Combines Reality Check, SPA, and DSR tests to provide robust
    validation of trading strategy performance.
    """
    
    def __init__(self):
        self.test_results: Dict[TestType, TestResult] = {}
        
    def comprehensive_validation(self, strategy_returns: np.ndarray,
                                benchmark_returns: np.ndarray,
                                alternative_strategies: List[np.ndarray] = None,
                                n_strategies_tested: int = 1,
                                config: StatisticalTest = None) -> Dict[TestType, TestResult]:
        """
        Perform comprehensive statistical validation.
        
        Args:
            strategy_returns: Returns of strategy to test
            benchmark_returns: Benchmark returns
            alternative_strategies: Alternative strategies for comparison
            n_strategies_tested: Total strategies tested (for DSR)
            config: Test configuration
            
        Returns:
            Dictionary of test results by test type
        """
        if config is None:
            config = StatisticalTest(TestType.REALITY_CHECK)
        
        results = {}
        
        # Reality Check test
        try:
            rc_test = RealityCheckTest(config)
            results[TestType.REALITY_CHECK] = rc_test.test_strategy_performance(
                strategy_returns, benchmark_returns, alternative_strategies
            )
            logger.info(f"Reality Check: p={results[TestType.REALITY_CHECK].p_value:.4f}")
        except Exception as e:
            logger.error(f"Reality Check test failed: {e}")
        
        # SPA test (if we have alternative strategies)
        if alternative_strategies and len(alternative_strategies) > 0:
            try:
                spa_test = SPATest(config)
                results[TestType.SPA] = spa_test.test_superior_ability(
                    strategy_returns, benchmark_returns, alternative_strategies
                )
                logger.info(f"SPA test: p={results[TestType.SPA].p_value:.4f}")
            except Exception as e:
                logger.error(f"SPA test failed: {e}")
        
        # DSR test (if multiple strategies were tested)
        if n_strategies_tested > 1:
            try:
                dsr_test = DSRTest(config)
                results[TestType.DSR] = dsr_test.test_data_snooping_robust(
                    strategy_returns, benchmark_returns, n_strategies_tested
                )
                logger.info(f"DSR test: p={results[TestType.DSR].p_value:.4f}")
            except Exception as e:
                logger.error(f"DSR test failed: {e}")
        
        self.test_results = results
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not self.test_results:
            return {"error": "No test results available"}
        
        report = {
            "summary": self._generate_summary(),
            "detailed_results": {},
            "recommendations": self._generate_recommendations(),
            "statistical_power": self._analyze_statistical_power(),
            "robustness_assessment": self._assess_robustness()
        }
        
        # Add detailed results for each test
        for test_type, result in self.test_results.items():
            report["detailed_results"][test_type.value] = {
                "test_statistic": result.test_statistic,
                "p_value": result.p_value,
                "critical_value": result.critical_value,
                "is_significant": result.is_significant,
                "confidence_interval": result.confidence_interval,
                "interpretation": result.interpretation,
                "metadata": result.metadata
            }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary of test results."""
        significant_tests = [test for test, result in self.test_results.items() 
                           if result.is_significant]
        
        total_tests = len(self.test_results)
        significant_count = len(significant_tests)
        
        # Overall conclusion
        if significant_count == total_tests and total_tests > 0:
            conclusion = "STRONG_EVIDENCE"
            confidence = "High"
        elif significant_count > total_tests // 2:
            conclusion = "MODERATE_EVIDENCE" 
            confidence = "Medium"
        elif significant_count > 0:
            conclusion = "WEAK_EVIDENCE"
            confidence = "Low"
        else:
            conclusion = "NO_EVIDENCE"
            confidence = "None"
        
        return {
            "overall_conclusion": conclusion,
            "confidence_level": confidence,
            "tests_performed": total_tests,
            "significant_tests": significant_count,
            "significant_test_types": [t.value for t in significant_tests],
            "min_p_value": min([r.p_value for r in self.test_results.values()]),
            "max_p_value": max([r.p_value for r in self.test_results.values()])
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        summary = self._generate_summary()
        
        if summary["overall_conclusion"] == "STRONG_EVIDENCE":
            recommendations.extend([
                "Results show strong statistical evidence of genuine skill",
                "Strategy performance unlikely due to data mining",
                "Consider deploying with appropriate risk management",
                "Monitor live performance for consistency with backtest"
            ])
        elif summary["overall_conclusion"] == "MODERATE_EVIDENCE":
            recommendations.extend([
                "Mixed evidence - some tests significant, others not",
                "Exercise caution in deployment decisions",
                "Consider additional out-of-sample testing",
                "Implement gradual position sizing increase"
            ])
        elif summary["overall_conclusion"] == "WEAK_EVIDENCE":
            recommendations.extend([
                "Limited evidence of genuine skill",
                "High probability results due to data mining",
                "Strongly recommend additional validation",
                "Consider paper trading before live deployment"
            ])
        else:
            recommendations.extend([
                "No statistical evidence of superior performance",
                "Results likely due to overfitting/data mining",
                "Do not deploy strategy in current form",
                "Revisit strategy development process"
            ])
        
        # Test-specific recommendations
        if TestType.DSR in self.test_results:
            dsr_result = self.test_results[TestType.DSR]
            if not dsr_result.is_significant:
                n_strategies = dsr_result.metadata.get('n_strategies_tested', 'many')
                recommendations.append(
                    f"DSR test failed with {n_strategies} "
                    "strategies tested - strong data mining concerns"
                )
        
        return recommendations
    
    def _analyze_statistical_power(self) -> Dict[str, Any]:
        """Analyze statistical power of performed tests."""
        power_analysis = {"estimated_power": {}}
        
        for test_type, result in self.test_results.items():
            n_obs = result.metadata.get('n_observations', 0)
            
            # Simple power estimation based on sample size and effect size
            if n_obs > 0:
                # Cohen's d approximation
                effect_size = abs(result.test_statistic) / np.sqrt(n_obs)
                
                # Rough power estimate (simplified)
                z_alpha = norm.ppf(1 - 0.05/2)  # Two-tailed test
                z_beta = norm.cdf(effect_size * np.sqrt(n_obs/2) - z_alpha)
                estimated_power = max(0.0, min(1.0, z_beta))
                
                power_analysis["estimated_power"][test_type.value] = {
                    "power": estimated_power,
                    "effect_size": effect_size,
                    "sample_size": n_obs,
                    "assessment": "High" if estimated_power > 0.8 else "Medium" if estimated_power > 0.5 else "Low"
                }
        
        return power_analysis
    
    def _assess_robustness(self) -> Dict[str, Any]:
        """Assess robustness of statistical conclusions."""
        robustness = {
            "cross_test_consistency": False,
            "bootstrap_stability": {},
            "sensitivity_analysis": {}
        }
        
        # Check consistency across tests
        significant_tests = [result.is_significant for result in self.test_results.values()]
        if len(set(significant_tests)) <= 1:  # All tests agree
            robustness["cross_test_consistency"] = True
        
        # Analyze bootstrap distributions for stability
        for test_type, result in self.test_results.items():
            if hasattr(result, 'bootstrap_distribution') and result.bootstrap_distribution is not None:
                boot_dist = result.bootstrap_distribution
                stability_metrics = {
                    "cv": np.std(boot_dist) / np.mean(boot_dist) if np.mean(boot_dist) != 0 else np.inf,
                    "skewness": stats.skew(boot_dist),
                    "kurtosis": stats.kurtosis(boot_dist)
                }
                robustness["bootstrap_stability"][test_type.value] = stability_metrics
        
        return robustness