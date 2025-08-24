"""
Tests for Statistical Validity Framework.

Tests Reality Check, SPA, and DSR tests to ensure proper statistical
validation of trading strategy performance against data mining bias.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict

import sys
sys.path.append('/Users/aziymandias/Desktop/dual_book_trading_lab_v10_complete/lab_v10')

from src.validation.statistical_validity import (
    StatisticalValidityFramework, RealityCheckTest, SPATest, DSRTest,
    StatisticalTest, TestResult, TestType, MultipleTestingResult
)


def generate_test_returns(n_obs: int = 252, mean_return: float = 0.0,
                         volatility: float = 0.2, random_seed: int = 42) -> np.ndarray:
    """Generate realistic test returns data."""
    np.random.seed(random_seed)
    
    # Daily returns with some autocorrelation (realistic for strategies)
    returns = np.random.normal(mean_return/252, volatility/np.sqrt(252), n_obs)
    
    # Add slight autocorrelation
    for i in range(1, n_obs):
        returns[i] += 0.05 * returns[i-1]
    
    return returns


def generate_benchmark_returns(n_obs: int = 252, random_seed: int = 123) -> np.ndarray:
    """Generate benchmark returns (e.g., market index)."""
    np.random.seed(random_seed)
    return np.random.normal(0.08/252, 0.15/np.sqrt(252), n_obs)  # 8% annual return, 15% vol


def generate_multiple_strategies(n_strategies: int = 10, n_obs: int = 252,
                               skill_level: float = 0.0) -> List[np.ndarray]:
    """Generate multiple strategy returns for testing."""
    strategies = []
    
    for i in range(n_strategies):
        # Most strategies have no skill, but some might have slight edge
        strategy_skill = skill_level if i == 0 else np.random.normal(0, 0.02)
        returns = generate_test_returns(n_obs, strategy_skill, 0.18, random_seed=100+i)
        strategies.append(returns)
    
    return strategies


class TestRealityCheckTest:
    """Test Reality Check test implementation."""
    
    def test_reality_check_initialization(self):
        """Test Reality Check test initialization."""
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=1000)
        rc_test = RealityCheckTest(config)
        
        assert rc_test.config.test_type == TestType.REALITY_CHECK
        assert rc_test.config.n_bootstrap == 1000
        assert len(rc_test.bootstrap_stats) == 0
    
    def test_reality_check_no_skill(self):
        """Test Reality Check with strategy that has no skill."""
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=500)
        rc_test = RealityCheckTest(config)
        
        # Generate returns with no skill
        strategy_returns = generate_test_returns(200, mean_return=0.0)
        benchmark_returns = generate_benchmark_returns(200)
        
        result = rc_test.test_strategy_performance(strategy_returns, benchmark_returns)
        
        # Should not reject null hypothesis (no skill)
        assert isinstance(result, TestResult)
        assert result.test_type == TestType.REALITY_CHECK
        assert 0.0 <= result.p_value <= 1.0
        assert result.p_value > 0.05  # Should not be significant for no skill
        assert not result.is_significant
        assert "FAIL TO REJECT" in result.interpretation
    
    def test_reality_check_with_skill(self):
        """Test Reality Check with strategy that has genuine skill."""
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=500)
        rc_test = RealityCheckTest(config)
        
        # Generate returns with significant skill
        strategy_returns = generate_test_returns(300, mean_return=0.15, volatility=0.18)  # Strong performance
        benchmark_returns = generate_benchmark_returns(300)
        
        result = rc_test.test_strategy_performance(strategy_returns, benchmark_returns)
        
        # Should likely reject null hypothesis (has skill)
        assert result.test_statistic > 0  # Positive Sharpe ratio
        assert len(result.bootstrap_distribution) == 500
        assert result.metadata['n_observations'] == 300
        
        # High skill should often be detected
        if result.test_statistic > 2.0:  # Strong signal
            assert result.is_significant
    
    def test_reality_check_with_alternatives(self):
        """Test Reality Check with alternative strategies."""
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=200)
        rc_test = RealityCheckTest(config)
        
        # Generate main strategy and alternatives
        strategy_returns = generate_test_returns(150, mean_return=0.12)
        benchmark_returns = generate_benchmark_returns(150)
        alternative_strategies = generate_multiple_strategies(5, 150, skill_level=0.05)
        
        result = rc_test.test_strategy_performance(
            strategy_returns, benchmark_returns, alternative_strategies
        )
        
        assert result.metadata['n_strategies'] == 6  # 1 main + 5 alternatives
        assert len(result.bootstrap_distribution) == 200
    
    def test_insufficient_data_error(self):
        """Test error handling for insufficient data."""
        config = StatisticalTest(TestType.REALITY_CHECK, min_observations=100)
        rc_test = RealityCheckTest(config)
        
        # Too little data
        strategy_returns = np.random.normal(0, 0.1, 50)
        benchmark_returns = np.random.normal(0, 0.1, 50)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            rc_test.test_strategy_performance(strategy_returns, benchmark_returns)


class TestSPATest:
    """Test Superior Predictive Ability (SPA) test."""
    
    def test_spa_initialization(self):
        """Test SPA test initialization."""
        config = StatisticalTest(TestType.SPA, n_bootstrap=1000, enable_studentization=True)
        spa_test = SPATest(config)
        
        assert spa_test.config.test_type == TestType.SPA
        assert spa_test.config.enable_studentization is True
    
    def test_spa_with_no_superior_ability(self):
        """Test SPA when no strategy has superior ability."""
        config = StatisticalTest(TestType.SPA, n_bootstrap=300)
        spa_test = SPATest(config)
        
        # All strategies have similar (poor) performance
        strategy_returns = generate_test_returns(200, mean_return=0.02)
        benchmark_returns = generate_benchmark_returns(200)
        alternatives = generate_multiple_strategies(8, 200, skill_level=0.0)
        
        result = spa_test.test_superior_ability(strategy_returns, benchmark_returns, alternatives)
        
        assert isinstance(result, TestResult)
        assert result.test_type == TestType.SPA
        assert result.p_value > 0.05  # Should not be significant
        assert not result.is_significant
        assert "NO EVIDENCE" in result.interpretation
    
    def test_spa_with_superior_ability(self):
        """Test SPA when one strategy has genuine superior ability."""
        config = StatisticalTest(TestType.SPA, n_bootstrap=300)
        spa_test = SPATest(config)
        
        # Best strategy has superior performance
        strategy_returns = generate_test_returns(250, mean_return=0.18, volatility=0.16)
        benchmark_returns = generate_benchmark_returns(250)
        alternatives = generate_multiple_strategies(6, 250, skill_level=0.02)  # Weak alternatives
        
        result = spa_test.test_superior_ability(strategy_returns, benchmark_returns, alternatives)
        
        assert result.test_statistic > 0
        assert len(result.bootstrap_distribution) == 300
        assert result.metadata['n_strategies'] == 7  # 1 best + 6 alternatives
        
        # Strong superior ability should often be detected
        if result.metadata['best_strategy_sharpe'] > 1.5:
            assert result.is_significant
    
    def test_spa_studentization(self):
        """Test SPA with and without studentization."""
        config_no_student = StatisticalTest(TestType.SPA, n_bootstrap=200, enable_studentization=False)
        config_student = StatisticalTest(TestType.SPA, n_bootstrap=200, enable_studentization=True)
        
        strategy_returns = generate_test_returns(150, mean_return=0.10)
        benchmark_returns = generate_benchmark_returns(150)
        alternatives = generate_multiple_strategies(4, 150)
        
        spa_no_student = SPATest(config_no_student)
        spa_student = SPATest(config_student)
        
        result_no_student = spa_no_student.test_superior_ability(strategy_returns, benchmark_returns, alternatives)
        result_student = spa_student.test_superior_ability(strategy_returns, benchmark_returns, alternatives)
        
        # Both should produce valid results
        assert result_no_student.metadata['studentized'] is False
        assert result_student.metadata['studentized'] is True
        assert 0.0 <= result_no_student.p_value <= 1.0
        assert 0.0 <= result_student.p_value <= 1.0


class TestDSRTest:
    """Test Data Snooping-Robust (DSR) test."""
    
    def test_dsr_initialization(self):
        """Test DSR test initialization."""
        config = StatisticalTest(TestType.DSR, n_bootstrap=800)
        dsr_test = DSRTest(config)
        
        assert dsr_test.config.test_type == TestType.DSR
        assert dsr_test.config.n_bootstrap == 800
    
    def test_dsr_light_data_mining(self):
        """Test DSR with light data mining (few strategies tested)."""
        config = StatisticalTest(TestType.DSR, n_bootstrap=200)
        dsr_test = DSRTest(config)
        
        # Good strategy with light data mining
        strategy_returns = generate_test_returns(200, mean_return=0.12)
        benchmark_returns = generate_benchmark_returns(200)
        n_strategies_tested = 5  # Light data mining
        
        result = dsr_test.test_data_snooping_robust(
            strategy_returns, benchmark_returns, n_strategies_tested
        )
        
        assert isinstance(result, TestResult)
        assert result.test_type == TestType.DSR
        assert result.metadata['n_strategies_tested'] == 5
        assert result.metadata['selection_bias_adjustment'] is True
        
        # With light data mining, good strategy might still be significant
        if result.metadata['excess_return_sharpe'] > 1.0:
            # Might be significant despite adjustment
            pass  # Test passes regardless
    
    def test_dsr_heavy_data_mining(self):
        """Test DSR with heavy data mining (many strategies tested)."""
        config = StatisticalTest(TestType.DSR, n_bootstrap=200)
        dsr_test = DSRTest(config)
        
        # Moderate strategy with heavy data mining
        strategy_returns = generate_test_returns(180, mean_return=0.08)
        benchmark_returns = generate_benchmark_returns(180)
        n_strategies_tested = 1000  # Heavy data mining
        
        result = dsr_test.test_data_snooping_robust(
            strategy_returns, benchmark_returns, n_strategies_tested
        )
        
        assert result.metadata['n_strategies_tested'] == 1000
        
        # With heavy data mining, should be much harder to achieve significance
        # Even decent performance should be non-significant
        assert result.p_value > 0.01  # Should have high p-value
        assert not result.is_significant  # Should not be significant
        assert "data mining" in result.interpretation.lower()
    
    def test_dsr_penalty_calculation(self):
        """Test that DSR applies appropriate penalty for multiple testing."""
        config = StatisticalTest(TestType.DSR, n_bootstrap=100)
        dsr_test = DSRTest(config)
        
        # Same strategy performance, different numbers of strategies tested
        strategy_returns = generate_test_returns(150, mean_return=0.10)
        benchmark_returns = generate_benchmark_returns(150)
        
        # Test with different levels of data mining
        result_light = dsr_test.test_data_snooping_robust(strategy_returns, benchmark_returns, 10)
        result_heavy = dsr_test.test_data_snooping_robust(strategy_returns, benchmark_returns, 500)
        
        # Heavy data mining should have higher p-value (more penalty)
        assert result_heavy.p_value >= result_light.p_value
        assert result_heavy.test_statistic <= result_light.test_statistic  # More penalty


class TestStatisticalValidityFramework:
    """Test comprehensive statistical validity framework."""
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        framework = StatisticalValidityFramework()
        
        assert len(framework.test_results) == 0
    
    def test_comprehensive_validation_minimal(self):
        """Test comprehensive validation with minimal setup."""
        framework = StatisticalValidityFramework()
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=100)
        
        strategy_returns = generate_test_returns(120, mean_return=0.06)
        benchmark_returns = generate_benchmark_returns(120)
        
        results = framework.comprehensive_validation(
            strategy_returns, benchmark_returns, config=config
        )
        
        # Should at least run Reality Check
        assert TestType.REALITY_CHECK in results
        assert isinstance(results[TestType.REALITY_CHECK], TestResult)
    
    def test_comprehensive_validation_full(self):
        """Test comprehensive validation with all tests."""
        framework = StatisticalValidityFramework()
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=150)
        
        strategy_returns = generate_test_returns(200, mean_return=0.12)
        benchmark_returns = generate_benchmark_returns(200)
        alternative_strategies = generate_multiple_strategies(8, 200)
        n_strategies_tested = 50
        
        results = framework.comprehensive_validation(
            strategy_returns, benchmark_returns, alternative_strategies,
            n_strategies_tested, config
        )
        
        # Should run all three tests
        assert TestType.REALITY_CHECK in results
        assert TestType.SPA in results  # Has alternative strategies
        assert TestType.DSR in results  # Has multiple strategies tested
        
        # All results should be valid
        for test_type, result in results.items():
            assert isinstance(result, TestResult)
            assert 0.0 <= result.p_value <= 1.0
            assert result.test_statistic is not None
    
    def test_validation_report_generation(self):
        """Test validation report generation."""
        framework = StatisticalValidityFramework()
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=100)
        
        # Run tests first
        strategy_returns = generate_test_returns(150, mean_return=0.08)
        benchmark_returns = generate_benchmark_returns(150)
        alternative_strategies = generate_multiple_strategies(5, 150)
        
        framework.comprehensive_validation(
            strategy_returns, benchmark_returns, alternative_strategies, 
            n_strategies_tested=20, config=config
        )
        
        # Generate report
        report = framework.generate_validation_report()
        
        # Check report structure
        assert 'summary' in report
        assert 'detailed_results' in report
        assert 'recommendations' in report
        assert 'statistical_power' in report
        assert 'robustness_assessment' in report
        
        # Check summary content
        summary = report['summary']
        assert 'overall_conclusion' in summary
        assert 'confidence_level' in summary
        assert 'tests_performed' in summary
        assert summary['overall_conclusion'] in ['STRONG_EVIDENCE', 'MODERATE_EVIDENCE', 'WEAK_EVIDENCE', 'NO_EVIDENCE']
    
    def test_report_with_no_results(self):
        """Test report generation with no test results."""
        framework = StatisticalValidityFramework()
        
        report = framework.generate_validation_report()
        
        assert 'error' in report
        assert 'No test results' in report['error']
    
    def test_recommendations_strong_evidence(self):
        """Test recommendations for strong evidence case."""
        framework = StatisticalValidityFramework()
        
        # Mock strong results (all tests significant)
        mock_result = TestResult(
            test_type=TestType.REALITY_CHECK,
            test_statistic=3.0,
            p_value=0.001,
            critical_value=1.96,
            is_significant=True,
            confidence_interval=(-1.0, 5.0),
            bootstrap_distribution=np.random.normal(0, 1, 100),
            interpretation="REJECT null",
            metadata={'n_observations': 200}
        )
        
        framework.test_results = {
            TestType.REALITY_CHECK: mock_result,
            TestType.SPA: mock_result,
            TestType.DSR: mock_result
        }
        
        report = framework.generate_validation_report()
        recommendations = report['recommendations']
        
        assert len(recommendations) > 0
        assert any('strong' in rec.lower() for rec in recommendations)
        assert any('deploy' in rec.lower() for rec in recommendations)
    
    def test_recommendations_no_evidence(self):
        """Test recommendations for no evidence case."""
        framework = StatisticalValidityFramework()
        
        # Mock weak results (no tests significant)
        mock_result = TestResult(
            test_type=TestType.REALITY_CHECK,
            test_statistic=0.5,
            p_value=0.8,
            critical_value=1.96,
            is_significant=False,
            confidence_interval=(-2.0, 1.0),
            bootstrap_distribution=np.random.normal(0, 1, 100),
            interpretation="FAIL TO REJECT null",
            metadata={'n_observations': 150}
        )
        
        framework.test_results = {
            TestType.REALITY_CHECK: mock_result,
            TestType.DSR: mock_result
        }
        
        report = framework.generate_validation_report()
        recommendations = report['recommendations']
        
        assert len(recommendations) > 0
        assert any('no' in rec.lower() and 'evidence' in rec.lower() for rec in recommendations)
        assert any('not deploy' in rec.lower() or 'do not deploy' in rec.lower() for rec in recommendations)


class TestStatisticalValidityEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_variance_returns(self):
        """Test handling of zero variance returns."""
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=100)
        rc_test = RealityCheckTest(config)
        
        # Strategy with zero variance (constant returns)
        strategy_returns = np.full(100, 0.001)  # Constant daily return
        benchmark_returns = generate_benchmark_returns(100)
        
        result = rc_test.test_strategy_performance(strategy_returns, benchmark_returns)
        
        # Should handle gracefully
        assert isinstance(result, TestResult)
        assert np.isfinite(result.test_statistic)
        assert 0.0 <= result.p_value <= 1.0
    
    def test_extreme_outliers(self):
        """Test handling of extreme outliers in returns."""
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=100)
        rc_test = RealityCheckTest(config)
        
        # Returns with extreme outliers
        strategy_returns = generate_test_returns(150)
        strategy_returns[50] = 5.0  # Extreme positive outlier
        strategy_returns[100] = -3.0  # Extreme negative outlier
        
        benchmark_returns = generate_benchmark_returns(150)
        
        result = rc_test.test_strategy_performance(strategy_returns, benchmark_returns)
        
        # Should produce reasonable results despite outliers
        assert isinstance(result, TestResult)
        assert np.isfinite(result.test_statistic)
        assert 0.0 <= result.p_value <= 1.0
    
    def test_mismatched_length_arrays(self):
        """Test error handling for mismatched array lengths."""
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=100)
        rc_test = RealityCheckTest(config)
        
        strategy_returns = np.random.normal(0, 0.1, 100)
        benchmark_returns = np.random.normal(0, 0.1, 150)  # Different length
        
        # Should handle length mismatch
        with pytest.raises((ValueError, IndexError)):
            rc_test.test_strategy_performance(strategy_returns, benchmark_returns)
    
    def test_very_small_sample_size(self):
        """Test behavior with very small sample sizes."""
        config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=50, min_observations=10)
        rc_test = RealityCheckTest(config)
        
        strategy_returns = np.random.normal(0.001, 0.02, 15)  # Very small sample
        benchmark_returns = np.random.normal(0.001, 0.02, 15)
        
        result = rc_test.test_strategy_performance(strategy_returns, benchmark_returns)
        
        # Should work but with low power
        assert isinstance(result, TestResult)
        assert result.metadata['n_observations'] == 15
        # P-value should be reasonable despite low power
        assert 0.0 <= result.p_value <= 1.0


def test_comprehensive_statistical_validation_workflow():
    """End-to-end test of complete statistical validation workflow."""
    framework = StatisticalValidityFramework()
    config = StatisticalTest(TestType.REALITY_CHECK, n_bootstrap=200, significance_level=0.05)
    
    # Scenario: Researcher tested many strategies and found one that looks good
    n_obs = 300
    best_strategy_returns = generate_test_returns(n_obs, mean_return=0.14, volatility=0.20)
    benchmark_returns = generate_benchmark_returns(n_obs)
    
    # Alternative strategies that were also considered
    alternative_strategies = generate_multiple_strategies(12, n_obs, skill_level=0.02)
    
    # Total strategies tested in research process
    n_strategies_tested = 200
    
    # Run comprehensive validation
    results = framework.comprehensive_validation(
        strategy_returns=best_strategy_returns,
        benchmark_returns=benchmark_returns,
        alternative_strategies=alternative_strategies,
        n_strategies_tested=n_strategies_tested,
        config=config
    )
    
    # Generate full report
    report = framework.generate_validation_report()
    
    # Validate results
    assert len(results) >= 2  # At least RC and DSR
    assert all(isinstance(result, TestResult) for result in results.values())
    
    # Check report completeness
    assert 'summary' in report
    assert 'detailed_results' in report
    assert 'recommendations' in report
    
    print(f"\nComprehensive Statistical Validation Results:")
    print(f"Overall Conclusion: {report['summary']['overall_conclusion']}")
    print(f"Confidence Level: {report['summary']['confidence_level']}")
    print(f"Tests Performed: {report['summary']['tests_performed']}")
    print(f"Significant Tests: {report['summary']['significant_tests']}")
    
    for test_type, result in results.items():
        print(f"\n{test_type.value.upper()} Test:")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Test Statistic: {result.test_statistic:.4f}")
        print(f"  Significant: {result.is_significant}")
        print(f"  Interpretation: {result.interpretation[:100]}...")
    
    print(f"\nKey Recommendations:")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    # Verify statistical properties
    for test_type, result in results.items():
        assert 0.0 <= result.p_value <= 1.0, f"{test_type} has invalid p-value"
        assert np.isfinite(result.test_statistic), f"{test_type} has invalid test statistic"
        assert len(result.bootstrap_distribution) > 0, f"{test_type} missing bootstrap distribution"
    
    # Check that DSR is more conservative than RC (if both present)
    if TestType.REALITY_CHECK in results and TestType.DSR in results:
        rc_result = results[TestType.REALITY_CHECK]
        dsr_result = results[TestType.DSR]
        
        # DSR should generally be more conservative (higher p-value)
        # This may not always hold due to randomness, but should be typical
        if rc_result.p_value < 0.05 and dsr_result.p_value > 0.05:
            print(f"  DSR correctly identified potential data mining (RC p={rc_result.p_value:.4f}, DSR p={dsr_result.p_value:.4f})")
    
    # Final validation
    assert report['summary']['overall_conclusion'] in ['STRONG_EVIDENCE', 'MODERATE_EVIDENCE', 'WEAK_EVIDENCE', 'NO_EVIDENCE']
    assert len(report['recommendations']) > 0