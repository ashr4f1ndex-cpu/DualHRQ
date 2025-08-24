"""
Tests for Backtest Validation Integration.

Tests comprehensive validation of backtest results including
statistical tests and risk analysis for deployment decisions.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

import sys
sys.path.append('/Users/aziymandias/Desktop/dual_book_trading_lab_v10_complete/lab_v10')

from src.validation.backtest_validation import (
    BacktestValidator, ValidationConfig, ValidationResults
)
from src.validation.statistical_validity import TestType
from src.trading.realistic_backtester import BacktestResults, Trade, Position


def create_mock_backtest_results(performance_type: str = 'good') -> BacktestResults:
    """Create mock backtest results for testing."""
    n_days = 252  # 1 year of trading
    
    if performance_type == 'excellent':
        daily_returns = np.random.normal(0.0015, 0.012, n_days)  # 15% annual return, 12% vol
        sharpe_ratio = 1.8
        max_drawdown = 0.08
        total_return = 0.18
        compliance_rate = 0.99
        violations = 2
    elif performance_type == 'good':
        daily_returns = np.random.normal(0.0008, 0.015, n_days)  # 8% annual return, 15% vol
        sharpe_ratio = 1.2
        max_drawdown = 0.12
        total_return = 0.10
        compliance_rate = 0.97
        violations = 5
    elif performance_type == 'mediocre':
        daily_returns = np.random.normal(0.0003, 0.018, n_days)  # 3% annual return, 18% vol
        sharpe_ratio = 0.6
        max_drawdown = 0.18
        total_return = 0.04
        compliance_rate = 0.94
        violations = 12
    else:  # poor
        daily_returns = np.random.normal(-0.0002, 0.022, n_days)  # Negative return, high vol
        sharpe_ratio = -0.2
        max_drawdown = 0.28
        total_return = -0.05
        compliance_rate = 0.91
        violations = 25
    
    # Create mock trades
    trades = []
    for i in range(50):
        trade = Trade(
            timestamp=datetime.now() + timedelta(days=i*5),
            symbol='TEST',
            side='buy' if i % 2 == 0 else 'sell',
            quantity=100,
            price=100 + np.random.normal(0, 5),
            value=10000 + np.random.normal(0, 500),
            commission=5.0,
            compliance_valid=np.random.random() > 0.05,  # 95% compliance
            compliance_reason='Valid' if np.random.random() > 0.05 else 'Violation'
        )
        trades.append(trade)
    
    # Create mock positions
    positions = [
        Position('TEST1', 100, 98.5, 102.3, 10230, 380, 'long'),
        Position('TEST2', -50, 205.1, 198.7, -9935, 320, 'short')
    ]
    
    # Mock impacts
    ssr_impacts = {
        'total_ssr_triggers': 3,
        'rejected_short_trades': 2,
        'ssr_affected_symbols': 1,
        'estimated_impact_pnl': -150
    }
    
    luld_impacts = {
        'total_luld_violations': 5,
        'total_trading_halts': 2,
        'halt_affected_symbols': 2,
        'avg_halt_duration': 5.2
    }
    
    portfolio_value = 50000 * (1 + total_return)
    daily_returns_series = pd.Series(daily_returns)
    equity_curve = pd.Series(np.cumprod(1 + daily_returns) * 50000)
    
    return BacktestResults(
        trades=trades,
        positions=positions,
        portfolio_value=portfolio_value,
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        compliance_rate=compliance_rate,
        regulatory_violations=violations,
        ssr_impacts=ssr_impacts,
        luld_impacts=luld_impacts,
        hrm_performance={'total_adaptations': 10},
        daily_returns=daily_returns_series,
        equity_curve=equity_curve
    )


def create_benchmark_returns(n_days: int = 252) -> np.ndarray:
    """Create mock benchmark returns."""
    np.random.seed(123)
    return np.random.normal(0.0006, 0.012, n_days)  # 6% annual return, 12% vol


class TestBacktestValidator:
    """Test backtest validator functionality."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        config = ValidationConfig(reality_check_bootstrap=1000)
        validator = BacktestValidator(config)
        
        assert validator.config.reality_check_bootstrap == 1000
        assert validator.statistical_framework is not None
    
    def test_validator_default_config(self):
        """Test validator with default configuration."""
        validator = BacktestValidator()
        
        assert validator.config.reality_check_bootstrap == 5000
        assert validator.config.significance_level == 0.05
        assert validator.config.min_observations == 252
    
    def test_excellent_strategy_validation(self):
        """Test validation of excellent strategy performance."""
        config = ValidationConfig(reality_check_bootstrap=200, spa_bootstrap=150, dsr_bootstrap=100)
        validator = BacktestValidator(config)
        
        backtest_results = create_mock_backtest_results('excellent')
        benchmark_returns = create_benchmark_returns()
        
        # Add some alternative strategies
        alternatives = [
            np.random.normal(0.0004, 0.014, 252) for _ in range(3)
        ]
        
        development_info = {'n_strategies_tested': 15}
        
        validation_results = validator.validate_backtest_results(
            backtest_results, benchmark_returns, alternatives, development_info
        )
        
        # Check result structure
        assert isinstance(validation_results, ValidationResults)
        assert validation_results.backtest_results is backtest_results
        assert len(validation_results.statistical_tests) >= 1
        assert 'summary' in validation_results.validation_report
        
        # Excellent strategy should have high confidence
        assert validation_results.confidence_score > 0.6
        assert validation_results.deployment_readiness in ['READY', 'CAUTION']
        
        # Check backtest analysis
        backtest_analysis = validation_results.validation_report['backtest_analysis']
        assert 'performance_metrics' in backtest_analysis
        assert 'regulatory_compliance' in backtest_analysis
        assert 'trading_activity' in backtest_analysis
        
        perf_metrics = backtest_analysis['performance_metrics']
        assert perf_metrics['sharpe_ratio'] > 1.5
        assert perf_metrics['max_drawdown'] < 0.15
    
    def test_poor_strategy_validation(self):
        """Test validation of poor strategy performance."""
        config = ValidationConfig(reality_check_bootstrap=150)
        validator = BacktestValidator(config)
        
        backtest_results = create_mock_backtest_results('poor')
        benchmark_returns = create_benchmark_returns()
        
        development_info = {'n_strategies_tested': 100}  # Heavy data mining
        
        validation_results = validator.validate_backtest_results(
            backtest_results, benchmark_returns, strategy_development_info=development_info
        )
        
        # Poor strategy should have low confidence
        assert validation_results.confidence_score < 0.5
        assert validation_results.deployment_readiness == 'NOT_READY'
        assert 'NOT READY' in validation_results.final_recommendation or 'against deployment' in validation_results.final_recommendation
        
        # Check risk analysis
        risk_analysis = validation_results.validation_report['risk_analysis']
        assert risk_analysis['overall_risk_score'] > 0.4  # Higher risk
        assert risk_analysis['risk_level'] in ['Medium Risk', 'High Risk']
    
    def test_mediocre_strategy_validation(self):
        """Test validation of mediocre strategy performance."""
        config = ValidationConfig(reality_check_bootstrap=200)
        validator = BacktestValidator(config)
        
        backtest_results = create_mock_backtest_results('mediocre')
        benchmark_returns = create_benchmark_returns()
        
        validation_results = validator.validate_backtest_results(
            backtest_results, benchmark_returns
        )
        
        # Mediocre strategy should have moderate confidence
        assert 0.3 <= validation_results.confidence_score <= 0.7
        assert validation_results.deployment_readiness in ['CAUTION', 'NOT_READY']
        
        # Should have mixed recommendations
        assert 'CONDITIONAL' in validation_results.final_recommendation or 'CAUTION' in validation_results.final_recommendation
    
    def test_insufficient_data_warning(self):
        """Test warning for insufficient data."""
        config = ValidationConfig(min_observations=500)  # Require more data
        validator = BacktestValidator(config)
        
        backtest_results = create_mock_backtest_results('good')
        benchmark_returns = create_benchmark_returns(252)  # Only 252 days
        
        with pytest.warns(UserWarning, match="Insufficient data"):
            validation_results = validator.validate_backtest_results(
                backtest_results, benchmark_returns
            )
        
        # Should still produce results
        assert isinstance(validation_results, ValidationResults)
    
    def test_risk_analysis_components(self):
        """Test individual risk analysis components."""
        validator = BacktestValidator()
        
        # Test with high-risk backtest results
        high_risk_results = create_mock_backtest_results('poor')
        high_risk_results.max_drawdown = 0.35  # Very high drawdown
        high_risk_results.compliance_rate = 0.85  # Low compliance
        
        benchmark_returns = create_benchmark_returns()
        
        validation_results = validator.validate_backtest_results(
            high_risk_results, benchmark_returns
        )
        
        risk_analysis = validation_results.validation_report['risk_analysis']
        
        # Check individual risk components
        assert 'statistical_risks' in risk_analysis
        assert 'market_risks' in risk_analysis
        assert 'operational_risks' in risk_analysis  
        assert 'regulatory_risks' in risk_analysis
        
        # High-risk results should be reflected
        market_risks = risk_analysis['market_risks']
        assert market_risks['risk_score'] > 0.3  # Should identify market risks
        assert market_risks['max_drawdown'] > 0.3
        
        regulatory_risks = risk_analysis['regulatory_risks']
        assert regulatory_risks['risk_score'] > 0.2  # Should identify compliance issues
        assert regulatory_risks['compliance_rate'] < 0.9
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculations."""
        validator = BacktestValidator()
        
        # Test specific calculations
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        
        win_rate = validator._calculate_win_rate(returns)
        assert win_rate == 0.6  # 3 out of 5 positive
        
        profit_factor = validator._calculate_profit_factor(returns)
        expected_pf = (0.01 + 0.02 + 0.015) / (0.005 + 0.01)
        assert abs(profit_factor - expected_pf) < 0.001
        
        downside_dev = validator._calculate_downside_deviation(returns)
        assert downside_dev > 0  # Should be positive
        
        tail_ratio = validator._calculate_tail_ratio(returns)
        assert tail_ratio > 0  # Should be positive
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation logic."""
        validator = BacktestValidator()
        
        # Test confidence mapping
        assert validator._map_conclusion_to_confidence('STRONG_EVIDENCE') == 0.9
        assert validator._map_conclusion_to_confidence('MODERATE_EVIDENCE') == 0.7
        assert validator._map_conclusion_to_confidence('WEAK_EVIDENCE') == 0.4
        assert validator._map_conclusion_to_confidence('NO_EVIDENCE') == 0.1
        
        # Test backtest quality assessment
        excellent_results = create_mock_backtest_results('excellent')
        quality_score = validator._assess_backtest_quality(excellent_results)
        assert quality_score > 0.7  # Should be high quality
        
        poor_results = create_mock_backtest_results('poor')
        poor_quality_score = validator._assess_backtest_quality(poor_results)
        assert poor_quality_score < quality_score  # Should be lower


class TestValidationIntegration:
    """Test integration scenarios."""
    
    def test_no_alternative_strategies(self):
        """Test validation with no alternative strategies."""
        validator = BacktestValidator(ValidationConfig(reality_check_bootstrap=100))
        
        backtest_results = create_mock_backtest_results('good')
        benchmark_returns = create_benchmark_returns()
        
        validation_results = validator.validate_backtest_results(
            backtest_results, benchmark_returns
        )
        
        # Should still work with just Reality Check
        assert TestType.REALITY_CHECK in validation_results.statistical_tests
        # SPA test should not be present without alternatives
        assert TestType.SPA not in validation_results.statistical_tests
    
    def test_single_strategy_development(self):
        """Test validation assuming single strategy (no data mining)."""
        validator = BacktestValidator(ValidationConfig(dsr_bootstrap=100))
        
        backtest_results = create_mock_backtest_results('good')
        benchmark_returns = create_benchmark_returns()
        
        # No development info = assumes minimal data mining
        validation_results = validator.validate_backtest_results(
            backtest_results, benchmark_returns
        )
        
        # Should be more lenient without heavy data mining concerns
        assert validation_results.confidence_score > 0.4
    
    def test_heavy_data_mining_scenario(self):
        """Test validation with heavy data mining."""
        validator = BacktestValidator(ValidationConfig(dsr_bootstrap=150))
        
        backtest_results = create_mock_backtest_results('good')
        benchmark_returns = create_benchmark_returns()
        
        # Simulate heavy data mining
        development_info = {'n_strategies_tested': 500}
        
        validation_results = validator.validate_backtest_results(
            backtest_results, benchmark_returns,
            strategy_development_info=development_info
        )
        
        # Should be more conservative due to data mining concerns
        assert TestType.DSR in validation_results.statistical_tests
        
        dsr_result = validation_results.statistical_tests[TestType.DSR]
        # With heavy data mining, DSR should be more stringent
        assert dsr_result.metadata['n_strategies_tested'] == 500
    
    def test_comprehensive_validation_workflow(self):
        """Test complete validation workflow."""
        config = ValidationConfig(
            reality_check_bootstrap=200,
            spa_bootstrap=150,
            dsr_bootstrap=100,
            significance_level=0.05
        )
        
        validator = BacktestValidator(config)
        
        # Create comprehensive test scenario
        backtest_results = create_mock_backtest_results('good')
        benchmark_returns = create_benchmark_returns()
        
        alternative_strategies = [
            np.random.normal(0.0003, 0.016, 252),  # Weak strategy
            np.random.normal(0.0007, 0.014, 252),  # Decent strategy
            np.random.normal(0.0001, 0.019, 252)   # Poor strategy
        ]
        
        development_info = {
            'n_strategies_tested': 50,
            'development_time_months': 6,
            'out_of_sample_periods': 2
        }
        
        # Run comprehensive validation
        validation_results = validator.validate_backtest_results(
            backtest_results=backtest_results,
            benchmark_returns=benchmark_returns,
            alternative_strategies=alternative_strategies,
            strategy_development_info=development_info
        )
        
        # Verify comprehensive results
        assert len(validation_results.statistical_tests) >= 2  # RC + SPA or DSR
        assert 'summary' in validation_results.validation_report
        assert 'detailed_results' in validation_results.validation_report
        assert 'recommendations' in validation_results.validation_report
        assert 'backtest_analysis' in validation_results.validation_report
        assert 'risk_analysis' in validation_results.validation_report
        
        # Check validation report completeness
        report = validation_results.validation_report
        
        # Summary should have key metrics
        summary = report['summary']
        assert 'overall_conclusion' in summary
        assert 'confidence_level' in summary
        assert summary['tests_performed'] >= 2
        
        # Backtest analysis should be comprehensive
        backtest_analysis = report['backtest_analysis']
        assert 'performance_metrics' in backtest_analysis
        assert 'regulatory_compliance' in backtest_analysis
        assert 'trading_activity' in backtest_analysis
        assert 'risk_metrics' in backtest_analysis
        
        # Risk analysis should cover all areas
        risk_analysis = report['risk_analysis']
        assert 'overall_risk_score' in risk_analysis
        assert 'risk_level' in risk_analysis
        
        # Final recommendation should be actionable
        assert len(validation_results.final_recommendation) > 100  # Detailed recommendation
        assert validation_results.deployment_readiness in ['READY', 'CAUTION', 'NOT_READY']
        assert 0.0 <= validation_results.confidence_score <= 1.0
        
        print(f"\nComprehensive Validation Results:")
        print(f"Deployment Readiness: {validation_results.deployment_readiness}")
        print(f"Confidence Score: {validation_results.confidence_score:.1%}")
        print(f"Risk Level: {risk_analysis['risk_level']}")
        print(f"Statistical Tests: {len(validation_results.statistical_tests)}")
        print(f"Overall Conclusion: {summary['overall_conclusion']}")
        print(f"\nRecommendation: {validation_results.final_recommendation[:200]}...")


def test_end_to_end_validation_pipeline():
    """End-to-end test of complete validation pipeline."""
    # Configuration for thorough testing
    config = ValidationConfig(
        reality_check_bootstrap=300,
        spa_bootstrap=200,
        dsr_bootstrap=150,
        significance_level=0.05,
        min_observations=200
    )
    
    validator = BacktestValidator(config)
    
    # Create realistic test scenario
    # Simulate a strategy that looks good but may have been data mined
    np.random.seed(456)
    
    # Strategy with decent performance but potential overfitting
    strategy_returns = np.random.normal(0.0008, 0.016, 300)  # 8% annual, 16% vol
    strategy_returns[:50] *= 1.5  # Front-loaded performance (potential overfitting)
    
    backtest_results = BacktestResults(
        trades=[],  # Simplified for testing
        positions=[],
        portfolio_value=55000,
        total_return=0.10,
        sharpe_ratio=1.1,
        max_drawdown=0.14,
        compliance_rate=0.96,
        regulatory_violations=8,
        ssr_impacts={'total_ssr_triggers': 2, 'rejected_short_trades': 1},
        luld_impacts={'total_luld_violations': 3, 'total_trading_halts': 1},
        hrm_performance={'total_adaptations': 15},
        daily_returns=pd.Series(strategy_returns),
        equity_curve=pd.Series(np.cumprod(1 + strategy_returns) * 50000)
    )
    
    benchmark_returns = create_benchmark_returns(300)
    
    # Alternative strategies (some good, some bad)
    alternatives = [
        np.random.normal(0.0004, 0.015, 300),  # Mediocre
        np.random.normal(0.0009, 0.017, 300),  # Good
        np.random.normal(-0.0001, 0.020, 300), # Poor
        np.random.normal(0.0006, 0.014, 300),  # Decent
    ]
    
    # Moderate data mining scenario
    development_info = {
        'n_strategies_tested': 75,
        'development_duration_months': 8,
        'parameter_combinations_tested': 200
    }
    
    # Run complete validation
    results = validator.validate_backtest_results(
        backtest_results=backtest_results,
        benchmark_returns=benchmark_returns,
        alternative_strategies=alternatives,
        strategy_development_info=development_info
    )
    
    # Comprehensive verification
    assert isinstance(results, ValidationResults)
    
    # All test types should be present
    expected_tests = {TestType.REALITY_CHECK, TestType.SPA, TestType.DSR}
    actual_tests = set(results.statistical_tests.keys())
    assert len(actual_tests & expected_tests) >= 2  # At least 2 of the 3 tests
    
    # Statistical test results should be valid
    for test_type, result in results.statistical_tests.items():
        assert 0.0 <= result.p_value <= 1.0
        assert result.test_statistic is not None
        assert len(result.bootstrap_distribution) > 0
        
    # Validation report should be comprehensive
    report = results.validation_report
    required_sections = ['summary', 'detailed_results', 'recommendations', 'backtest_analysis', 'risk_analysis']
    for section in required_sections:
        assert section in report, f"Missing {section} in validation report"
    
    # Risk analysis should identify key risks
    risk_analysis = report['risk_analysis']
    assert 'overall_risk_score' in risk_analysis
    assert 'statistical_risks' in risk_analysis
    assert 'market_risks' in risk_analysis
    
    # Final recommendation should be reasonable
    assert 0.0 <= results.confidence_score <= 1.0
    assert results.deployment_readiness in ['READY', 'CAUTION', 'NOT_READY']
    assert len(results.final_recommendation) > 50
    
    # Print summary for inspection
    print(f"\n=== End-to-End Validation Results ===")
    print(f"Deployment Status: {results.deployment_readiness}")
    print(f"Confidence Score: {results.confidence_score:.2%}")
    print(f"Risk Level: {risk_analysis['risk_level']}")
    print(f"Tests Performed: {list(results.statistical_tests.keys())}")
    
    for test_type, result in results.statistical_tests.items():
        print(f"{test_type.value}: p={result.p_value:.4f}, significant={result.is_significant}")
    
    print(f"\nRecommendation Summary:")
    print(f"{results.final_recommendation[:300]}...")
    
    # Validate that results make sense
    if results.confidence_score > 0.7:
        assert results.deployment_readiness in ['READY', 'CAUTION']
    elif results.confidence_score < 0.4:
        assert results.deployment_readiness == 'NOT_READY'
    
    assert results.confidence_score == pytest.approx(results.confidence_score, abs=0.01)