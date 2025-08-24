"""
Backtest Validation Integration for DualHRQ.

Integrates statistical validity testing with backtesting results
to provide comprehensive validation of trading strategies.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import torch

from .statistical_validity import (
    StatisticalValidityFramework, StatisticalTest, TestType, TestResult
)
from ..trading.realistic_backtester import BacktestResults

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for backtest validation."""
    reality_check_bootstrap: int = 5000
    spa_bootstrap: int = 3000
    dsr_bootstrap: int = 2000
    significance_level: float = 0.05
    min_observations: int = 252  # Minimum 1 year of daily data
    enable_all_tests: bool = True
    benchmark_symbol: str = 'SPY'


@dataclass
class ValidationResults:
    """Combined validation results."""
    backtest_results: BacktestResults
    statistical_tests: Dict[TestType, TestResult]
    validation_report: Dict[str, Any]
    final_recommendation: str
    confidence_score: float
    deployment_readiness: str  # 'READY', 'CAUTION', 'NOT_READY'


class BacktestValidator:
    """
    Comprehensive backtest validator combining statistical tests.
    
    Validates trading strategy results against data mining bias
    and provides actionable deployment recommendations.
    """
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.statistical_framework = StatisticalValidityFramework()
        
    def validate_backtest_results(self, backtest_results: BacktestResults,
                                benchmark_returns: np.ndarray,
                                alternative_strategies: Optional[List[np.ndarray]] = None,
                                strategy_development_info: Optional[Dict] = None) -> ValidationResults:
        """
        Comprehensively validate backtest results.
        
        Args:
            backtest_results: Results from realistic backtesting
            benchmark_returns: Benchmark return series for comparison
            alternative_strategies: Alternative strategy returns tested
            strategy_development_info: Info about development process
            
        Returns:
            ValidationResults with comprehensive analysis
        """
        # Extract strategy returns from backtest
        strategy_returns = backtest_results.daily_returns.values
        
        # Validate data sufficiency
        if len(strategy_returns) < self.config.min_observations:
            warnings.warn(f"Insufficient data for validation: {len(strategy_returns)} < {self.config.min_observations}")
        
        # Prepare statistical test configuration
        test_config = StatisticalTest(
            test_type=TestType.REALITY_CHECK,
            n_bootstrap=self.config.reality_check_bootstrap,
            significance_level=self.config.significance_level,
            min_observations=self.config.min_observations
        )
        
        # Determine number of strategies tested
        n_strategies_tested = self._estimate_strategies_tested(
            strategy_development_info, alternative_strategies
        )
        
        # Run comprehensive statistical validation
        statistical_results = self.statistical_framework.comprehensive_validation(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            alternative_strategies=alternative_strategies,
            n_strategies_tested=n_strategies_tested,
            config=test_config
        )
        
        # Generate validation report
        validation_report = self.statistical_framework.generate_validation_report()
        
        # Add backtest-specific analysis
        validation_report['backtest_analysis'] = self._analyze_backtest_results(backtest_results)
        validation_report['risk_analysis'] = self._analyze_risks(backtest_results, statistical_results)
        
        # Generate final recommendation
        final_recommendation, confidence_score, deployment_status = self._generate_final_recommendation(
            backtest_results, statistical_results, validation_report
        )
        
        return ValidationResults(
            backtest_results=backtest_results,
            statistical_tests=statistical_results,
            validation_report=validation_report,
            final_recommendation=final_recommendation,
            confidence_score=confidence_score,
            deployment_readiness=deployment_status
        )
    
    def _estimate_strategies_tested(self, development_info: Optional[Dict],
                                  alternatives: Optional[List]) -> int:
        """Estimate total number of strategies tested during development."""
        if development_info and 'n_strategies_tested' in development_info:
            return development_info['n_strategies_tested']
        
        if alternatives:
            return len(alternatives) + 1  # Alternatives plus main strategy
        
        # Conservative default estimate
        return 50  # Assume moderate data mining
    
    def _analyze_backtest_results(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze backtest results for additional insights."""
        analysis = {
            'performance_metrics': {
                'total_return': results.total_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': self._calculate_win_rate(results.daily_returns),
                'profit_factor': self._calculate_profit_factor(results.daily_returns),
                'calmar_ratio': results.total_return / results.max_drawdown if results.max_drawdown > 0 else np.inf
            },
            'regulatory_compliance': {
                'compliance_rate': results.compliance_rate,
                'regulatory_violations': results.regulatory_violations,
                'ssr_impacts': results.ssr_impacts,
                'luld_impacts': results.luld_impacts
            },
            'trading_activity': {
                'total_trades': len(results.trades),
                'successful_trades': len([t for t in results.trades if t.compliance_valid]),
                'avg_trade_size': np.mean([t.value for t in results.trades]) if results.trades else 0,
                'turnover_rate': self._calculate_turnover(results)
            }
        }
        
        # Risk-adjusted metrics
        if len(results.daily_returns) > 0:
            analysis['risk_metrics'] = {
                'volatility': np.std(results.daily_returns) * np.sqrt(252),
                'downside_deviation': self._calculate_downside_deviation(results.daily_returns),
                'var_95': np.percentile(results.daily_returns, 5),
                'cvar_95': np.mean(results.daily_returns[results.daily_returns <= np.percentile(results.daily_returns, 5)]),
                'tail_ratio': self._calculate_tail_ratio(results.daily_returns)
            }
        
        return analysis
    
    def _analyze_risks(self, backtest_results: BacktestResults, 
                      statistical_results: Dict[TestType, TestResult]) -> Dict[str, Any]:
        """Analyze risks associated with strategy deployment."""
        risks = {
            'statistical_risks': self._assess_statistical_risks(statistical_results),
            'market_risks': self._assess_market_risks(backtest_results),
            'operational_risks': self._assess_operational_risks(backtest_results),
            'regulatory_risks': self._assess_regulatory_risks(backtest_results)
        }
        
        # Overall risk score
        risk_scores = []
        for risk_category, risk_data in risks.items():
            if isinstance(risk_data, dict) and 'risk_score' in risk_data:
                risk_scores.append(risk_data['risk_score'])
        
        risks['overall_risk_score'] = np.mean(risk_scores) if risk_scores else 0.5
        risks['risk_level'] = self._classify_risk_level(risks['overall_risk_score'])
        
        return risks
    
    def _generate_final_recommendation(self, backtest_results: BacktestResults,
                                     statistical_results: Dict[TestType, TestResult],
                                     validation_report: Dict[str, Any]) -> Tuple[str, float, str]:
        """Generate final deployment recommendation."""
        # Statistical evidence strength
        statistical_summary = validation_report['summary']
        statistical_confidence = self._map_conclusion_to_confidence(
            statistical_summary['overall_conclusion']
        )
        
        # Backtest quality score
        backtest_quality = self._assess_backtest_quality(backtest_results)
        
        # Risk assessment
        risk_score = validation_report.get('risk_analysis', {}).get('overall_risk_score', 0.5)
        
        # Regulatory compliance score
        compliance_score = backtest_results.compliance_rate
        
        # Combined confidence score
        confidence_score = (
            statistical_confidence * 0.4 +
            backtest_quality * 0.3 +
            (1 - risk_score) * 0.2 +
            compliance_score * 0.1
        )
        
        # Generate recommendation
        if confidence_score >= 0.8 and statistical_confidence >= 0.7:
            deployment_status = "READY"
            recommendation = (
                f"STRONG RECOMMENDATION for deployment (confidence: {confidence_score:.1%}). "
                f"Strategy shows robust statistical evidence with {statistical_summary['significant_tests']}/{statistical_summary['tests_performed']} significant tests. "
                f"Risk-adjusted returns of {backtest_results.sharpe_ratio:.2f} Sharpe ratio with {backtest_results.compliance_rate:.1%} regulatory compliance. "
                f"Recommend gradual scaling with continued monitoring."
            )
        elif confidence_score >= 0.6 and statistical_confidence >= 0.5:
            deployment_status = "CAUTION"
            recommendation = (
                f"CONDITIONAL RECOMMENDATION for deployment (confidence: {confidence_score:.1%}). "
                f"Mixed statistical evidence requires careful consideration. "
                f"Consider paper trading or reduced position sizing initially. "
                f"Monitor performance closely for {len(backtest_results.daily_returns)//4} quarters before full deployment."
            )
        elif confidence_score >= 0.4:
            deployment_status = "NOT_READY"
            recommendation = (
                f"WEAK RECOMMENDATION against deployment (confidence: {confidence_score:.1%}). "
                f"Limited statistical evidence suggests potential overfitting. "
                f"Recommend additional out-of-sample validation or strategy refinement. "
                f"Consider paper trading only with strict performance thresholds."
            )
        else:
            deployment_status = "NOT_READY"
            recommendation = (
                f"STRONG RECOMMENDATION against deployment (confidence: {confidence_score:.1%}). "
                f"Results likely due to data mining or statistical artifacts. "
                f"Strategy requires fundamental redesign and additional validation. "
                f"Do not deploy in current form."
            )
        
        return recommendation, confidence_score, deployment_status
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (proportion of positive returns)."""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns > 0)
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(returns) == 0:
            return 1.0
        
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        return profits / losses if losses > 0 else np.inf
    
    def _calculate_turnover(self, results: BacktestResults) -> float:
        """Calculate portfolio turnover rate."""
        if not results.trades:
            return 0.0
        
        total_traded_value = sum(t.value for t in results.trades)
        avg_portfolio_value = results.portfolio_value  # Simplified
        
        return total_traded_value / avg_portfolio_value if avg_portfolio_value > 0 else 0.0
    
    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation (volatility of negative returns)."""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        
        return np.std(negative_returns) * np.sqrt(252)
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        if len(returns) < 20:
            return 1.0
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        return abs(p95 / p5) if p5 != 0 else np.inf
    
    def _assess_statistical_risks(self, statistical_results: Dict[TestType, TestResult]) -> Dict[str, Any]:
        """Assess risks from statistical test results."""
        significant_tests = sum(1 for result in statistical_results.values() if result.is_significant)
        total_tests = len(statistical_results)
        
        # Higher risk if fewer tests are significant
        risk_score = 1 - (significant_tests / total_tests) if total_tests > 0 else 0.8
        
        concerns = []
        if TestType.DSR in statistical_results and not statistical_results[TestType.DSR].is_significant:
            concerns.append("Data snooping bias detected")
        if significant_tests == 0:
            concerns.append("No statistical evidence of skill")
        if total_tests < 2:
            concerns.append("Limited statistical validation")
        
        return {
            'risk_score': risk_score,
            'significant_tests': significant_tests,
            'total_tests': total_tests,
            'concerns': concerns,
            'assessment': 'High Risk' if risk_score > 0.7 else 'Medium Risk' if risk_score > 0.4 else 'Low Risk'
        }
    
    def _assess_market_risks(self, results: BacktestResults) -> Dict[str, Any]:
        """Assess market-related risks."""
        risk_factors = []
        risk_score = 0.0
        
        # Drawdown risk
        if results.max_drawdown > 0.2:  # > 20%
            risk_factors.append(f"High maximum drawdown: {results.max_drawdown:.1%}")
            risk_score += 0.3
        elif results.max_drawdown > 0.1:
            risk_factors.append(f"Moderate drawdown: {results.max_drawdown:.1%}")
            risk_score += 0.1
        
        # Sharpe ratio risk
        if results.sharpe_ratio < 1.0:
            risk_factors.append(f"Low risk-adjusted returns: Sharpe {results.sharpe_ratio:.2f}")
            risk_score += 0.2
        
        # Volatility concentration
        if len(results.daily_returns) > 50:
            daily_vol = np.std(results.daily_returns) * np.sqrt(252)
            if daily_vol > 0.3:  # > 30% annual volatility
                risk_factors.append(f"High volatility: {daily_vol:.1%}")
                risk_score += 0.2
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'max_drawdown': results.max_drawdown,
            'sharpe_ratio': results.sharpe_ratio,
            'assessment': 'High Risk' if risk_score > 0.6 else 'Medium Risk' if risk_score > 0.3 else 'Low Risk'
        }
    
    def _assess_operational_risks(self, results: BacktestResults) -> Dict[str, Any]:
        """Assess operational implementation risks."""
        risk_factors = []
        risk_score = 0.0
        
        # Trading frequency risk
        if results.trades:
            trading_days = len(results.daily_returns)
            trades_per_day = len(results.trades) / trading_days if trading_days > 0 else 0
            
            if trades_per_day > 10:  # Very high frequency
                risk_factors.append("Very high trading frequency may face execution challenges")
                risk_score += 0.3
            elif trades_per_day > 2:
                risk_factors.append("High trading frequency requires robust execution")
                risk_score += 0.1
        
        # Position concentration
        if results.positions:
            position_count = len(results.positions)
            if position_count < 5:
                risk_factors.append("Low diversification - concentrated positions")
                risk_score += 0.2
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'assessment': 'High Risk' if risk_score > 0.5 else 'Medium Risk' if risk_score > 0.2 else 'Low Risk'
        }
    
    def _assess_regulatory_risks(self, results: BacktestResults) -> Dict[str, Any]:
        """Assess regulatory compliance risks."""
        risk_factors = []
        risk_score = 0.0
        
        # Compliance rate risk
        if results.compliance_rate < 0.95:  # < 95%
            risk_factors.append(f"Low compliance rate: {results.compliance_rate:.1%}")
            risk_score += 0.4
        elif results.compliance_rate < 0.98:
            risk_factors.append(f"Moderate compliance issues: {results.compliance_rate:.1%}")
            risk_score += 0.2
        
        # Regulatory violations
        if results.regulatory_violations > 10:
            risk_factors.append(f"High violation count: {results.regulatory_violations}")
            risk_score += 0.3
        elif results.regulatory_violations > 0:
            risk_factors.append(f"Some regulatory violations: {results.regulatory_violations}")
            risk_score += 0.1
        
        # SSR/LULD impacts
        ssr_triggers = results.ssr_impacts.get('total_ssr_triggers', 0)
        luld_violations = results.luld_impacts.get('total_luld_violations', 0)
        
        if ssr_triggers > 5 or luld_violations > 3:
            risk_factors.append("Significant SSR/LULD impacts on trading")
            risk_score += 0.2
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'compliance_rate': results.compliance_rate,
            'violations': results.regulatory_violations,
            'assessment': 'High Risk' if risk_score > 0.5 else 'Medium Risk' if risk_score > 0.2 else 'Low Risk'
        }
    
    def _map_conclusion_to_confidence(self, conclusion: str) -> float:
        """Map statistical conclusion to confidence score."""
        mapping = {
            'STRONG_EVIDENCE': 0.9,
            'MODERATE_EVIDENCE': 0.7,
            'WEAK_EVIDENCE': 0.4,
            'NO_EVIDENCE': 0.1
        }
        return mapping.get(conclusion, 0.3)
    
    def _assess_backtest_quality(self, results: BacktestResults) -> float:
        """Assess overall quality of backtest implementation."""
        quality_score = 0.0
        
        # Data sufficiency
        if len(results.daily_returns) >= 252 * 2:  # >= 2 years
            quality_score += 0.3
        elif len(results.daily_returns) >= 252:  # >= 1 year
            quality_score += 0.2
        else:
            quality_score += 0.1
        
        # Trading activity
        if results.trades:
            quality_score += 0.2
        
        # Performance consistency
        if results.sharpe_ratio > 1.0:
            quality_score += 0.2
        elif results.sharpe_ratio > 0.5:
            quality_score += 0.1
        
        # Risk management
        if results.max_drawdown < 0.15:  # < 15%
            quality_score += 0.2
        elif results.max_drawdown < 0.25:
            quality_score += 0.1
        
        # Regulatory compliance
        if results.compliance_rate >= 0.98:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify overall risk level."""
        if risk_score > 0.7:
            return "High Risk"
        elif risk_score > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"