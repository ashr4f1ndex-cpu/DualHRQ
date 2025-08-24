"""
Advanced Statistical Validation Suite

Research-grade statistical testing framework:
- Reality Check test for null hypothesis validation
- Superior Predictive Ability (SPA) test with multiple testing correction
- Data Snooping-Robust (DSR) test for extensive data mining control
- Bootstrap procedures for statistical inference
- Comprehensive backtest validation integration
"""

from .statistical_validity import (
    StatisticalValidityFramework,
    RealityCheckTest,
    SPATest,
    DSRTest,
    StatisticalTest,
    TestResult,
    TestType,
    MultipleTestingResult
)

from .backtest_validation import (
    BacktestValidator,
    ValidationConfig,
    ValidationResults
)

__all__ = [
    'StatisticalValidityFramework',
    'RealityCheckTest',
    'SPATest',
    'DSRTest',
    'StatisticalTest',
    'TestResult',
    'TestType',
    'MultipleTestingResult',
    'BacktestValidator',
    'ValidationConfig',
    'ValidationResults'
]