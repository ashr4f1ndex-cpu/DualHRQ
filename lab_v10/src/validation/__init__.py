"""
Advanced Statistical Validation Suite

Research-grade statistical testing framework:
- Deflated Sharpe Ratio with multiple testing correction
- White's Reality Check for data snooping
- Hansen's Superior Predictive Ability (SPA) test
- Probabilistic Sharpe Ratio with higher moments
- Stationary block bootstrap for time-series inference
- Multiple testing corrections (FDR control)
"""

from .statistical_tests import (
    StatisticalTestResult,
    DeflatedSharpeRatio,
    ProbabilisticSharpeRatio,
    WhitesRealityCheck,
    HansenSPATest,
    StationaryBlockBootstrap,
    MultipleTestingCorrection,
    StatisticalValidationSuite
)

__all__ = [
    'StatisticalTestResult',
    'DeflatedSharpeRatio',
    'ProbabilisticSharpeRatio', 
    'WhitesRealityCheck',
    'HansenSPATest',
    'StationaryBlockBootstrap',
    'MultipleTestingCorrection',
    'StatisticalValidationSuite'
]