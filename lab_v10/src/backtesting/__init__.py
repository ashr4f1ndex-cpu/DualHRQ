"""
Advanced Backtesting Suite

World-class simulation engines surpassing top institutions:
- Production-grade SSR/LULD compliant execution
- Almgren-Chriss optimal order execution
- Advanced options simulation with complete Greeks
- Real-time risk management and compliance
- HRM-integrated signal processing
"""

from .advanced_backtester import (
    Trade,
    Position, 
    ExecutionModel,
    AlmgrenChrissExecution,
    SSRLULDCompliance,
    RiskManager,
    AdvancedBacktester
)

from .options_backtester import (
    OptionContract,
    OptionPosition,
    BlackScholesEngine,
    CRRBinomialEngine,
    VolatilitySurface,
    OptionsBacktester
)

__all__ = [
    'Trade',
    'Position',
    'ExecutionModel', 
    'AlmgrenChrissExecution',
    'SSRLULDCompliance',
    'RiskManager',
    'AdvancedBacktester',
    'OptionContract',
    'OptionPosition',
    'BlackScholesEngine',
    'CRRBinomialEngine',
    'VolatilitySurface',
    'OptionsBacktester'
]