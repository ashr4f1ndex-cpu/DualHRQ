"""
Advanced Feature Engineering Suite

World-class financial feature engineering rivaling top institutions:
- Institution-grade options features with SVI parametrization
- HFT-quality intraday features with microsecond precision
- Advanced leakage prevention with CPCV
- Production-ready integration with HRM system
"""

from .advanced_options import (
    SVIParametrization,
    AdvancedGreeks,
    VolatilityRegimeDetection,
    TermStructurePCA,
    SmileArbitrageDetector,
    OptionsFeatureEngine
)

from .hft_intraday import (
    VWAPEngine,
    ATREngine,
    OrderBookFeatures,
    MicrostructureNoise,
    RegulatoryCompliance,
    IntradayFeatureEngine
)

from .leakage_prevention import (
    PurgedKFold,
    LeakageAuditor
)

__all__ = [
    'SVIParametrization',
    'AdvancedGreeks', 
    'VolatilityRegimeDetection',
    'TermStructurePCA',
    'SmileArbitrageDetector',
    'OptionsFeatureEngine',
    'VWAPEngine',
    'ATREngine',
    'OrderBookFeatures',
    'MicrostructureNoise',
    'RegulatoryCompliance',
    'IntradayFeatureEngine',
    'PurgedKFold',
    'LeakageAuditor'
]