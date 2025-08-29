"""
Conditioning Package
===================

Core conditioning system components for DualHRQ 2.0.
"""

# Import stubs for immediate test compatibility
try:
    from .film_conditioning import *
    from .regime_features import *
    from .rag_system import *
    from .feature_flags import *
except ImportError:
    pass  # Graceful degradation for tests