"""
Validation module for DualHRQ system.

This module contains tools for validating that the system doesn't have
information leakage, particularly puzzle_id leakage.
"""

from .leakage_detector import MutualInformationTester, LeakageValidator
from .shuffle_test import ShuffleTestValidator

__all__ = [
    'MutualInformationTester',
    'LeakageValidator', 
    'ShuffleTestValidator'
]