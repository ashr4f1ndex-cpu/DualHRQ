"""
Models package for DualHRQ system

This package provides unified access to all model implementations,
including the Hierarchical Reasoning Model (HRM) with adaptive
computation time and continuous learning capabilities.
"""

# Import compatibility layer for main orchestrator
from ..options.hrm_net import HRMNet as HierarchicalReasoningModel, HRMConfig
from ..common.act import AdaptiveComputationTime

__all__ = [
    'HierarchicalReasoningModel', 
    'HRMConfig',
    'AdaptiveComputationTime'
]