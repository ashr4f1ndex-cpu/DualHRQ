"""
conditioning_system.py
======================

Unified Conditioning System API for DualHRQ 2.0
Combines Pattern Library, RAG System, and HRM Integration with feature flags.

DRQ-103: Single ConditioningSystem API with feature flags and parameter budget compliance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import warnings
from dataclasses import dataclass
import threading

# Import all conditioning components
from .pattern_library import PatternLibrary, PatternDetector, Pattern
from .rag_system import RAGSystem, RetrievalContext, PatternRAG
from .hrm_integration import HRMAdapter, ConditioningInterface, ParameterBudgetManager, HRMConfig


@dataclass
class ConditioningConfig:
    """Configuration for unified conditioning system."""
    
    # Feature flags
    enable_patterns: bool = True
    enable_rag: bool = True
    enable_regime_classification: bool = True
    
    # Component-specific configs
    max_patterns: int = 10000
    pattern_library_ttl_days: int = 30
    rag_max_contexts: int = 10000
    rag_retrieval_k: int = 10
    rag_embedding_dim: int = 256
    enable_neural_rag: bool = True
    neural_rag_budget: int = 100_000
    
    # HRM integration
    hrm_config: Optional[HRMConfig] = None
    conditioning_dim: int = 192
    max_conditioning_params: int = 300_000
    
    # Performance settings
    pattern_search_timeout_ms: float = 20.0
    rag_retrieval_timeout_ms: float = 60.0
    total_conditioning_timeout_ms: float = 100.0
    
    # Parameter budget
    total_parameter_budget: int = 300_000  # Total conditioning system budget
    
    # Fail-open settings
    fail_open_on_timeout: bool = True
    fail_open_on_error: bool = True


class FeatureFlags:
    """Feature flag management for conditioning components."""
    
    def __init__(self, config: ConditioningConfig):
        self.config = config
        self._flags = {
            'patterns': config.enable_patterns,
            'rag': config.enable_rag,
            'regime_classification': config.enable_regime_classification,
            'neural_rag': config.enable_neural_rag,
        }
        self._lock = threading.RLock()
    
    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled."""
        with self._lock:
            return self._flags.get(flag_name, False)
    
    def set_flag(self, flag_name: str, enabled: bool) -> None:
        """Set a feature flag state."""
        with self._lock:
            if flag_name in self._flags:
                self._flags[flag_name] = enabled
                print(f"Feature flag '{flag_name}' set to {enabled}")
            else:
                raise ValueError(f"Unknown feature flag: {flag_name}")
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get current state of all feature flags."""
        with self._lock:
            return self._flags.copy()
    
    def disable_all(self) -> None:
        """Emergency disable all feature flags."""
        with self._lock:
            for flag_name in self._flags:
                self._flags[flag_name] = False
            print("All feature flags disabled (emergency mode)")
    
    def enable_all(self) -> None:
        """Enable all feature flags."""
        with self._lock:
            for flag_name in self._flags:
                self._flags[flag_name] = True
            print("All feature flags enabled")


class ConditioningSystem(nn.Module):
    """Unified conditioning system combining all components with feature flags."""
    
    def __init__(self, config: ConditioningConfig):
        super().__init__()
        self.config = config
        
        # Feature flag management
        self.feature_flags = FeatureFlags(config)
        
        # Parameter budget manager
        self.parameter_manager = ParameterBudgetManager(
            total_budget=config.total_parameter_budget,
            strict_enforcement=True
        )
        
        # Initialize core components based on feature flags
        self._initialize_components()
        
        # Unified conditioning interface
        self.conditioning_interface = ConditioningInterface(
            pattern_dim=128,  # Pattern library embedding dim
            rag_dim=config.rag_embedding_dim,
            regime_dim=64,    # Regime features dim
            output_dim=config.conditioning_dim,
            strength_control=True,
            temporal_smoothing=True
        )
        
        # HRM adapter (if HRM config provided)
        if config.hrm_config:
            self.hrm_adapter = HRMAdapter(
                hrm_config=config.hrm_config,
                conditioning_dim=config.conditioning_dim,
                max_additional_params=config.max_conditioning_params
            )
        else:
            self.hrm_adapter = None
        
        # Register components for parameter tracking
        self.parameter_manager.register_component('conditioning_interface', self.conditioning_interface)
        if self.hrm_adapter:
            self.parameter_manager.register_component('hrm_adapter', self.hrm_adapter)
        
        # Performance tracking
        self.performance_stats = {
            'pattern_search_times': [],
            'rag_retrieval_times': [],
            'total_conditioning_times': [],
            'feature_flag_usage': {'patterns': 0, 'rag': 0, 'regime': 0}
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Verify parameter budget compliance
        self._verify_parameter_budget()
    
    def _initialize_components(self):
        """Initialize conditioning components based on feature flags."""
        # Pattern Library
        if self.feature_flags.is_enabled('patterns'):
            self.pattern_library = PatternLibrary(
                max_patterns=self.config.max_patterns,
                pattern_ttl_days=self.config.pattern_library_ttl_days
            )
            self.pattern_detector = PatternDetector(
                scales=['5m', '15m', '30m', '60m'],
                streaming_mode=True
            )
            print("Pattern Library initialized")
        else:
            self.pattern_library = None
            self.pattern_detector = None
        
        # RAG System
        if self.feature_flags.is_enabled('rag'):
            self.rag_system = RAGSystem(
                max_contexts=self.config.rag_max_contexts,
                retrieval_k=self.config.rag_retrieval_k,
                embedding_dim=self.config.rag_embedding_dim,
                enable_neural_rag=self.config.enable_neural_rag,
                neural_rag_budget=self.config.neural_rag_budget
            )
            print("RAG System initialized")
        else:
            self.rag_system = None
        
        # Regime Classification (placeholder - would integrate with actual regime classifier)
        if self.feature_flags.is_enabled('regime_classification'):
            self.regime_classifier = self._create_regime_classifier()
            print("Regime Classifier initialized")
        else:
            self.regime_classifier = None
    
    def _create_regime_classifier(self) -> nn.Module:
        """Create lightweight regime classifier within parameter budget."""
        # Ultra-lightweight regime classifier to stay within budget
        return nn.Sequential(
            nn.Linear(10, 32, bias=False),  # Market features -> hidden
            nn.ReLU(),
            nn.Linear(32, 64, bias=True),   # Hidden -> regime features
            nn.Tanh()
        )
    
    def _verify_parameter_budget(self):
        """Verify total parameter count is within budget."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if total_params > self.config.total_parameter_budget:
            raise ValueError(f"Conditioning system exceeds parameter budget: "
                           f"{total_params:,} > {self.config.total_parameter_budget:,}")
        
        print(f"Conditioning system initialized with {total_params:,} parameters "
              f"(budget: {self.config.total_parameter_budget:,})")
    
    def condition_market_context(self, market_data: Dict[str, Any], 
                                current_patterns: Optional[List[Pattern]] = None,
                                use_fail_open: bool = None) -> Dict[str, Any]:
        """Main conditioning method combining all enabled components."""
        start_time = datetime.now()
        use_fail_open = use_fail_open if use_fail_open is not None else self.config.fail_open_on_timeout
        
        with self._lock:
            try:
                conditioning_inputs = {}
                
                # Step 1: Pattern-based conditioning
                if self.feature_flags.is_enabled('patterns') and self.pattern_library:
                    pattern_conditioning = self._get_pattern_conditioning(
                        market_data, current_patterns
                    )
                    if pattern_conditioning is not None:
                        conditioning_inputs['patterns'] = pattern_conditioning
                        self.performance_stats['feature_flag_usage']['patterns'] += 1
                
                # Step 2: RAG-based conditioning  
                if self.feature_flags.is_enabled('rag') and self.rag_system:
                    rag_conditioning = self._get_rag_conditioning(market_data)
                    if rag_conditioning is not None:
                        conditioning_inputs['rag_context'] = rag_conditioning
                        self.performance_stats['feature_flag_usage']['rag'] += 1
                
                # Step 3: Regime-based conditioning
                if self.feature_flags.is_enabled('regime_classification') and self.regime_classifier:
                    regime_conditioning = self._get_regime_conditioning(market_data)
                    if regime_conditioning is not None:
                        conditioning_inputs['regime_state'] = regime_conditioning
                        self.performance_stats['feature_flag_usage']['regime'] += 1
                
                # Step 4: Combine all conditioning sources
                if conditioning_inputs:
                    combined_conditioning = self.conditioning_interface.combine_conditioning_sources(
                        **conditioning_inputs,
                        conditioning_strength=market_data.get('conditioning_strength', 1.0)
                    )
                else:
                    # No conditioning available - return neutral conditioning
                    combined_conditioning = torch.zeros(1, self.config.conditioning_dim)
                
                # Track performance
                elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
                self.performance_stats['total_conditioning_times'].append(elapsed_ms)
                
                if elapsed_ms > self.config.total_conditioning_timeout_ms:
                    if use_fail_open:
                        warnings.warn(f"Conditioning timeout ({elapsed_ms:.2f}ms), using fail-open")
                        return self._get_fail_open_response(market_data)
                    else:
                        warnings.warn(f"Conditioning exceeded timeout: {elapsed_ms:.2f}ms")
                
                return {
                    'conditioning_vector': combined_conditioning,
                    'components_used': list(conditioning_inputs.keys()),
                    'processing_time_ms': elapsed_ms,
                    'feature_flags': self.feature_flags.get_all_flags()
                }
                
            except Exception as e:
                if use_fail_open:
                    warnings.warn(f"Conditioning error: {e}, using fail-open")
                    return self._get_fail_open_response(market_data)
                else:
                    raise
    
    def _get_pattern_conditioning(self, market_data: Dict[str, Any], 
                                 current_patterns: Optional[List[Pattern]]) -> Optional[torch.Tensor]:
        """Get pattern-based conditioning."""
        try:
            import time
            start_time = time.time()
            
            if current_patterns:
                # Use provided patterns
                patterns = current_patterns[:5]  # Limit for performance
            else:
                # Detect patterns from market data if data is provided
                if 'price_data' in market_data:
                    patterns = self.pattern_detector.detect_patterns(market_data['price_data'])
                    patterns = patterns[:5]  # Limit for performance
                else:
                    return None
            
            if not patterns:
                return None
            
            # Convert patterns to conditioning features
            pattern_features = []
            for pattern in patterns:
                if hasattr(pattern, 'to_feature_vector'):
                    feature_vec = pattern.to_feature_vector()
                    # Pad to 128 dims
                    if len(feature_vec) < 128:
                        padded = np.zeros(128)
                        padded[:len(feature_vec)] = feature_vec
                        pattern_features.append(padded)
                    else:
                        pattern_features.append(feature_vec[:128])
            
            if pattern_features:
                # Average pattern features
                avg_features = np.mean(pattern_features, axis=0)
                
                # Track timing
                elapsed_ms = (time.time() - start_time) * 1000
                self.performance_stats['pattern_search_times'].append(elapsed_ms)
                
                return torch.tensor(avg_features, dtype=torch.float32).unsqueeze(0)
            
            return None
            
        except Exception as e:
            warnings.warn(f"Pattern conditioning failed: {e}")
            return None
    
    def _get_rag_conditioning(self, market_data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Get RAG-based conditioning."""
        try:
            import time
            start_time = time.time()
            
            # Retrieve relevant historical contexts
            relevant_contexts = self.rag_system.retrieve_relevant_contexts(market_data)
            
            if not relevant_contexts:
                return None
            
            # Convert contexts to conditioning tensor
            context_embeddings = []
            for ctx in relevant_contexts[:5]:  # Limit for performance
                if hasattr(ctx, '_embedding') and ctx._embedding is not None:
                    context_embeddings.append(ctx._embedding)
                else:
                    # Generate embedding
                    embedding = self.rag_system.encoder.encode_market_context(ctx.market_state)
                    context_embeddings.append(embedding)
            
            if context_embeddings:
                # Average context embeddings
                avg_embedding = np.mean(context_embeddings, axis=0)
                
                # Track timing
                elapsed_ms = (time.time() - start_time) * 1000
                self.performance_stats['rag_retrieval_times'].append(elapsed_ms)
                
                return torch.tensor(avg_embedding, dtype=torch.float32).unsqueeze(0)
            
            return None
            
        except Exception as e:
            warnings.warn(f"RAG conditioning failed: {e}")
            return None
    
    def _get_regime_conditioning(self, market_data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Get regime-based conditioning."""
        try:
            # Extract basic market features for regime classification
            market_features = []
            
            # Price-based features
            market_features.append(market_data.get('price_change_1m', 0.0))
            market_features.append(market_data.get('price_change_5m', 0.0))
            market_features.append(market_data.get('volatility', 0.2))
            market_features.append(market_data.get('volume_ratio', 1.0))
            market_features.append(market_data.get('trend_strength', 0.0))
            
            # Technical indicators
            market_features.append(market_data.get('rsi', 50.0) / 100.0)  # Normalize
            market_features.append(market_data.get('vix_level', 20.0) / 50.0)  # Normalize
            
            # Time-based features
            current_time = market_data.get('timestamp', datetime.now())
            market_features.append(current_time.hour / 24.0)
            market_features.append(current_time.weekday() / 7.0)
            
            # Session indicator
            session = market_data.get('session', 'regular')
            market_features.append(1.0 if session in ['open', 'close'] else 0.0)
            
            # Pad to 10 features
            while len(market_features) < 10:
                market_features.append(0.0)
            
            # Run through regime classifier
            feature_tensor = torch.tensor(market_features[:10], dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                regime_features = self.regime_classifier(feature_tensor)
            
            return regime_features
            
        except Exception as e:
            warnings.warn(f"Regime conditioning failed: {e}")
            return None
    
    def _get_fail_open_response(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return fail-open response when conditioning fails."""
        return {
            'conditioning_vector': torch.zeros(1, self.config.conditioning_dim),
            'components_used': [],
            'processing_time_ms': 0.0,
            'feature_flags': self.feature_flags.get_all_flags(),
            'fail_open': True
        }
    
    def apply_conditioning_to_hrm(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor,
                                 conditioning_result: Dict[str, Any]) -> tuple:
        """Apply conditioning to HRM tokens if HRM adapter is available."""
        if self.hrm_adapter is None:
            return h_tokens, l_tokens
        
        conditioning_vector = conditioning_result['conditioning_vector']
        return self.hrm_adapter.apply_conditioning(h_tokens, l_tokens, conditioning_vector)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'parameter_count': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'parameter_budget': self.config.total_parameter_budget,
            'feature_flags': self.feature_flags.get_all_flags(),
            'component_usage': self.performance_stats['feature_flag_usage'].copy(),
        }
        
        # Timing statistics
        for timing_key in ['pattern_search_times', 'rag_retrieval_times', 'total_conditioning_times']:
            times = self.performance_stats[timing_key]
            if times:
                stats[timing_key] = {
                    'mean': np.mean(times),
                    'p95': np.percentile(times, 95),
                    'max': np.max(times),
                    'count': len(times)
                }
            else:
                stats[timing_key] = {'mean': 0, 'p95': 0, 'max': 0, 'count': 0}
        
        return stats
    
    def emergency_disable(self):
        """Emergency disable all conditioning components."""
        self.feature_flags.disable_all()
        print("Emergency disable activated - all conditioning disabled")
    
    def set_feature_flag(self, flag_name: str, enabled: bool):
        """Set individual feature flag."""
        self.feature_flags.set_flag(flag_name, enabled)
    
    def forward(self, market_data: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for training/inference."""
        result = self.condition_market_context(market_data)
        return result['conditioning_vector']