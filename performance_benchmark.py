"""
Performance Benchmark for DualHRQ 2.0 Conditioning System
==========================================================

Comprehensive performance validation against DRQ requirements.
"""

import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.conditioning.conditioning_system import ConditioningSystem, ConditioningConfig
from src.conditioning.pattern_library import PatternLibrary, Pattern, PatternDetector
from src.conditioning.rag_system import RAGSystem, RetrievalContext
from src.conditioning.hrm_integration import HRMConfig


def benchmark_pattern_library():
    """Benchmark Pattern Library against DRQ-101 requirements."""
    print("=== DRQ-101 Pattern Library Benchmark ===")
    
    # Test with 10K patterns
    library = PatternLibrary(max_patterns=10000)
    
    # Generate test patterns
    print("Generating 10,000 test patterns...")
    patterns = []
    for i in range(10000):
        pattern = Pattern(
            pattern_id=f"perf_test_{i}",
            pattern_type=np.random.choice(['trend', 'reversal', 'breakout']),
            scale=np.random.choice(['5m', '15m', '30m', '60m']),
            features={
                'duration': np.random.randint(10, 200),
                'amplitude': np.random.exponential(0.02),
                'volume_profile': np.random.lognormal(0, 0.5)
            },
            strength=np.random.random()
        )
        patterns.append(pattern)
    
    # Batch storage test
    start_time = time.time()
    stored = library.store_patterns_batch(patterns)
    storage_time = time.time() - start_time
    
    print(f"✓ Stored {stored:,} patterns in {storage_time:.2f}s")
    print(f"✓ Storage capacity: {library.size():,} patterns")
    
    # Similarity search performance test
    query_features = {
        'pattern_type': 'trend',
        'scale': '15m',
        'min_strength': 0.5
    }
    
    # Warm-up query
    library.find_similar_patterns(query_features, top_k=10)
    
    # Performance test - multiple queries
    search_times = []
    for _ in range(100):
        start_time = time.time()
        results = library.find_similar_patterns(query_features, top_k=10)
        search_time = (time.time() - start_time) * 1000  # ms
        search_times.append(search_time)
    
    avg_search_time = np.mean(search_times)
    p95_search_time = np.percentile(search_times, 95)
    max_search_time = np.max(search_times)
    
    print(f"✓ Average search time: {avg_search_time:.2f}ms")
    print(f"✓ P95 search time: {p95_search_time:.2f}ms")
    print(f"✓ Max search time: {max_search_time:.2f}ms")
    print(f"✓ <20ms requirement: {'PASS' if p95_search_time < 20 else 'FAIL'}")
    
    return {
        'storage_time': storage_time,
        'avg_search_time': avg_search_time,
        'p95_search_time': p95_search_time,
        'max_search_time': max_search_time
    }


def benchmark_rag_system():
    """Benchmark RAG System against DRQ-102 requirements."""
    print("\n=== DRQ-102 RAG System Benchmark ===")
    
    # Test with neural RAG enabled
    rag = RAGSystem(
        max_contexts=10000,
        embedding_dim=256,
        enable_neural_rag=True,
        neural_rag_budget=100_000
    )
    
    # Verify parameter budget
    if rag.neural_rag:
        neural_params = sum(p.numel() for p in rag.neural_rag.parameters())
        print(f"✓ Neural RAG parameters: {neural_params:,} (≤100K: {'PASS' if neural_params <= 100_000 else 'FAIL'})")
    
    # Add test contexts
    print("Adding 5,000 test contexts...")
    contexts = []
    for i in range(5000):
        context = RetrievalContext(
            context_id=f"ctx_{i}",
            market_state={
                'price': 100 + np.random.randn() * 10,
                'volatility': 0.2 + np.random.randn() * 0.05,
                'volume': 1000 + np.random.randint(0, 500),
                'regime': np.random.choice(['trending', 'ranging', 'volatile'])
            },
            patterns=[],
            timestamp=datetime.now(),
            outcome={'return_1h': np.random.randn() * 0.01}
        )
        contexts.append(context)
        rag.add_historical_context(context)
    
    # Retrieval performance test
    query_context = {
        'price': 105,
        'volatility': 0.18,
        'volume': 1200,
        'regime': 'trending'
    }
    
    # Warm-up retrieval
    rag.retrieve_relevant_contexts(query_context)
    
    # Performance test
    retrieval_times = []
    for _ in range(50):
        start_time = time.time()
        results = rag.retrieve_relevant_contexts(query_context)
        retrieval_time = (time.time() - start_time) * 1000  # ms
        retrieval_times.append(retrieval_time)
    
    avg_retrieval_time = np.mean(retrieval_times)
    p95_retrieval_time = np.percentile(retrieval_times, 95)
    max_retrieval_time = np.max(retrieval_times)
    
    print(f"✓ Average retrieval time: {avg_retrieval_time:.2f}ms")
    print(f"✓ P95 retrieval time: {p95_retrieval_time:.2f}ms")
    print(f"✓ Max retrieval time: {max_retrieval_time:.2f}ms")
    print(f"✓ <60ms requirement: {'PASS' if p95_retrieval_time < 60 else 'FAIL'}")
    
    # Circuit breaker test
    cb_state = rag.circuit_breaker.get_state()
    print(f"✓ Circuit breaker state: {cb_state['state']}")
    
    return {
        'avg_retrieval_time': avg_retrieval_time,
        'p95_retrieval_time': p95_retrieval_time,
        'max_retrieval_time': max_retrieval_time,
        'neural_params': neural_params if rag.neural_rag else 0
    }


def benchmark_conditioning_system():
    """Benchmark Unified Conditioning System against DRQ-103 requirements."""
    print("\n=== DRQ-103 Unified Conditioning System Benchmark ===")
    
    # Full configuration with all components
    config = ConditioningConfig(
        total_parameter_budget=300_000,
        enable_patterns=True,
        enable_rag=True,
        enable_regime_classification=True,
        enable_neural_rag=True
    )
    
    system = ConditioningSystem(config)
    
    # Parameter budget validation
    total_params = sum(p.numel() for p in system.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Parameter budget: {config.total_parameter_budget:,}")
    print(f"✓ Budget compliance: {'PASS' if total_params <= config.total_parameter_budget else 'FAIL'}")
    print(f"✓ Budget utilization: {total_params/config.total_parameter_budget:.1%}")
    
    # Feature flag functionality test
    print("\n--- Feature Flag Test ---")
    initial_flags = system.feature_flags.get_all_flags()
    print(f"✓ Initial flags: {initial_flags}")
    
    # Test independent control
    system.set_feature_flag('patterns', False)
    system.set_feature_flag('rag', False)
    flags_after_disable = system.feature_flags.get_all_flags()
    print(f"✓ After disable: {flags_after_disable}")
    
    # Re-enable for performance test
    system.set_feature_flag('patterns', True)
    system.set_feature_flag('rag', True)
    
    # Performance test
    print("\n--- Performance Test ---")
    market_data = {
        'price_change_1m': 0.002,
        'price_change_5m': -0.005,
        'volatility': 0.18,
        'volume_ratio': 1.3,
        'trend_strength': 0.7,
        'timestamp': datetime.now()
    }
    
    # Warm-up
    system.condition_market_context(market_data)
    
    # Performance measurement
    conditioning_times = []
    for _ in range(100):
        start_time = time.time()
        result = system.condition_market_context(market_data)
        conditioning_time = (time.time() - start_time) * 1000  # ms
        conditioning_times.append(conditioning_time)
    
    avg_conditioning_time = np.mean(conditioning_times)
    p95_conditioning_time = np.percentile(conditioning_times, 95)
    max_conditioning_time = np.max(conditioning_times)
    
    print(f"✓ Average conditioning time: {avg_conditioning_time:.2f}ms")
    print(f"✓ P95 conditioning time: {p95_conditioning_time:.2f}ms")
    print(f"✓ Max conditioning time: {max_conditioning_time:.2f}ms")
    print(f"✓ <100ms requirement: {'PASS' if p95_conditioning_time < 100 else 'FAIL'}")
    
    # Fail-open test
    print("\n--- Fail-Open Test ---")
    config_short_timeout = ConditioningConfig(
        total_conditioning_timeout_ms=1.0,  # Very short timeout
        fail_open_on_timeout=True
    )
    
    system_failopen = ConditioningSystem(config_short_timeout)
    result = system_failopen.condition_market_context(market_data)
    
    print(f"✓ Fail-open result: {result.get('fail_open', False)}")
    print(f"✓ Fail-open vector shape: {result['conditioning_vector'].shape}")
    
    return {
        'total_params': total_params,
        'avg_conditioning_time': avg_conditioning_time,
        'p95_conditioning_time': p95_conditioning_time,
        'max_conditioning_time': max_conditioning_time,
        'parameter_budget_pass': total_params <= config.total_parameter_budget,
        'performance_pass': p95_conditioning_time < 100
    }


def benchmark_integration():
    """End-to-end integration benchmark."""
    print("\n=== End-to-End Integration Benchmark ===")
    
    # Full system with HRM integration
    hrm_config = HRMConfig(h_dim=256, l_dim=384, h_layers=2, l_layers=2)
    config = ConditioningConfig(
        hrm_config=hrm_config,
        conditioning_dim=192,
        total_parameter_budget=300_000,
        enable_patterns=True,
        enable_rag=True,
        enable_regime_classification=True
    )
    
    system = ConditioningSystem(config)
    
    # Integration test
    market_data = {
        'price_change_1m': 0.002,
        'volatility': 0.18,
        'timestamp': datetime.now()
    }
    
    # Mock HRM tokens
    batch_size = 4
    h_tokens = torch.randn(batch_size, 10, 256)
    l_tokens = torch.randn(batch_size, 15, 384)
    
    # Full pipeline test
    integration_times = []
    for _ in range(20):
        start_time = time.time()
        
        # Step 1: Get conditioning
        conditioning_result = system.condition_market_context(market_data)
        
        # Step 2: Apply to HRM
        conditioned_h, conditioned_l = system.apply_conditioning_to_hrm(
            h_tokens, l_tokens, conditioning_result
        )
        
        integration_time = (time.time() - start_time) * 1000  # ms
        integration_times.append(integration_time)
    
    avg_integration_time = np.mean(integration_times)
    p95_integration_time = np.percentile(integration_times, 95)
    
    print(f"✓ Average integration time: {avg_integration_time:.2f}ms")
    print(f"✓ P95 integration time: {p95_integration_time:.2f}ms")
    print(f"✓ HRM conditioning shapes: H{conditioned_h.shape}, L{conditioned_l.shape}")
    print(f"✓ Tokens modified: {not torch.allclose(conditioned_h, h_tokens)}")
    
    return {
        'avg_integration_time': avg_integration_time,
        'p95_integration_time': p95_integration_time
    }


def main():
    """Run comprehensive performance benchmark."""
    print("DualHRQ 2.0 Conditioning System Performance Benchmark")
    print("=" * 60)
    
    results = {}
    
    try:
        results['pattern_library'] = benchmark_pattern_library()
        results['rag_system'] = benchmark_rag_system()
        results['conditioning_system'] = benchmark_conditioning_system()
        results['integration'] = benchmark_integration()
        
        # Summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # DRQ-101 Validation
        pl_pass = results['pattern_library']['p95_search_time'] < 20
        print(f"DRQ-101 Pattern Library: {'✓ PASS' if pl_pass else '✗ FAIL'}")
        print(f"  - Search performance: {results['pattern_library']['p95_search_time']:.2f}ms P95 (req: <20ms)")
        
        # DRQ-102 Validation  
        rag_pass = results['rag_system']['p95_retrieval_time'] < 60
        rag_params_pass = results['rag_system']['neural_params'] <= 100_000
        print(f"DRQ-102 RAG System: {'✓ PASS' if rag_pass and rag_params_pass else '✗ FAIL'}")
        print(f"  - Retrieval performance: {results['rag_system']['p95_retrieval_time']:.2f}ms P95 (req: <60ms)")
        print(f"  - Neural parameters: {results['rag_system']['neural_params']:,} (req: ≤100K)")
        
        # DRQ-103 Validation
        cs_params_pass = results['conditioning_system']['parameter_budget_pass']
        cs_perf_pass = results['conditioning_system']['performance_pass']
        print(f"DRQ-103 Conditioning System: {'✓ PASS' if cs_params_pass and cs_perf_pass else '✗ FAIL'}")
        print(f"  - Parameter budget: {results['conditioning_system']['total_params']:,} (req: ≤300K)")
        print(f"  - Conditioning performance: {results['conditioning_system']['p95_conditioning_time']:.2f}ms P95 (req: <100ms)")
        
        # Integration
        integration_pass = results['integration']['p95_integration_time'] < 150
        print(f"End-to-End Integration: {'✓ PASS' if integration_pass else '✗ FAIL'}")
        print(f"  - Integration time: {results['integration']['p95_integration_time']:.2f}ms P95")
        
        # Overall status
        all_pass = pl_pass and rag_pass and rag_params_pass and cs_params_pass and cs_perf_pass and integration_pass
        print(f"\nOVERALL STATUS: {'✓ ALL TESTS PASS' if all_pass else '✗ SOME TESTS FAILED'}")
        
        return results
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()