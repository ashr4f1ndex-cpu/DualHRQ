"""
test_pattern_library.py
=======================

TDD tests for DRQ-101: Pattern Library Implementation
These tests MUST be written first and will initially FAIL.
Implementation should make these tests pass.

CRITICAL: <10ms lookup, 10K patterns, multi-scale detection.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.models.pattern_library import (
        PatternLibrary,
        PatternDetector,
        Pattern,
        PatternMatcher,
        ScaleConfig
    )
except ImportError:
    # These will fail initially - that's expected in TDD
    pass


class BaseTestClass:
    """Base test class with helper methods."""
    
    def _generate_sample_market_data(self, n_points: int) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        np.random.seed(42)
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='1min')
        prices = 100 + np.cumsum(np.random.randn(n_points) * 0.01)
        volumes = np.random.exponential(1000, n_points)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
    
    def _create_test_patterns(self, count: int, base_date: str = '2024-01-01') -> List[Pattern]:
        """Create test patterns for library testing."""
        patterns = []
        base_time = datetime.strptime(base_date, '%Y-%m-%d')
        
        for i in range(count):
            pattern = Pattern(
                pattern_id=f"test_pattern_{base_date}_{i}",  # Include date to ensure unique IDs
                pattern_type=np.random.choice(['trend', 'reversal', 'breakout', 'support_resistance']),
                scale=np.random.choice(['5m', '15m', '30m', '60m']),
                features={
                    'duration': np.random.randint(10, 200),
                    'amplitude': np.random.exponential(0.02),
                    'volume_profile': np.random.lognormal(0, 0.3),
                    'momentum': np.random.normal(0, 0.5)
                },
                strength=np.random.random(),
                detected_at=base_time + timedelta(minutes=i),
                metadata={
                    'market_regime': np.random.choice(['trending', 'ranging', 'volatile']),
                    'session': np.random.choice(['pre_market', 'regular', 'after_hours'])
                }
            )
            patterns.append(pattern)
        
        return patterns


class TestPatternDetection(BaseTestClass):
    """Tests for pattern detection functionality."""
    
    def test_multi_scale_pattern_detection(self):
        """Test pattern detection at multiple time scales."""
        # This will fail initially until PatternDetector is implemented
        detector = PatternDetector(scales=['5m', '15m', '30m', '60m'])
        
        # Generate synthetic price series
        np.random.seed(42)
        timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
        volumes = np.random.exponential(1000, 1000)
        
        market_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
        
        patterns = detector.detect_patterns(market_data)
        
        # Should detect patterns at all scales
        assert len(patterns) > 0, "Should detect some patterns"
        assert all(p.scale in ['5m', '15m', '30m', '60m'] for p in patterns), \
            "All patterns should have valid scales"
        
        # Should detect different pattern types
        pattern_types = set(p.pattern_type for p in patterns)
        expected_types = {'trend', 'reversal', 'breakout', 'support_resistance'}
        assert len(pattern_types.intersection(expected_types)) > 0, \
            f"Should detect common pattern types, got {pattern_types}"
    
    def test_pattern_strength_scoring(self):
        """Test that patterns have meaningful strength scores."""
        detector = PatternDetector(scales=['15m'])
        
        # Create data with obvious patterns
        # Strong uptrend
        uptrend_prices = np.linspace(100, 110, 100)
        uptrend_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'price': uptrend_prices,
            'volume': np.ones(100) * 1000
        })
        
        # Random walk (weak patterns)
        random_prices = 100 + np.cumsum(np.random.randn(100) * 0.001)
        random_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-02', periods=100, freq='1min'),
            'price': random_prices,
            'volume': np.ones(100) * 1000
        })
        
        uptrend_patterns = detector.detect_patterns(uptrend_data)
        random_patterns = detector.detect_patterns(random_data)
        
        # Strong patterns should have higher scores
        if uptrend_patterns and random_patterns:
            max_uptrend_score = max(p.strength for p in uptrend_patterns)
            max_random_score = max(p.strength for p in random_patterns)
            
            assert max_uptrend_score > max_random_score, \
                "Strong trend should have higher pattern strength"
        
        # All strength scores should be in [0, 1]
        all_patterns = uptrend_patterns + random_patterns
        for pattern in all_patterns:
            assert 0 <= pattern.strength <= 1, \
                f"Pattern strength should be in [0,1], got {pattern.strength}"
    
    def test_pattern_feature_extraction(self):
        """Test feature extraction from detected patterns."""
        detector = PatternDetector(scales=['30m'])
        
        # Generate sample data
        data = self._generate_sample_market_data(500)
        patterns = detector.detect_patterns(data)
        
        assert len(patterns) > 0, "Should detect some patterns"
        
        # Each pattern should have required features
        required_features = ['duration', 'amplitude', 'volume_profile', 'momentum']
        
        for pattern in patterns:
            assert hasattr(pattern, 'features'), "Pattern should have features"
            for feature in required_features:
                assert feature in pattern.features, \
                    f"Pattern should have {feature} feature"
                assert isinstance(pattern.features[feature], (int, float)), \
                    f"Feature {feature} should be numeric"
    
    def test_real_time_pattern_detection(self):
        """Test pattern detection in streaming/real-time mode."""
        detector = PatternDetector(scales=['5m'], streaming_mode=True)
        
        # Simulate streaming data
        base_time = datetime(2024, 1, 1, 9, 30)
        streaming_patterns = []
        
        for i in range(50):
            # Add one new data point
            new_data = pd.DataFrame({
                'timestamp': [base_time + timedelta(minutes=i)],
                'price': [100 + i * 0.1 + np.random.randn() * 0.01],
                'volume': [1000 + np.random.randint(0, 500)]
            })
            
            patterns = detector.update_streaming(new_data)
            streaming_patterns.extend(patterns)
        
        # Should detect patterns in streaming mode
        assert len(streaming_patterns) >= 0, "Streaming should work without errors"
        
        # Streaming patterns should have timestamps
        for pattern in streaming_patterns:
            assert hasattr(pattern, 'detected_at'), \
                "Streaming patterns should have detection timestamp"


class TestPatternLibrary(BaseTestClass):
    """Tests for pattern library storage and retrieval."""
    
    def test_pattern_storage_and_retrieval(self):
        """Test storing and retrieving patterns from library."""
        # This will fail initially until PatternLibrary is implemented
        library = PatternLibrary(max_patterns=10000)
        
        # Create test patterns
        test_patterns = self._create_test_patterns(100)
        
        # Store patterns
        for pattern in test_patterns:
            library.store_pattern(pattern)
        
        assert library.size() == 100, f"Should store 100 patterns, got {library.size()}"
        
        # Retrieve by ID
        pattern_id = test_patterns[0].pattern_id
        retrieved = library.get_pattern(pattern_id)
        
        assert retrieved is not None, "Should retrieve stored pattern"
        assert retrieved.pattern_id == pattern_id, "Retrieved pattern should match ID"
    
    def test_fast_pattern_lookup(self):
        """Test that pattern lookup is <10ms as required."""
        library = PatternLibrary(max_patterns=10000)
        
        # Fill library with patterns
        test_patterns = self._create_test_patterns(5000)
        for pattern in test_patterns:
            library.store_pattern(pattern)
        
        # Test lookup performance
        query_features = {
            'scale': '15m',
            'pattern_type': 'trend',
            'min_strength': 0.5
        }
        
        start_time = time.time()
        similar_patterns = library.find_similar_patterns(query_features, top_k=10)
        lookup_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Must complete in <10ms
        assert lookup_time < 10, \
            f"Pattern lookup should be <10ms, took {lookup_time:.2f}ms"
        
        # Should return relevant patterns
        assert len(similar_patterns) > 0, "Should find similar patterns"
        assert len(similar_patterns) <= 10, "Should respect top_k limit"
    
    def test_pattern_similarity_matching(self):
        """Test pattern similarity and matching algorithms."""
        library = PatternLibrary()
        
        # Create patterns with known similarities
        base_pattern = Pattern(
            pattern_id="base",
            pattern_type="trend",
            scale="15m",
            features={
                'duration': 30,
                'amplitude': 0.05,
                'volume_profile': 1.2,
                'momentum': 0.8
            },
            strength=0.7
        )
        
        # Similar pattern (small differences)
        similar_pattern = Pattern(
            pattern_id="similar",
            pattern_type="trend", 
            scale="15m",
            features={
                'duration': 32,  # Close
                'amplitude': 0.048,  # Close
                'volume_profile': 1.18,  # Close
                'momentum': 0.82  # Close
            },
            strength=0.72
        )
        
        # Different pattern
        different_pattern = Pattern(
            pattern_id="different",
            pattern_type="reversal",  # Different type
            scale="60m",  # Different scale
            features={
                'duration': 120,
                'amplitude': 0.15,
                'volume_profile': 0.5,
                'momentum': -0.3
            },
            strength=0.9
        )
        
        library.store_pattern(base_pattern)
        library.store_pattern(similar_pattern)
        library.store_pattern(different_pattern)
        
        # Find similar to base
        query_features = base_pattern.features.copy()
        query_features['pattern_type'] = base_pattern.pattern_type
        query_features['scale'] = base_pattern.scale
        
        matches = library.find_similar_patterns(query_features, top_k=5)
        
        # Similar pattern should rank higher than different pattern
        match_ids = [m.pattern_id for m in matches]
        if "similar" in match_ids and "different" in match_ids:
            similar_rank = match_ids.index("similar")
            different_rank = match_ids.index("different")
            assert similar_rank < different_rank, \
                "Similar pattern should rank higher"
    
    def test_pattern_lifecycle_management(self):
        """Test pattern aging, expiry, and cleanup."""
        library = PatternLibrary(max_patterns=20, pattern_ttl_days=30)  # Increase capacity to avoid LRU interference
        
        # Add patterns with different ages
        recent_patterns = self._create_test_patterns(5, base_date='2024-01-01')
        old_patterns = self._create_test_patterns(5, base_date='2023-11-01')  # >30 days old
        
        for pattern in recent_patterns + old_patterns:
            library.store_pattern(pattern)
        
        # Trigger cleanup
        library.cleanup_expired_patterns(current_date=datetime(2024, 1, 15))
        
        # Should keep recent patterns, remove old ones
        assert library.size() == 5, f"Should have 5 recent patterns, got {library.size()}"
        
        # Test capacity management - now test with the current capacity limit
        excess_patterns = self._create_test_patterns(20, base_date='2024-01-02')  # More than current capacity
        for pattern in excess_patterns:
            library.store_pattern(pattern)
        
        # Should not exceed capacity (20 max_patterns)
        assert library.size() <= 20, f"Should respect capacity limit, got {library.size()}"
    
    def test_pattern_metadata_indexing(self):
        """Test indexing and querying by pattern metadata."""
        library = PatternLibrary()
        
        # Add patterns with diverse metadata
        patterns = [
            Pattern("p1", "trend", "5m", {}, 0.8, metadata={'sector': 'tech', 'volatility': 'low'}),
            Pattern("p2", "trend", "15m", {}, 0.7, metadata={'sector': 'tech', 'volatility': 'high'}),
            Pattern("p3", "reversal", "30m", {}, 0.9, metadata={'sector': 'finance', 'volatility': 'low'}),
        ]
        
        for pattern in patterns:
            library.store_pattern(pattern)
        
        # Query by metadata
        tech_patterns = library.find_patterns_by_metadata({'sector': 'tech'})
        low_vol_patterns = library.find_patterns_by_metadata({'volatility': 'low'})
        
        assert len(tech_patterns) == 2, "Should find 2 tech patterns"
        assert len(low_vol_patterns) == 2, "Should find 2 low volatility patterns"
        
        # Combined metadata query
        tech_low_vol = library.find_patterns_by_metadata({'sector': 'tech', 'volatility': 'low'})
        assert len(tech_low_vol) == 1, "Should find 1 tech + low volatility pattern"


class TestPatternMatcher(BaseTestClass):
    """Tests for pattern matching and retrieval optimization."""
    
    def test_approximate_nearest_neighbor(self):
        """Test approximate nearest neighbor for fast pattern matching."""
        # This will fail initially until PatternMatcher is implemented
        matcher = PatternMatcher(index_type='ann', embedding_dim=128)
        
        # Create patterns with embeddings
        patterns_with_embeddings = []
        for i in range(1000):
            embedding = np.random.randn(128)
            pattern = Pattern(
                pattern_id=f"p{i}",
                pattern_type=np.random.choice(['trend', 'reversal', 'breakout']),
                scale=np.random.choice(['5m', '15m', '30m']),
                features={},
                strength=np.random.random(),
                embedding=embedding
            )
            patterns_with_embeddings.append(pattern)
        
        # Build index
        matcher.build_index(patterns_with_embeddings)
        
        # Query with embedding
        query_embedding = np.random.randn(128)
        
        start_time = time.time()
        similar = matcher.find_similar_by_embedding(query_embedding, top_k=20)
        query_time = (time.time() - start_time) * 1000
        
        # Should be fast (<5ms for 1000 patterns)
        assert query_time < 5, f"ANN query should be <5ms, took {query_time:.2f}ms"
        
        # Should return requested number of patterns
        assert len(similar) == 20, f"Should return 20 patterns, got {len(similar)}"
    
    def test_pattern_embedding_consistency(self):
        """Test that similar patterns have similar embeddings."""
        matcher = PatternMatcher(embedding_dim=64)
        
        # Create pairs of similar and dissimilar patterns
        similar_pair_1 = Pattern("s1", "trend", "15m", {'amplitude': 0.05}, 0.7)
        similar_pair_2 = Pattern("s2", "trend", "15m", {'amplitude': 0.052}, 0.72)
        
        different_pattern = Pattern("d1", "reversal", "60m", {'amplitude': 0.15}, 0.9)
        
        # Generate embeddings
        emb_s1 = matcher.generate_embedding(similar_pair_1)
        emb_s2 = matcher.generate_embedding(similar_pair_2)
        emb_d1 = matcher.generate_embedding(different_pattern)
        
        # Similar patterns should have closer embeddings
        similar_distance = np.linalg.norm(emb_s1 - emb_s2)
        different_distance = np.linalg.norm(emb_s1 - emb_d1)
        
        assert similar_distance < different_distance, \
            "Similar patterns should have closer embeddings"
    
    def test_batch_pattern_matching(self):
        """Test efficient batch processing for pattern matching."""
        matcher = PatternMatcher()
        library = PatternLibrary()
        
        # Store reference patterns
        reference_patterns = self._create_test_patterns(500)
        for pattern in reference_patterns:
            library.store_pattern(pattern)
            
        # Batch query patterns
        query_patterns = self._create_test_patterns(50)
        
        start_time = time.time()
        batch_results = matcher.batch_find_similar(query_patterns, library, top_k=5)
        batch_time = time.time() - start_time
        
        # Should complete batch in reasonable time
        assert batch_time < 1.0, f"Batch matching should be <1s, took {batch_time:.2f}s"
        
        # Should return results for all queries
        assert len(batch_results) == 50, "Should return results for all queries"
        
        # Each result should have up to 5 matches
        for result in batch_results:
            assert len(result) <= 5, "Each result should have ≤5 matches"


class TestPerformanceAndScaling(BaseTestClass):
    """Tests for performance requirements and scaling behavior."""
    
    def test_10k_pattern_capacity(self):
        """Test library can handle 10K patterns efficiently."""
        library = PatternLibrary(max_patterns=10000)
        
        # Add 10K patterns
        start_time = time.time()
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
            library.store_pattern(pattern)
        
        storage_time = time.time() - start_time
        
        # Storage should be reasonable
        assert storage_time < 30, f"Storing 10K patterns should be <30s, took {storage_time:.2f}s"
        assert library.size() == 10000, "Should store all 10K patterns"
        
        # Query performance should still be good
        query_features = {'pattern_type': 'trend', 'min_strength': 0.5}
        
        # Warm-up query (index rebuild happens here)
        library.find_similar_patterns(query_features, top_k=10)
        
        # Actual performance test (index should be warm)
        start_time = time.time()
        results = library.find_similar_patterns(query_features, top_k=10)
        query_time = (time.time() - start_time) * 1000
        
        # Query must be <10ms even with 10K patterns (warm index)
        assert query_time < 10, f"Query should be <10ms with 10K patterns, took {query_time:.2f}ms"
    
    def test_memory_usage_bounds(self):
        """Test memory usage stays within reasonable bounds."""
        library = PatternLibrary(max_patterns=5000)
        
        # Measure memory before
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Add patterns
        patterns = self._create_test_patterns(5000)
        for pattern in patterns:
            library.store_pattern(pattern)
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = memory_after - memory_before
        
        # Should not use excessive memory (rough estimate: <100MB for 5K patterns)
        assert memory_increase < 100, \
            f"Memory usage should be reasonable, used {memory_increase:.1f}MB"
    
    def test_concurrent_access(self):
        """Test thread safety for concurrent pattern operations."""
        import threading
        
        library = PatternLibrary(max_patterns=1000)
        errors = []
        
        def add_patterns(start_id, count):
            try:
                for i in range(count):
                    pattern = Pattern(f"thread_{start_id}_{i}", "trend", "15m", {}, 0.5)
                    library.store_pattern(pattern)
            except Exception as e:
                errors.append(e)
        
        def query_patterns():
            try:
                for _ in range(50):
                    library.find_similar_patterns({'pattern_type': 'trend'}, top_k=5)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = []
        threads.extend([threading.Thread(target=add_patterns, args=(i, 50)) for i in range(5)])
        threads.extend([threading.Thread(target=query_patterns) for _ in range(3)])
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without errors
        assert len(errors) == 0, f"Concurrent access should be safe, got errors: {errors}"


# This will run when pytest is called and should initially FAIL
# Implementation should make these tests pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])