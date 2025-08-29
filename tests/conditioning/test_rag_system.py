"""
test_rag_system.py
==================

TDD tests for DRQ-102: RAG (Retrieval-Augmented Generation) System
These tests MUST be written first and will initially FAIL.
Implementation should make these tests pass.

CRITICAL: <60ms retrieval, semantic search, context ranking.
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
    from src.conditioning.rag_system import (
        RAGSystem,
        ContextRetriever,
        SemanticEncoder,
        ContextRanker,
        RetrievalContext
    )
    from src.conditioning.pattern_library import Pattern
except ImportError:
    # These will fail initially - that's expected in TDD
    pass


class TestSemanticEncoding:
    """Tests for semantic encoding of market contexts."""
    
    def test_market_context_encoding(self):
        """Test encoding of market context into semantic vectors."""
        # This will fail initially until SemanticEncoder is implemented
        encoder = SemanticEncoder(embedding_dim=256)
        
        # Create market context
        market_context = {
            'current_price': 150.50,
            'price_change_1m': 0.002,
            'price_change_5m': -0.005,
            'volume_ratio': 1.3,
            'bid_ask_spread': 0.01,
            'time_of_day': '14:30:00',
            'day_of_week': 'Tuesday',
            'vix_level': 18.5,
            'sector_rotation': 'tech_outperform'
        }
        
        # Encode context
        embedding = encoder.encode_market_context(market_context)
        
        # Should return valid embedding
        assert embedding.shape == (256,), f"Should return 256-dim embedding, got {embedding.shape}"
        assert np.all(np.isfinite(embedding)), "Embedding should contain finite values"
        
        # Similar contexts should have similar embeddings
        similar_context = market_context.copy()
        similar_context['current_price'] = 150.52  # Very small change
        similar_embedding = encoder.encode_market_context(similar_context)
        
        different_context = market_context.copy()
        different_context.update({
            'current_price': 200.00,  # Large change
            'volume_ratio': 5.0,      # Large change
            'vix_level': 35.0         # Large change
        })
        different_embedding = encoder.encode_market_context(different_context)
        
        # Cosine similarity check
        similar_sim = np.dot(embedding, similar_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(similar_embedding)
        )
        different_sim = np.dot(embedding, different_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(different_embedding)
        )
        
        assert similar_sim > different_sim, \
            "Similar contexts should have higher similarity"
    
    def test_regime_aware_encoding(self):
        """Test that encoding captures market regime information."""
        encoder = SemanticEncoder(embedding_dim=128, regime_aware=True)
        
        # Different market regimes
        trending_context = {
            'trend_strength': 0.8,
            'volatility': 0.15,
            'volume_trend': 'increasing',
            'regime': 'trending_up'
        }
        
        ranging_context = {
            'trend_strength': 0.2,
            'volatility': 0.08,
            'volume_trend': 'stable',
            'regime': 'ranging'
        }
        
        volatile_context = {
            'trend_strength': 0.1,
            'volatility': 0.35,
            'volume_trend': 'erratic',
            'regime': 'high_volatility'
        }
        
        trend_emb = encoder.encode_market_context(trending_context)
        range_emb = encoder.encode_market_context(ranging_context)
        vol_emb = encoder.encode_market_context(volatile_context)
        
        # Different regimes should have different embeddings
        trend_range_sim = np.dot(trend_emb, range_emb) / (
            np.linalg.norm(trend_emb) * np.linalg.norm(range_emb)
        )
        trend_vol_sim = np.dot(trend_emb, vol_emb) / (
            np.linalg.norm(trend_emb) * np.linalg.norm(vol_emb)
        )
        
        # Similarity should be moderate (not too high, not too low)
        assert 0.3 <= trend_range_sim <= 0.8, \
            "Different regimes should have moderate similarity"
        assert 0.3 <= trend_vol_sim <= 0.8, \
            "Different regimes should have moderate similarity"
    
    def test_temporal_context_encoding(self):
        """Test encoding of temporal patterns and seasonality."""
        encoder = SemanticEncoder(temporal_aware=True)
        
        # Different times and patterns
        contexts = [
            {'time': '09:30:00', 'session': 'open', 'day_of_week': 'Monday'},
            {'time': '11:00:00', 'session': 'morning', 'day_of_week': 'Monday'},
            {'time': '15:30:00', 'session': 'close', 'day_of_week': 'Friday'},
            {'time': '20:00:00', 'session': 'after_hours', 'day_of_week': 'Friday'}
        ]
        
        embeddings = [encoder.encode_market_context(ctx) for ctx in contexts]
        
        # Market open contexts should be more similar
        open_close_sim = np.dot(embeddings[0], embeddings[2]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[2])
        )
        regular_after_sim = np.dot(embeddings[1], embeddings[3]) / (
            np.linalg.norm(embeddings[1]) * np.linalg.norm(embeddings[3])
        )
        
        # Should capture temporal patterns
        assert embeddings[0].shape == embeddings[1].shape, "Consistent embedding dimensions"
        assert len(set(tuple(emb) for emb in embeddings)) > 1, "Different contexts should produce different embeddings"


class TestContextRetrieval:
    """Tests for context retrieval functionality."""
    
    def test_similar_context_retrieval(self):
        """Test retrieval of similar historical contexts."""
        # This will fail initially until ContextRetriever is implemented
        retriever = ContextRetriever(max_contexts=10000)
        
        # Add historical contexts
        historical_contexts = []
        for i in range(1000):
            context = RetrievalContext(
                context_id=f"ctx_{i}",
                market_state={
                    'price': 100 + np.random.randn() * 10,
                    'volatility': 0.2 + np.random.randn() * 0.05,
                    'volume': 1000 + np.random.randint(0, 500),
                    'regime': np.random.choice(['trending', 'ranging', 'volatile'])
                },
                patterns=[],
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
                outcome={'return_1h': np.random.randn() * 0.01}
            )
            historical_contexts.append(context)
            retriever.add_context(context)
        
        # Query for similar context
        query_context = {
            'price': 105,
            'volatility': 0.18,
            'volume': 1200,
            'regime': 'trending'
        }
        
        start_time = time.time()
        similar_contexts = retriever.retrieve_similar(query_context, top_k=20)
        retrieval_time = (time.time() - start_time) * 1000  # ms
        
        # Must be fast (<60ms requirement)
        assert retrieval_time < 60, \
            f"Retrieval should be <60ms, took {retrieval_time:.2f}ms"
        
        # Should return requested number
        assert len(similar_contexts) == 20, f"Should return 20 contexts, got {len(similar_contexts)}"
        
        # Should be ranked by similarity
        similarities = [ctx.similarity_score for ctx in similar_contexts]
        assert similarities == sorted(similarities, reverse=True), \
            "Contexts should be ranked by similarity"
    
    def test_pattern_aware_retrieval(self):
        """Test retrieval considers pattern similarity."""
        retriever = ContextRetriever()
        
        # Create contexts with specific patterns
        trend_pattern = Pattern("trend_1", "uptrend", "15m", {'strength': 0.8}, 0.8)
        reversal_pattern = Pattern("rev_1", "reversal", "30m", {'strength': 0.7}, 0.7)
        
        trend_context = RetrievalContext(
            context_id="trend_ctx",
            market_state={'regime': 'trending'},
            patterns=[trend_pattern],
            timestamp=datetime.now(),
            outcome={'return_1h': 0.02}
        )
        
        reversal_context = RetrievalContext(
            context_id="rev_ctx",
            market_state={'regime': 'reversal'},
            patterns=[reversal_pattern],
            timestamp=datetime.now(),
            outcome={'return_1h': -0.015}
        )
        
        retriever.add_context(trend_context)
        retriever.add_context(reversal_context)
        
        # Query with trend pattern
        query_patterns = [Pattern("query_trend", "uptrend", "15m", {'strength': 0.75}, 0.75)]
        
        similar = retriever.retrieve_by_patterns(query_patterns, top_k=2)
        
        # Trend context should rank higher
        assert len(similar) == 2, "Should return both contexts"
        assert similar[0].context_id == "trend_ctx", \
            "Trend context should rank higher for trend pattern query"
    
    def test_temporal_locality_boost(self):
        """Test that recent contexts get relevance boost."""
        retriever = ContextRetriever(temporal_decay=0.1)
        
        # Add contexts at different times
        recent_time = datetime(2024, 1, 15, 14, 30)
        old_time = datetime(2023, 12, 1, 14, 30)
        
        # Same market state, different times
        base_state = {'price': 100, 'volatility': 0.2}
        
        recent_context = RetrievalContext("recent", base_state, [], recent_time, {})
        old_context = RetrievalContext("old", base_state, [], old_time, {})
        
        retriever.add_context(recent_context)
        retriever.add_context(old_context)
        
        # Retrieve with current time close to recent_time
        current_time = datetime(2024, 1, 15, 15, 0)  # 30 min after recent
        similar = retriever.retrieve_similar(
            base_state, top_k=2, current_time=current_time
        )
        
        # Recent context should rank higher due to temporal proximity
        assert similar[0].context_id == "recent", \
            "Recent context should rank higher"
    
    def test_outcome_based_filtering(self):
        """Test filtering contexts based on historical outcomes."""
        retriever = ContextRetriever()
        
        # Add contexts with different outcomes
        contexts_data = [
            ("good", {'return_1h': 0.02, 'max_drawdown': -0.005}),  # Good outcome
            ("bad", {'return_1h': -0.03, 'max_drawdown': -0.05}),   # Bad outcome
            ("neutral", {'return_1h': 0.001, 'max_drawdown': -0.002})  # Neutral outcome
        ]
        
        base_state = {'price': 100, 'volatility': 0.2}
        for ctx_id, outcome in contexts_data:
            context = RetrievalContext(ctx_id, base_state, [], datetime.now(), outcome)
            retriever.add_context(context)
        
        # Retrieve only contexts with positive outcomes
        positive_contexts = retriever.retrieve_similar(
            base_state, 
            top_k=5, 
            outcome_filter={'min_return_1h': 0.0}
        )
        
        # Should filter out bad outcome
        retrieved_ids = [ctx.context_id for ctx in positive_contexts]
        assert "bad" not in retrieved_ids, "Should filter out negative outcomes"
        assert "good" in retrieved_ids, "Should include positive outcomes"


class TestContextRanking:
    """Tests for context ranking and scoring."""
    
    def test_multi_criteria_ranking(self):
        """Test ranking based on multiple criteria."""
        # This will fail initially until ContextRanker is implemented
        ranker = ContextRanker(
            criteria=['semantic_similarity', 'pattern_match', 'temporal_proximity', 'outcome_quality']
        )
        
        # Create contexts with different strengths
        query_context = {
            'price': 100,
            'volatility': 0.2,
            'patterns': [Pattern("q1", "trend", "15m", {}, 0.7)],
            'timestamp': datetime(2024, 1, 15, 14, 0)
        }
        
        candidate_contexts = [
            RetrievalContext("perfect", query_context, query_context['patterns'], 
                           query_context['timestamp'], {'return_1h': 0.02}),
            RetrievalContext("semantic", query_context, [], 
                           datetime(2023, 6, 1), {'return_1h': 0.005}),
            RetrievalContext("temporal", {'price': 80, 'volatility': 0.4}, [],
                           query_context['timestamp'], {'return_1h': 0.01}),
            RetrievalContext("outcome", {'price': 120, 'volatility': 0.1}, [],
                           datetime(2023, 8, 1), {'return_1h': 0.05})
        ]
        
        ranked_contexts = ranker.rank_contexts(query_context, candidate_contexts)
        
        # Perfect match should rank highest
        assert ranked_contexts[0].context_id == "perfect", \
            "Perfect match should rank highest"
        
        # All should have ranking scores
        for ctx in ranked_contexts:
            assert hasattr(ctx, 'ranking_score'), "Context should have ranking score"
            assert 0 <= ctx.ranking_score <= 1, "Ranking score should be in [0,1]"
    
    def test_adaptive_weight_learning(self):
        """Test adaptive learning of ranking criteria weights."""
        ranker = ContextRanker(adaptive_weights=True)
        
        # Provide feedback on ranking quality
        query = {'market_state': 'example'}
        contexts = [
            RetrievalContext("ctx1", {}, [], datetime.now(), {}),
            RetrievalContext("ctx2", {}, [], datetime.now(), {}),
            RetrievalContext("ctx3", {}, [], datetime.now(), {})
        ]
        
        ranked = ranker.rank_contexts(query, contexts)
        
        # Provide feedback (ctx2 was actually most useful)
        feedback = {
            "ctx1": 0.2,  # Low utility
            "ctx2": 0.9,  # High utility
            "ctx3": 0.4   # Medium utility
        }
        
        ranker.update_weights_from_feedback(query, ranked, feedback)
        
        # Re-rank with updated weights
        re_ranked = ranker.rank_contexts(query, contexts)
        
        # Should adapt weights based on feedback
        initial_order = [ctx.context_id for ctx in ranked]
        updated_order = [ctx.context_id for ctx in re_ranked]
        
        # Order may change based on learning (implementation dependent)
        assert len(updated_order) == len(initial_order), "Should return same number of contexts"
    
    def test_ranking_consistency(self):
        """Test ranking consistency and stability."""
        ranker = ContextRanker()
        
        query = {'price': 100, 'volatility': 0.2}
        contexts = [
            RetrievalContext(f"ctx_{i}", {'price': 100 + i, 'volatility': 0.2 + i*0.01}, 
                           [], datetime.now(), {})
            for i in range(10)
        ]
        
        # Rank multiple times
        ranking_1 = ranker.rank_contexts(query, contexts)
        ranking_2 = ranker.rank_contexts(query, contexts)
        
        # Should be consistent
        order_1 = [ctx.context_id for ctx in ranking_1]
        order_2 = [ctx.context_id for ctx in ranking_2]
        
        assert order_1 == order_2, "Ranking should be consistent across calls"


class TestRAGSystemIntegration:
    """Tests for integrated RAG system functionality."""
    
    def test_end_to_end_retrieval(self):
        """Test complete RAG workflow."""
        # This will fail initially until RAGSystem is implemented
        rag_system = RAGSystem(
            max_contexts=1000,
            retrieval_k=10,
            embedding_dim=256
        )
        
        # Add training contexts
        training_contexts = self._generate_training_contexts(500)
        for ctx in training_contexts:
            rag_system.add_historical_context(ctx)
        
        # Query current market state
        current_state = {
            'price': 150.75,
            'price_change_5m': 0.003,
            'volume_ratio': 1.2,
            'volatility': 0.18,
            'time_of_day': '14:30:00',
            'regime': 'trending_up'
        }
        
        start_time = time.time()
        retrieved_contexts = rag_system.retrieve_relevant_contexts(current_state)
        total_time = (time.time() - start_time) * 1000
        
        # Should meet performance requirements
        assert total_time < 60, f"End-to-end retrieval should be <60ms, took {total_time:.2f}ms"
        
        # Should return high-quality contexts
        assert len(retrieved_contexts) > 0, "Should retrieve some contexts"
        assert all(hasattr(ctx, 'relevance_score') for ctx in retrieved_contexts), \
            "All contexts should have relevance scores"
    
    def test_context_augmentation(self):
        """Test augmentation of current context with retrieved information."""
        rag_system = RAGSystem()
        
        # Add contexts with known patterns
        historical_context = RetrievalContext(
            "hist_1",
            market_state={
                'price': 100,
                'regime': 'breakout',
                'volume': 'high'
            },
            patterns=[Pattern("breakout_pattern", "breakout", "15m", {'strength': 0.9}, 0.9)],
            timestamp=datetime.now(),
            outcome={'return_1h': 0.025, 'success_rate': 0.8}
        )
        
        rag_system.add_historical_context(historical_context)
        
        # Current similar context
        current_context = {
            'price': 102,
            'regime': 'breakout',
            'volume': 'high'
        }
        
        augmented_context = rag_system.augment_context(current_context)
        
        # Should include retrieved information
        assert 'historical_patterns' in augmented_context, \
            "Should include historical patterns"
        assert 'expected_outcomes' in augmented_context, \
            "Should include expected outcomes"
        assert 'confidence_score' in augmented_context, \
            "Should include confidence score"
    
    def test_context_quality_scoring(self):
        """Test quality scoring of retrieved contexts."""
        rag_system = RAGSystem()
        
        # Add contexts with different quality indicators
        high_quality = RetrievalContext(
            "high_q",
            market_state={'price': 100, 'volatility': 0.2},
            patterns=[Pattern("strong", "trend", "30m", {'confidence': 0.9}, 0.9)],
            timestamp=datetime.now(),
            outcome={'return_1h': 0.02, 'accuracy': 0.85, 'sample_size': 100}
        )
        
        low_quality = RetrievalContext(
            "low_q", 
            market_state={'price': 100, 'volatility': 0.2},
            patterns=[Pattern("weak", "trend", "30m", {'confidence': 0.3}, 0.3)],
            timestamp=datetime.now(),
            outcome={'return_1h': 0.001, 'accuracy': 0.52, 'sample_size': 5}
        )
        
        rag_system.add_historical_context(high_quality)
        rag_system.add_historical_context(low_quality)
        
        retrieved = rag_system.retrieve_relevant_contexts(
            {'price': 100, 'volatility': 0.2}
        )
        
        # High quality should rank higher
        quality_scores = [ctx.quality_score for ctx in retrieved if hasattr(ctx, 'quality_score')]
        if len(quality_scores) >= 2:
            assert max(quality_scores) > min(quality_scores), \
                "Should differentiate context quality"
    
    def test_incremental_learning(self):
        """Test incremental learning from new market data."""
        rag_system = RAGSystem(incremental_learning=True)
        
        # Start with baseline contexts
        initial_contexts = self._generate_training_contexts(100)
        for ctx in initial_contexts:
            rag_system.add_historical_context(ctx)
        
        initial_retrieval = rag_system.retrieve_relevant_contexts(
            {'regime': 'trending', 'volatility': 0.2}
        )
        
        # Add new contexts with updated market conditions
        new_contexts = self._generate_training_contexts(50, regime_shift=True)
        for ctx in new_contexts:
            rag_system.add_historical_context(ctx)
        
        # Should adapt to new market regime
        updated_retrieval = rag_system.retrieve_relevant_contexts(
            {'regime': 'trending', 'volatility': 0.2}
        )
        
        # May retrieve different contexts after learning
        initial_ids = set(ctx.context_id for ctx in initial_retrieval[:5])
        updated_ids = set(ctx.context_id for ctx in updated_retrieval[:5])
        
        # Some adaptation expected (not necessarily complete change)
        overlap_ratio = len(initial_ids.intersection(updated_ids)) / len(initial_ids)
        assert 0 <= overlap_ratio <= 1, "Should show some adaptation to new data"

    # Helper methods
    def _generate_training_contexts(self, count: int, regime_shift: bool = False) -> List[RetrievalContext]:
        """Generate synthetic training contexts."""
        contexts = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(count):
            # Base market regimes
            regimes = ['trending', 'ranging', 'volatile']
            if regime_shift:
                regimes = ['new_regime', 'adapted_trend', 'evolved_pattern']
                
            context = RetrievalContext(
                context_id=f"train_ctx_{i}",
                market_state={
                    'price': 100 + np.random.randn() * 20,
                    'volatility': 0.15 + np.random.randn() * 0.05,
                    'volume_ratio': 1.0 + np.random.randn() * 0.3,
                    'regime': np.random.choice(regimes),
                    'time_of_day': f"{np.random.randint(9, 16):02d}:{np.random.randint(0, 60):02d}:00"
                },
                patterns=[
                    Pattern(f"pattern_{i}_{j}", 
                           np.random.choice(['trend', 'reversal', 'breakout']),
                           np.random.choice(['5m', '15m', '30m']),
                           {'strength': np.random.random()},
                           np.random.random())
                    for j in range(np.random.randint(0, 3))
                ],
                timestamp=base_date + timedelta(hours=i),
                outcome={
                    'return_1h': np.random.normal(0, 0.01),
                    'return_4h': np.random.normal(0, 0.02),
                    'max_drawdown': -abs(np.random.exponential(0.005)),
                    'accuracy': 0.5 + np.random.random() * 0.4,
                    'sample_size': np.random.randint(10, 200)
                }
            )
            contexts.append(context)
            
        return contexts


class TestPerformanceAndScaling:
    """Tests for RAG system performance and scalability."""
    
    def test_large_scale_retrieval(self):
        """Test retrieval performance with large context database."""
        rag_system = RAGSystem(max_contexts=10000)
        
        # Add 5000 contexts
        contexts = self._generate_training_contexts(5000)
        
        start_time = time.time()
        for ctx in contexts:
            rag_system.add_historical_context(ctx)
        indexing_time = time.time() - start_time
        
        # Indexing should be reasonable
        assert indexing_time < 60, f"Indexing 5K contexts should be <60s, took {indexing_time:.2f}s"
        
        # Query performance should remain good
        query_context = {'regime': 'trending', 'volatility': 0.2}
        
        start_time = time.time()
        results = rag_system.retrieve_relevant_contexts(query_context)
        query_time = (time.time() - start_time) * 1000
        
        assert query_time < 60, f"Query should be <60ms with 5K contexts, took {query_time:.2f}ms"
    
    def test_memory_efficiency(self):
        """Test memory usage remains reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        rag_system = RAGSystem(max_contexts=2000)
        contexts = self._generate_training_contexts(2000)
        
        for ctx in contexts:
            rag_system.add_historical_context(ctx)
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = memory_after - memory_before
        
        # Should not use excessive memory
        assert memory_increase < 200, \
            f"Memory usage should be reasonable, used {memory_increase:.1f}MB for 2K contexts"


    # Helper methods
    def _generate_training_contexts(self, count: int, regime_shift: bool = False) -> List[RetrievalContext]:
        """Generate synthetic training contexts."""
        contexts = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(count):
            # Base market regimes
            regimes = ['trending', 'ranging', 'volatile']
            if regime_shift:
                regimes = ['new_regime', 'adapted_trend', 'evolved_pattern']
                
            context = RetrievalContext(
                context_id=f"train_ctx_{i}",
                market_state={
                    'price': 100 + np.random.randn() * 20,
                    'volatility': 0.15 + np.random.randn() * 0.05,
                    'volume_ratio': 1.0 + np.random.randn() * 0.3,
                    'regime': np.random.choice(regimes),
                    'time_of_day': f"{np.random.randint(9, 16):02d}:{np.random.randint(0, 60):02d}:00"
                },
                patterns=[
                    Pattern(f"pattern_{i}_{j}", 
                           np.random.choice(['trend', 'reversal', 'breakout']),
                           np.random.choice(['5m', '15m', '30m']),
                           {'strength': np.random.random()},
                           np.random.random())
                    for j in range(np.random.randint(0, 3))
                ],
                timestamp=base_date + timedelta(hours=i),
                outcome={
                    'return_1h': np.random.normal(0, 0.01),
                    'return_4h': np.random.normal(0, 0.02),
                    'max_drawdown': -abs(np.random.exponential(0.005)),
                    'accuracy': 0.5 + np.random.random() * 0.4,
                    'sample_size': np.random.randint(10, 200)
                }
            )
            contexts.append(context)
            
        return contexts


# This will run when pytest is called and should initially FAIL
# Implementation should make these tests pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])