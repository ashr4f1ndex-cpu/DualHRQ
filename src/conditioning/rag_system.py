"""
rag_system.py - RAG (Retrieval-Augmented Generation) System
===========================================================

Core RAG system functionality for DualHRQ 2.0
Implements fast semantic search, context ranking, and retrieval with <60ms performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import threading
import time
import uuid
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import warnings


class RetrievalContext:
    """Individual context for retrieval with scores and metadata."""
    
    def __init__(self, context_id: str, market_state: Dict[str, Any], 
                 patterns: List, timestamp: datetime, outcome: Dict[str, float]):
        self.context_id = context_id
        self.market_state = market_state
        self.patterns = patterns
        self.timestamp = timestamp
        self.outcome = outcome
        
        # Scores set by retrieval system
        self.similarity_score = 0.0
        self.relevance_score = 0.0
        self.quality_score = 0.0
        self.ranking_score = 0.0
        
        # Cached embedding for performance
        self._embedding = None
        
        # Advanced RAG features (DRQ-104)
        self.importance_weight = 0.0  # Episodic memory importance weighting
        self.temporal_order = 0       # Temporal ordering for episodic memory
        self.access_count = 0         # Track access frequency for importance
        self.last_accessed = timestamp
        self.compressed_summary = None # Compressed context summary
        self.linked_contexts = []     # Multi-hop reasoning links
        self.attention_weights = {}   # Attention-based selection weights
    
    def __repr__(self):
        return f"RetrievalContext(id={self.context_id}, score={self.relevance_score:.3f}, importance={self.importance_weight:.3f})"


class EpisodicMemoryManager:
    """Episodic memory with temporal ordering and importance weighting (DRQ-104)."""
    
    def __init__(self, max_episodes: int = 50000, decay_factor: float = 0.95):
        self.max_episodes = max_episodes
        self.decay_factor = decay_factor
        
        # Episode storage with temporal ordering
        self.episodes = {}  # episode_id -> RetrievalContext
        self.temporal_index = []  # List of (timestamp, episode_id) tuples, sorted
        self.importance_index = {}  # episode_id -> importance_score
        
        # Access tracking for importance weighting
        self.access_counts = defaultdict(int)
        self.last_access_times = {}
        
        # Memory consolidation parameters
        self.consolidation_threshold = 1000  # Consolidate after this many additions
        self.additions_since_consolidation = 0
        
        self._lock = threading.RLock()
    
    def add_episode(self, context: RetrievalContext) -> None:
        """Add episodic memory with temporal ordering and importance weighting."""
        with self._lock:
            # Check capacity and consolidate if needed
            if len(self.episodes) >= self.max_episodes:
                self._consolidate_memory()
            
            # Set temporal order
            context.temporal_order = len(self.temporal_index)
            
            # Calculate initial importance weight
            context.importance_weight = self._calculate_importance_weight(context)
            
            # Store episode
            self.episodes[context.context_id] = context
            
            # Update temporal index (maintain sorted order)
            bisect.insort(self.temporal_index, (context.timestamp, context.context_id))
            
            # Update importance index
            self.importance_index[context.context_id] = context.importance_weight
            
            self.additions_since_consolidation += 1
            
            # Periodic consolidation
            if self.additions_since_consolidation >= self.consolidation_threshold:
                self._consolidate_memory()
    
    def get_temporal_neighbors(self, target_timestamp: datetime, window_hours: int = 24) -> List[RetrievalContext]:
        """Get episodic contexts within temporal window."""
        with self._lock:
            start_time = target_timestamp - timedelta(hours=window_hours)
            end_time = target_timestamp + timedelta(hours=window_hours)
            
            # Binary search for efficient temporal range retrieval
            start_idx = bisect.bisect_left(self.temporal_index, (start_time, ''))
            end_idx = bisect.bisect_right(self.temporal_index, (end_time, 'zzz'))
            
            neighbors = []
            for i in range(start_idx, end_idx):
                if i < len(self.temporal_index):
                    _, episode_id = self.temporal_index[i]
                    if episode_id in self.episodes:
                        neighbors.append(self.episodes[episode_id])
            
            return neighbors
    
    def get_important_episodes(self, top_k: int = 100) -> List[RetrievalContext]:
        """Get most important episodes based on importance weighting."""
        with self._lock:
            # Sort by importance weight
            sorted_episodes = sorted(
                self.importance_index.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k]
            
            return [self.episodes[episode_id] for episode_id, _ in sorted_episodes 
                   if episode_id in self.episodes]
    
    def update_importance_weight(self, context_id: str) -> None:
        """Update importance weight based on access patterns."""
        with self._lock:
            if context_id in self.episodes:
                context = self.episodes[context_id]
                
                # Increment access count
                self.access_counts[context_id] += 1
                self.last_access_times[context_id] = datetime.now()
                context.access_count = self.access_counts[context_id]
                context.last_accessed = self.last_access_times[context_id]
                
                # Recalculate importance weight
                context.importance_weight = self._calculate_importance_weight(context)
                self.importance_index[context_id] = context.importance_weight
    
    def _calculate_importance_weight(self, context: RetrievalContext) -> float:
        """Calculate importance weight based on multiple factors."""
        weight = 0.0
        
        # Outcome quality (40% of importance)
        if context.outcome:
            return_quality = np.tanh(context.outcome.get('return_1h', 0) * 100)
            accuracy = context.outcome.get('accuracy', 0.5)
            sample_size = min(1.0, np.log10(context.outcome.get('sample_size', 1)) / 2)
            outcome_weight = (return_quality + (accuracy - 0.5) * 2) * sample_size
            weight += np.tanh(outcome_weight) * 0.4
        
        # Pattern strength (25% of importance)
        if context.patterns:
            pattern_strengths = [p.strength for p in context.patterns if hasattr(p, 'strength')]
            if pattern_strengths:
                weight += np.mean(pattern_strengths) * 0.25
        
        # Access frequency (20% of importance)
        access_factor = min(1.0, np.log10(context.access_count + 1) / 2)
        weight += access_factor * 0.20
        
        # Recency (15% of importance)
        hours_ago = (datetime.now() - context.timestamp).total_seconds() / 3600
        recency_factor = np.exp(-0.001 * hours_ago)  # Slow decay over 1000 hours
        weight += recency_factor * 0.15
        
        return max(0.0, min(1.0, weight))
    
    def _consolidate_memory(self) -> None:
        """Consolidate memory by removing least important episodes."""
        if len(self.episodes) < self.max_episodes * 0.9:
            return  # Only consolidate when near capacity
        
        # Get episodes to remove (bottom 10% by importance)
        sorted_by_importance = sorted(
            self.importance_index.items(),
            key=lambda x: x[1]
        )
        
        num_to_remove = int(len(self.episodes) * 0.1)
        to_remove = [episode_id for episode_id, _ in sorted_by_importance[:num_to_remove]]
        
        # Remove episodes
        for episode_id in to_remove:
            if episode_id in self.episodes:
                context = self.episodes[episode_id]
                
                # Remove from all indexes
                del self.episodes[episode_id]
                del self.importance_index[episode_id]
                
                # Remove from temporal index
                self.temporal_index = [
                    (ts, eid) for ts, eid in self.temporal_index if eid != episode_id
                ]
                
                # Clean up access tracking
                if episode_id in self.access_counts:
                    del self.access_counts[episode_id]
                if episode_id in self.last_access_times:
                    del self.last_access_times[episode_id]
        
        self.additions_since_consolidation = 0
        print(f"Memory consolidated: removed {len(to_remove)} episodes, {len(self.episodes)} remaining")


class AttentionMechanism(nn.Module):
    """Attention-based context selection with learned query representations (DRQ-104)."""
    
    def __init__(self, embedding_dim: int = 256, num_heads: int = 8, max_params: int = 50000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.max_params = max_params
        
        # Calculate optimal dimensions to stay within parameter budget
        # Total params ≈ 3 * embedding_dim * embedding_dim (Q, K, V projections)
        if 3 * embedding_dim * embedding_dim > max_params:
            # Reduce embedding dimension to fit budget
            max_dim = int(np.sqrt(max_params / 3))
            self.embedding_dim = min(embedding_dim, max_dim)
            self.head_dim = self.embedding_dim // num_heads
        
        # Multi-head attention components (parameter-efficient)
        self.query_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.key_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.value_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        
        # Context compression for efficiency
        self.context_compressor = nn.Linear(self.embedding_dim, self.embedding_dim // 2, bias=False)
        
        # Verify parameter budget
        total_params = sum(p.numel() for p in self.parameters())
        if total_params > max_params:
            raise ValueError(f"AttentionMechanism exceeds parameter budget: {total_params} > {max_params}")
        
        print(f"AttentionMechanism initialized with {total_params:,} parameters (limit: {max_params:,})")
    
    def forward(self, query_embedding: torch.Tensor, context_embeddings: torch.Tensor, 
                context_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention to select relevant contexts."""
        batch_size, num_contexts, embed_dim = context_embeddings.shape
        
        # Project to Q, K, V
        Q = self.query_proj(query_embedding)  # [batch_size, embed_dim]
        K = self.key_proj(context_embeddings)  # [batch_size, num_contexts, embed_dim]
        V = self.value_proj(context_embeddings)  # [batch_size, num_contexts, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_contexts, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_contexts, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if context_mask is not None:
            attention_scores = attention_scores.masked_fill(context_mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project to output
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, 1, self.embedding_dim
        )
        
        output = self.out_proj(attended_values)
        
        # Return output and attention weights for interpretability
        return output.squeeze(1), attention_weights.mean(dim=1).squeeze(1)  # Average over heads


class ContextCompressor:
    """Context compression with summarization for storage efficiency (DRQ-104)."""
    
    def __init__(self, compression_ratio: float = 0.1, max_summary_length: int = 100):
        self.compression_ratio = compression_ratio
        self.max_summary_length = max_summary_length
        
        # Key extraction patterns for market contexts
        self.key_features = [
            'current_price', 'price_change_1m', 'price_change_5m', 'volume_ratio',
            'volatility', 'trend_strength', 'regime', 'session', 'patterns'
        ]
    
    def compress_context(self, context: RetrievalContext) -> str:
        """Compress context into a summary string."""
        summary_parts = []
        
        # Extract key market state information
        market_state = context.market_state
        
        # Price information
        if 'current_price' in market_state:
            price_info = f"Price: {market_state['current_price']:.2f}"
            if 'price_change_1m' in market_state:
                price_info += f" (1m: {market_state['price_change_1m']*100:+.1f}%)"
            summary_parts.append(price_info)
        
        # Volume and volatility
        vol_info = []
        if 'volume_ratio' in market_state:
            vol_info.append(f"Vol: {market_state['volume_ratio']:.1f}x")
        if 'volatility' in market_state:
            vol_info.append(f"Vola: {market_state['volatility']:.3f}")
        if vol_info:
            summary_parts.append(" ".join(vol_info))
        
        # Regime and trend
        regime_info = []
        if 'regime' in market_state:
            regime_info.append(f"Regime: {market_state['regime']}")
        if 'trend_strength' in market_state:
            regime_info.append(f"Trend: {market_state['trend_strength']:.2f}")
        if regime_info:
            summary_parts.append(" ".join(regime_info))
        
        # Patterns (compressed)
        if context.patterns:
            pattern_types = [p.pattern_type for p in context.patterns[:3] if hasattr(p, 'pattern_type')]
            if pattern_types:
                summary_parts.append(f"Patterns: {','.join(pattern_types)}")
        
        # Outcome summary
        if context.outcome:
            outcome_items = []
            if 'return_1h' in context.outcome:
                outcome_items.append(f"R1h: {context.outcome['return_1h']*100:+.1f}%")
            if 'accuracy' in context.outcome:
                outcome_items.append(f"Acc: {context.outcome['accuracy']:.2f}")
            if outcome_items:
                summary_parts.append(f"Outcome: {' '.join(outcome_items)}")
        
        # Join and truncate
        summary = " | ".join(summary_parts)
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length-3] + "..."
        
        return summary
    
    def decompress_context(self, compressed_summary: str, original_context: RetrievalContext) -> Dict[str, Any]:
        """Attempt to reconstruct key information from compressed summary."""
        # For now, return original context as decompression is complex
        # In a full implementation, this would parse the summary back to structured data
        return original_context.market_state.copy()


class MultiHopReasoner:
    """Multi-hop reasoning capabilities across retrieved contexts (DRQ-104)."""
    
    def __init__(self, max_hops: int = 3, similarity_threshold: float = 0.7):
        self.max_hops = max_hops
        self.similarity_threshold = similarity_threshold
    
    def build_context_graph(self, contexts: List[RetrievalContext]) -> Dict[str, List[str]]:
        """Build a graph of related contexts for multi-hop reasoning."""
        graph = defaultdict(list)
        
        # Build similarity matrix
        n = len(contexts)
        if n < 2:
            return graph
        
        # Extract embeddings for similarity computation
        embeddings = []
        for ctx in contexts:
            if ctx._embedding is not None:
                embeddings.append(ctx._embedding)
            else:
                # Generate a simple embedding from market state
                embedding = self._generate_simple_embedding(ctx.market_state)
                embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Compute pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # Build graph based on similarity threshold
        for i in range(n):
            for j in range(i+1, n):
                if similarities[i, j] >= self.similarity_threshold:
                    ctx_i_id = contexts[i].context_id
                    ctx_j_id = contexts[j].context_id
                    graph[ctx_i_id].append(ctx_j_id)
                    graph[ctx_j_id].append(ctx_i_id)
        
        return dict(graph)
    
    def find_reasoning_paths(self, start_context_id: str, target_features: Dict[str, Any], 
                           context_graph: Dict[str, List[str]], 
                           contexts_by_id: Dict[str, RetrievalContext]) -> List[List[str]]:
        """Find reasoning paths from start context to contexts matching target features."""
        paths = []
        visited = set()
        
        def dfs(current_id: str, path: List[str], remaining_hops: int):
            if remaining_hops <= 0 or current_id in visited:
                return
            
            visited.add(current_id)
            path.append(current_id)
            
            # Check if current context matches target features
            if current_id in contexts_by_id:
                current_context = contexts_by_id[current_id]
                if self._matches_target_features(current_context, target_features):
                    paths.append(path.copy())
            
            # Explore neighbors
            for neighbor_id in context_graph.get(current_id, []):
                if neighbor_id not in visited:
                    dfs(neighbor_id, path.copy(), remaining_hops - 1)
        
        dfs(start_context_id, [], self.max_hops)
        return paths
    
    def aggregate_multi_hop_insights(self, reasoning_paths: List[List[str]], 
                                   contexts_by_id: Dict[str, RetrievalContext]) -> Dict[str, Any]:
        """Aggregate insights from multi-hop reasoning paths."""
        insights = {
            'connected_outcomes': [],
            'pattern_chains': [],
            'regime_transitions': [],
            'confidence_scores': []
        }
        
        for path in reasoning_paths:
            if len(path) < 2:
                continue
            
            # Extract outcomes along the path
            path_outcomes = []
            path_patterns = []
            path_regimes = []
            
            for ctx_id in path:
                if ctx_id in contexts_by_id:
                    ctx = contexts_by_id[ctx_id]
                    
                    if ctx.outcome:
                        path_outcomes.append(ctx.outcome)
                    
                    if ctx.patterns:
                        path_patterns.extend(ctx.patterns)
                    
                    regime = ctx.market_state.get('regime', 'unknown')
                    path_regimes.append(regime)
            
            if path_outcomes:
                insights['connected_outcomes'].append(path_outcomes)
            
            if path_patterns:
                insights['pattern_chains'].append(path_patterns)
            
            if path_regimes:
                insights['regime_transitions'].append(path_regimes)
            
            # Calculate path confidence (based on number of hops and context quality)
            confidence = 1.0 / len(path)  # Shorter paths are more reliable
            avg_quality = np.mean([contexts_by_id[ctx_id].quality_score for ctx_id in path 
                                 if ctx_id in contexts_by_id and contexts_by_id[ctx_id].quality_score > 0])
            if avg_quality > 0:
                confidence *= avg_quality
            
            insights['confidence_scores'].append(confidence)
        
        return insights
    
    def _generate_simple_embedding(self, market_state: Dict[str, Any]) -> np.ndarray:
        """Generate simple embedding from market state for similarity comparison."""
        features = []
        
        # Numeric features
        for key in ['current_price', 'price_change_1m', 'volume_ratio', 'volatility', 'trend_strength']:
            features.append(market_state.get(key, 0.0))
        
        # Categorical features (simple encoding)
        regime = market_state.get('regime', 'unknown')
        regime_encoding = {'trending_up': 1, 'trending_down': -1, 'ranging': 0, 'high_volatility': 2}.get(regime, 0)
        features.append(regime_encoding)
        
        session = market_state.get('session', 'regular')
        session_encoding = {'open': 1, 'close': -1, 'regular': 0}.get(session, 0)
        features.append(session_encoding)
        
        # Pad to consistent length
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10], dtype=np.float32)
    
    def _matches_target_features(self, context: RetrievalContext, target_features: Dict[str, Any]) -> bool:
        """Check if context matches target features for reasoning."""
        for key, target_value in target_features.items():
            if key == 'pattern_type' and context.patterns:
                pattern_types = [p.pattern_type for p in context.patterns if hasattr(p, 'pattern_type')]
                if target_value not in pattern_types:
                    return False
            elif key == 'regime':
                if context.market_state.get('regime') != target_value:
                    return False
            elif key == 'min_return' and context.outcome:
                if context.outcome.get('return_1h', 0) < target_value:
                    return False
        
        return True


class SemanticEncoder:
    """Semantic encoding of market contexts with regime and temporal awareness."""
    
    def __init__(self, embedding_dim: int = 256, regime_aware: bool = False, 
                 temporal_aware: bool = False):
        self.embedding_dim = embedding_dim
        self.regime_aware = regime_aware
        self.temporal_aware = temporal_aware
        
        # Scalers for different feature types
        self._price_scaler = MinMaxScaler()
        self._volume_scaler = MinMaxScaler()
        self._ratio_scaler = StandardScaler()
        self._time_scaler = MinMaxScaler()
        
        # Feature dimension allocation
        self.base_features_dim = 10
        self.regime_features_dim = 8 if regime_aware else 0
        self.temporal_features_dim = 6 if temporal_aware else 0
        self.remaining_dim = embedding_dim - self.base_features_dim - self.regime_features_dim - self.temporal_features_dim
        
        # Pre-fitted with reasonable ranges
        self._initialize_scalers()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def _initialize_scalers(self):
        """Initialize scalers with reasonable market data ranges."""
        # Price features: typical range 50-500
        self._price_scaler.fit(np.array([[50], [500]]))
        
        # Volume features: typical range 0.1-10.0 (ratios)
        self._volume_scaler.fit(np.array([[0.1], [10.0]]))
        
        # Ratio features: typical range -3 to 3 std devs
        self._ratio_scaler.fit(np.array([[-3], [3]]))
        
        # Time features: 0-24 hours, 0-7 days
        self._time_scaler.fit(np.array([[0, 0], [24, 7]]))
    
    def encode_market_context(self, market_context: Dict[str, Any]) -> np.ndarray:
        """Encode market context into semantic vector."""
        with self._lock:
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            idx = 0
            
            # Base market features
            base_features = self._extract_base_features(market_context)
            embedding[idx:idx+len(base_features)] = base_features
            idx += self.base_features_dim
            
            # Regime-aware features
            if self.regime_aware:
                regime_features = self._extract_regime_features(market_context)
                embedding[idx:idx+len(regime_features)] = regime_features
                idx += self.regime_features_dim
            
            # Temporal features
            if self.temporal_aware:
                temporal_features = self._extract_temporal_features(market_context)
                embedding[idx:idx+len(temporal_features)] = temporal_features
                idx += self.temporal_features_dim
            
            # Fill remaining with interaction features
            if self.remaining_dim > 0:
                interaction_features = self._extract_interaction_features(market_context, base_features)
                n_interactions = min(len(interaction_features), self.remaining_dim)
                embedding[idx:idx+n_interactions] = interaction_features[:n_interactions]
            
            return embedding
    
    def _extract_base_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract core market features."""
        features = np.zeros(self.base_features_dim, dtype=np.float32)
        
        # Price-based features
        current_price = context.get('current_price', 100.0)
        features[0] = self._price_scaler.transform([[current_price]])[0, 0]
        
        # Price changes (normalized)
        features[1] = np.tanh(context.get('price_change_1m', 0.0) * 1000)  # Scale to reasonable range
        features[2] = np.tanh(context.get('price_change_5m', 0.0) * 200)
        
        # Volume features
        volume_ratio = context.get('volume_ratio', 1.0)
        features[3] = self._volume_scaler.transform([[volume_ratio]])[0, 0]
        
        # Spread and volatility
        features[4] = np.tanh(context.get('bid_ask_spread', 0.01) * 1000)
        features[5] = np.tanh(context.get('vix_level', 20.0) / 50.0)  # VIX typically 10-50
        
        # Trend strength and momentum
        features[6] = np.tanh(context.get('trend_strength', 0.0))
        features[7] = np.tanh(context.get('volatility', 0.2) * 5)  # Scale volatility
        
        # Market session indicators
        features[8] = float('open' in context.get('session', '').lower())
        features[9] = float('close' in context.get('session', '').lower())
        
        return features
    
    def _extract_regime_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract regime-aware features."""
        features = np.zeros(self.regime_features_dim, dtype=np.float32)
        
        # Regime type encoding
        regime = context.get('regime', 'unknown')
        regime_encodings = {
            'trending_up': [1, 0, 0, 0],
            'trending_down': [0, 1, 0, 0],
            'ranging': [0, 0, 1, 0],
            'high_volatility': [0, 0, 0, 1],
            'trending': [0.5, 0.5, 0, 0],  # Generic trending
            'volatile': [0, 0, 0, 1]
        }
        
        encoding = regime_encodings.get(regime, [0, 0, 0, 0])
        features[:4] = encoding
        
        # Regime strength and confidence
        features[4] = context.get('trend_strength', 0.0)
        features[5] = context.get('regime_confidence', 0.5)
        
        # Sector rotation and market conditions  
        features[6] = float('outperform' in context.get('sector_rotation', '').lower())
        features[7] = float('underperform' in context.get('sector_rotation', '').lower())
        
        return features
    
    def _extract_temporal_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract temporal patterns and seasonality."""
        features = np.zeros(self.temporal_features_dim, dtype=np.float32)
        
        # Time of day features
        time_str = context.get('time_of_day', '12:00:00')
        if isinstance(time_str, str) and ':' in time_str:
            try:
                hour = float(time_str.split(':')[0])
                minute = float(time_str.split(':')[1])
                time_features = self._time_scaler.transform([[hour, minute/60]])[0]
                features[0] = time_features[0]
                features[1] = np.sin(2 * np.pi * hour / 24)  # Cyclical encoding
                features[2] = np.cos(2 * np.pi * hour / 24)
            except:
                features[0:3] = 0.5  # Default to midday
        
        # Day of week encoding
        day_mapping = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        day_of_week = context.get('day_of_week', 'wednesday').lower()
        if day_of_week in day_mapping:
            day_num = day_mapping[day_of_week]
            features[3] = np.sin(2 * np.pi * day_num / 7)
            features[4] = np.cos(2 * np.pi * day_num / 7)
        
        # Market session features
        session = context.get('session', 'regular').lower()
        features[5] = float(session in ['open', 'close', 'after_hours'])
        
        return features
    
    def _extract_interaction_features(self, context: Dict[str, Any], base_features: np.ndarray) -> np.ndarray:
        """Generate interaction features between market variables."""
        interactions = []
        
        # Price-volume interactions
        price_change = context.get('price_change_1m', 0.0)
        volume_ratio = context.get('volume_ratio', 1.0)
        interactions.append(price_change * volume_ratio)
        interactions.append(price_change * context.get('volatility', 0.2))
        
        # Trend-volatility interactions
        trend_strength = context.get('trend_strength', 0.0)
        volatility = context.get('volatility', 0.2)
        interactions.append(trend_strength * volatility)
        interactions.append(trend_strength * volume_ratio)
        
        # VIX interactions
        vix = context.get('vix_level', 20.0) / 50.0
        interactions.append(vix * volatility)
        interactions.append(vix * abs(price_change))
        
        # Add squares of key features
        interactions.extend([f**2 for f in base_features[:4]])
        
        # Cross-products of normalized features
        for i in range(min(4, len(base_features))):
            for j in range(i+1, min(4, len(base_features))):
                interactions.append(base_features[i] * base_features[j])
        
        return np.array(interactions[:self.remaining_dim], dtype=np.float32)


class ContextRetriever:
    """Context retrieval with similarity search and performance optimization."""
    
    def __init__(self, max_contexts: int = 10000, temporal_decay: float = 0.1):
        self.max_contexts = max_contexts
        self.temporal_decay = temporal_decay
        
        # Storage
        self._contexts: Dict[str, RetrievalContext] = {}
        self._embeddings = np.array([]).reshape(0, 256)  # Default embedding size
        self._context_ids = []
        
        # Fast retrieval indexes
        self._nn_index = None
        self._scaler = StandardScaler()
        self._encoder = SemanticEncoder()
        
        # Performance tracking
        self._rebuild_threshold = 100  # Rebuild index after this many additions
        self._additions_since_rebuild = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def add_context(self, context: RetrievalContext) -> None:
        """Add context to retrieval database."""
        with self._lock:
            # Check capacity
            if len(self._contexts) >= self.max_contexts:
                self._evict_oldest_context()
            
            # Store context
            self._contexts[context.context_id] = context
            
            # Generate embedding if not cached
            if context._embedding is None:
                context._embedding = self._encoder.encode_market_context(context.market_state)
            
            # Update embeddings array
            if len(self._embeddings) == 0:
                self._embeddings = context._embedding.reshape(1, -1)
                self._context_ids = [context.context_id]
            else:
                self._embeddings = np.vstack([self._embeddings, context._embedding])
                self._context_ids.append(context.context_id)
            
            self._additions_since_rebuild += 1
            
            # Rebuild index periodically for performance
            if self._additions_since_rebuild >= self._rebuild_threshold:
                self._rebuild_index()
    
    def retrieve_similar(self, query_context: Dict[str, Any], top_k: int = 10,
                        current_time: datetime = None, outcome_filter: Dict[str, float] = None) -> List[RetrievalContext]:
        """Retrieve similar contexts with temporal weighting."""
        start_time = time.time()
        
        with self._lock:
            if len(self._contexts) == 0:
                return []
            
            # Encode query
            query_embedding = self._encoder.encode_market_context(query_context)
            
            # Apply outcome filters first (fast pre-filter)
            candidate_contexts = self._apply_outcome_filter(outcome_filter) if outcome_filter else list(self._contexts.values())
            
            if not candidate_contexts:
                return []
            
            # Compute similarities efficiently
            candidate_embeddings = np.array([ctx._embedding for ctx in candidate_contexts])
            similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings)[0]
            
            # Apply temporal weighting if current_time provided
            if current_time:
                temporal_weights = self._compute_temporal_weights(candidate_contexts, current_time)
                similarities = similarities * temporal_weights
            
            # Get top-k indices
            if len(similarities) <= top_k:
                top_indices = np.arange(len(similarities))
            else:
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                
            # Sort by similarity descending
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
            
            # Return contexts with scores
            results = []
            for idx in top_indices:
                ctx = candidate_contexts[idx]
                ctx.similarity_score = similarities[idx]
                ctx.relevance_score = similarities[idx]  # Will be updated by ranker
                results.append(ctx)
            
            # Performance check
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= 60:
                warnings.warn(f"Retrieval took {elapsed_ms:.2f}ms, exceeding 60ms requirement")
            
            return results
    
    def retrieve_by_patterns(self, query_patterns: List, top_k: int = 10) -> List[RetrievalContext]:
        """Retrieve contexts by pattern similarity."""
        if not query_patterns or len(self._contexts) == 0:
            return []
        
        # Score contexts by pattern matching
        context_scores = {}
        
        for ctx in self._contexts.values():
            score = self._compute_pattern_similarity(query_patterns, ctx.patterns)
            # Include all contexts with any pattern match (score > 0)
            # This allows the ranker to decide final ordering
            context_scores[ctx.context_id] = (ctx, score)
        
        # Sort by score and return top-k (including zero scores if needed)
        sorted_contexts = sorted(context_scores.values(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for ctx, score in sorted_contexts:
            ctx.similarity_score = score
            ctx.relevance_score = score
            results.append(ctx)
        
        return results
    
    def _apply_outcome_filter(self, outcome_filter: Dict[str, float]) -> List[RetrievalContext]:
        """Filter contexts based on historical outcomes."""
        filtered = []
        
        for ctx in self._contexts.values():
            passes_filter = True
            
            for metric, threshold in outcome_filter.items():
                if metric.startswith('min_'):
                    actual_metric = metric[4:]  # Remove 'min_' prefix
                    if ctx.outcome.get(actual_metric, float('-inf')) < threshold:
                        passes_filter = False
                        break
                elif metric.startswith('max_'):
                    actual_metric = metric[4:]  # Remove 'max_' prefix
                    if ctx.outcome.get(actual_metric, float('inf')) > threshold:
                        passes_filter = False
                        break
            
            if passes_filter:
                filtered.append(ctx)
        
        return filtered
    
    def _compute_temporal_weights(self, contexts: List[RetrievalContext], current_time: datetime) -> np.ndarray:
        """Compute temporal proximity weights for contexts."""
        weights = np.ones(len(contexts))
        
        for i, ctx in enumerate(contexts):
            time_diff = abs((current_time - ctx.timestamp).total_seconds()) / 3600  # Hours
            weights[i] = np.exp(-self.temporal_decay * time_diff)
        
        return weights
    
    def _compute_pattern_similarity(self, query_patterns: List, context_patterns: List) -> float:
        """Compute similarity between pattern sets."""
        if not query_patterns:
            return 0.5  # Neutral score when no query patterns
        if not context_patterns:
            return 0.1  # Low but non-zero score for contexts with no patterns
        
        total_similarity = 0.0
        
        for query_pattern in query_patterns:
            best_match = 0.0
            for ctx_pattern in context_patterns:
                similarity = self._pattern_similarity(query_pattern, ctx_pattern)
                best_match = max(best_match, similarity)
            
            # Always add the best match (even if 0) to avoid empty results
            total_similarity += best_match
        
        return total_similarity / len(query_patterns)  # Normalize by query patterns
    
    def _pattern_similarity(self, pattern1, pattern2) -> float:
        """Compute similarity between two individual patterns."""
        if not hasattr(pattern1, 'pattern_type') or not hasattr(pattern2, 'pattern_type'):
            return 0.0
        
        # Handle similar pattern types (uptrend vs trend, etc.)
        type1 = pattern1.pattern_type.lower()
        type2 = pattern2.pattern_type.lower()
        
        # Type match with fuzzy matching for related types
        if type1 == type2:
            type_match = 1.0
        elif ('trend' in type1 and 'trend' in type2) or ('uptrend' in type1 and 'trend' in type2) or ('trend' in type1 and 'uptrend' in type2):
            type_match = 0.8  # High similarity for trend variants
        else:
            type_match = 0.3
        
        # Scale match bonus  
        scale_match = 1.0 if pattern1.scale == pattern2.scale else 0.7
        
        # Strength similarity
        strength_sim = 1.0 - abs(pattern1.strength - pattern2.strength)
        
        return type_match * scale_match * strength_sim
    
    def _evict_oldest_context(self):
        """Remove oldest context to maintain capacity."""
        if not self._contexts:
            return
        
        # Find oldest context
        oldest_id = min(self._contexts.keys(), 
                       key=lambda x: self._contexts[x].timestamp)
        
        # Remove from storage
        del self._contexts[oldest_id]
        
        # Remove from embeddings (requires rebuild for efficiency)
        self._additions_since_rebuild = self._rebuild_threshold  # Force rebuild
    
    def _rebuild_index(self):
        """Rebuild similarity search index for performance."""
        if len(self._contexts) == 0:
            return
        
        # Rebuild embeddings array
        embeddings = []
        context_ids = []
        
        for ctx_id, ctx in self._contexts.items():
            if ctx._embedding is not None:
                embeddings.append(ctx._embedding)
                context_ids.append(ctx_id)
        
        if embeddings:
            self._embeddings = np.array(embeddings)
            self._context_ids = context_ids
            
            # Rebuild NN index if we have enough contexts
            if len(embeddings) >= 10:
                self._nn_index = NearestNeighbors(
                    n_neighbors=min(50, len(embeddings)),
                    metric='cosine',
                    algorithm='auto'
                )
                self._nn_index.fit(self._embeddings)
        
        self._additions_since_rebuild = 0


class ContextRanker:
    """Context ranking with multiple criteria and adaptive weights."""
    
    def __init__(self, criteria: List[str] = None, adaptive_weights: bool = False):
        self.criteria = criteria or ['semantic_similarity', 'pattern_match', 'temporal_proximity', 'outcome_quality']
        self.adaptive_weights = adaptive_weights
        
        # Default weights for ranking criteria
        self.weights = {
            'semantic_similarity': 0.35,
            'pattern_match': 0.25,
            'temporal_proximity': 0.15,
            'outcome_quality': 0.25
        }
        
        # Adaptive learning state
        if adaptive_weights:
            self._feedback_history = []
            self._weight_update_rate = 0.1
        
        self._lock = threading.Lock()
    
    def rank_contexts(self, query_context: Dict[str, Any], 
                     candidate_contexts: List[RetrievalContext]) -> List[RetrievalContext]:
        """Rank contexts by relevance using multiple criteria."""
        if not candidate_contexts:
            return []
        
        # Compute scores for each criterion
        scores = {}
        
        if 'semantic_similarity' in self.criteria:
            scores['semantic_similarity'] = self._compute_semantic_scores(candidate_contexts)
        
        if 'pattern_match' in self.criteria:
            scores['pattern_match'] = self._compute_pattern_scores(query_context, candidate_contexts)
        
        if 'temporal_proximity' in self.criteria:
            scores['temporal_proximity'] = self._compute_temporal_scores(query_context, candidate_contexts)
        
        if 'outcome_quality' in self.criteria:
            scores['outcome_quality'] = self._compute_outcome_scores(candidate_contexts)
        
        # Combine scores using weights
        final_scores = []
        for i, ctx in enumerate(candidate_contexts):
            combined_score = 0.0
            for criterion in self.criteria:
                if criterion in scores:
                    combined_score += self.weights.get(criterion, 0.0) * scores[criterion][i]
            
            ctx.ranking_score = combined_score
            ctx.relevance_score = combined_score  # Update relevance score
            final_scores.append((ctx, combined_score))
        
        # Sort by combined score descending
        ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)
        return [ctx for ctx, score in ranked]
    
    def update_weights_from_feedback(self, query: Dict[str, Any], 
                                   ranked_contexts: List[RetrievalContext],
                                   feedback: Dict[str, float]) -> None:
        """Update ranking weights based on user feedback."""
        if not self.adaptive_weights:
            return
        
        with self._lock:
            # Store feedback for learning
            self._feedback_history.append({
                'query': query.copy(),
                'contexts': [ctx.context_id for ctx in ranked_contexts],
                'feedback': feedback.copy()
            })
            
            # Limit history size
            if len(self._feedback_history) > 100:
                self._feedback_history = self._feedback_history[-50:]
            
            # Update weights based on feedback patterns
            self._update_weights_from_history()
    
    def _compute_semantic_scores(self, contexts: List[RetrievalContext]) -> List[float]:
        """Extract semantic similarity scores (already computed during retrieval)."""
        return [ctx.similarity_score for ctx in contexts]
    
    def _compute_pattern_scores(self, query_context: Dict[str, Any], 
                              contexts: List[RetrievalContext]) -> List[float]:
        """Compute pattern matching scores."""
        scores = []
        query_patterns = query_context.get('patterns', [])
        
        for ctx in contexts:
            if not query_patterns or not ctx.patterns:
                scores.append(0.0)
            else:
                # Pattern type matching
                query_types = set(p.pattern_type for p in query_patterns if hasattr(p, 'pattern_type'))
                ctx_types = set(p.pattern_type for p in ctx.patterns if hasattr(p, 'pattern_type'))
                
                if query_types and ctx_types:
                    type_overlap = len(query_types.intersection(ctx_types)) / len(query_types.union(ctx_types))
                    scores.append(type_overlap)
                else:
                    scores.append(0.0)
        
        return scores
    
    def _compute_temporal_scores(self, query_context: Dict[str, Any], 
                               contexts: List[RetrievalContext]) -> List[float]:
        """Compute temporal proximity scores."""
        scores = []
        current_time = query_context.get('timestamp', datetime.now())
        
        if not isinstance(current_time, datetime):
            return [0.5] * len(contexts)  # Neutral score if no time info
        
        for ctx in contexts:
            time_diff_hours = abs((current_time - ctx.timestamp).total_seconds()) / 3600
            # Exponential decay: more recent = higher score
            score = np.exp(-0.01 * time_diff_hours)  # Decay over ~100 hours
            scores.append(score)
        
        return scores
    
    def _compute_outcome_scores(self, contexts: List[RetrievalContext]) -> List[float]:
        """Compute outcome quality scores."""
        scores = []
        
        for ctx in contexts:
            outcome = ctx.outcome
            if not outcome:
                scores.append(0.5)  # Neutral score for no outcome data
                continue
            
            # Combine multiple outcome metrics
            quality_score = 0.0
            
            # Return-based scoring
            return_1h = outcome.get('return_1h', 0.0)
            quality_score += np.tanh(return_1h * 100) * 0.4  # Scale and bound
            
            # Risk-adjusted scoring (penalize high drawdowns)
            max_drawdown = outcome.get('max_drawdown', 0.0)
            quality_score -= abs(max_drawdown) * 2.0  # Penalty for drawdown
            
            # Accuracy/confidence scoring
            accuracy = outcome.get('accuracy', 0.5)
            quality_score += (accuracy - 0.5) * 0.6  # Bonus for high accuracy
            
            # Sample size weighting (more data = higher confidence)
            sample_size = outcome.get('sample_size', 1)
            size_weight = min(1.0, np.log10(sample_size) / 2)  # Log scaling
            quality_score *= size_weight
            
            # Normalize to [0, 1]
            quality_score = (np.tanh(quality_score) + 1) / 2
            scores.append(quality_score)
        
        return scores
    
    def _update_weights_from_history(self):
        """Update weights based on feedback history using simple gradient descent."""
        if len(self._feedback_history) < 5:  # Need minimum feedback
            return
        
        # Analyze recent feedback patterns
        recent_feedback = self._feedback_history[-10:]
        
        # Simple weight adjustment based on feedback correlation
        weight_adjustments = {criterion: 0.0 for criterion in self.criteria}
        
        for feedback_entry in recent_feedback:
            feedback_scores = feedback_entry['feedback']
            
            # Compute correlation between each criterion and user satisfaction
            for criterion in self.criteria:
                # Simple proxy: high feedback scores should correlate with our rankings
                avg_feedback = np.mean(list(feedback_scores.values()))
                
                if avg_feedback > 0.6:  # Good feedback
                    weight_adjustments[criterion] += self._weight_update_rate * 0.1
                elif avg_feedback < 0.4:  # Poor feedback
                    weight_adjustments[criterion] -= self._weight_update_rate * 0.1
        
        # Apply weight adjustments
        total_adjustment = sum(abs(adj) for adj in weight_adjustments.values())
        if total_adjustment > 0:
            for criterion in self.criteria:
                self.weights[criterion] += weight_adjustments[criterion]
            
            # Renormalize weights
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for criterion in self.weights:
                    self.weights[criterion] /= total_weight


class PatternRAG(nn.Module):
    """Lightweight neural RAG component with ≤0.1M parameters."""
    
    def __init__(self, pattern_dim: int = 128, context_dim: int = 256, hidden_dim: int = 64,
                 max_params: int = 100_000):
        super().__init__()
        self.pattern_dim = pattern_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.max_params = max_params
        
        # Calculate optimal dimensions to stay within parameter budget
        # Total params = pattern_dim * hidden_dim + hidden_dim * context_dim + context_dim * 1
        # We need: pattern_dim * hidden_dim + hidden_dim * context_dim + context_dim ≤ max_params
        
        # Optimize hidden_dim to fit budget
        optimal_hidden = self._calculate_optimal_hidden_dim()
        self.hidden_dim = min(hidden_dim, optimal_hidden)
        
        # Ultra-lightweight architecture
        self.pattern_encoder = nn.Linear(pattern_dim, self.hidden_dim, bias=False)
        self.context_projector = nn.Linear(self.hidden_dim, context_dim, bias=False)
        self.relevance_scorer = nn.Linear(context_dim, 1, bias=True)  # Only bias here
        
        # Verify parameter budget compliance
        total_params = sum(p.numel() for p in self.parameters())
        if total_params > max_params:
            raise ValueError(f"PatternRAG exceeds parameter budget: {total_params} > {max_params}")
        
        print(f"PatternRAG initialized with {total_params:,} parameters (limit: {max_params:,})")
    
    def _calculate_optimal_hidden_dim(self) -> int:
        """Calculate optimal hidden dimension to fit parameter budget."""
        # Equation: pattern_dim * h + h * context_dim + context_dim + 1 ≤ max_params
        # Solve for h: h * (pattern_dim + context_dim) ≤ max_params - context_dim - 1
        max_hidden = (self.max_params - self.context_dim - 1) // (self.pattern_dim + self.context_dim)
        return max(1, max_hidden)  # At least 1
    
    def forward(self, pattern_features: torch.Tensor, context_candidates: torch.Tensor) -> torch.Tensor:
        """Forward pass for pattern-context relevance scoring."""
        # pattern_features: [batch_size, pattern_dim]
        # context_candidates: [batch_size, num_candidates, context_dim] or [batch_size, context_dim]
        
        # Encode pattern features
        pattern_encoded = self.pattern_encoder(pattern_features)  # [batch_size, hidden_dim]
        
        # Project to context space
        pattern_projected = self.context_projector(pattern_encoded)  # [batch_size, context_dim]
        
        # Handle different context tensor shapes
        if context_candidates.dim() == 3:
            # Multiple candidates: [batch_size, num_candidates, context_dim]
            batch_size, num_candidates, context_dim = context_candidates.shape
            
            # Expand pattern projection for broadcasting
            pattern_expanded = pattern_projected.unsqueeze(1).expand(-1, num_candidates, -1)
            
            # Compute element-wise interaction
            interaction = pattern_expanded * context_candidates  # Element-wise product
            
            # Score each candidate
            relevance_scores = self.relevance_scorer(interaction)  # [batch_size, num_candidates, 1]
            return relevance_scores.squeeze(-1)  # [batch_size, num_candidates]
        
        else:
            # Single context: [batch_size, context_dim]
            interaction = pattern_projected * context_candidates
            relevance_score = self.relevance_scorer(interaction)  # [batch_size, 1]
            return relevance_score.squeeze(-1)  # [batch_size]


class CircuitBreaker:
    """Circuit breaker pattern for RAG timeout handling."""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0,
                 timeout_ms: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout_ms = timeout_ms / 1000.0  # Convert to seconds
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'OPEN':
                # Check if recovery timeout has passed
                if (self.last_failure_time and 
                    time.time() - self.last_failure_time > self.recovery_timeout):
                    self.state = 'HALF_OPEN'
                    self.failure_count = 0
                else:
                    # Circuit is open, fail fast
                    return None
        
        # Execute with timeout
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > self.timeout_ms:
                self._record_failure()
                return None
            else:
                self._record_success()
                return result
                
        except Exception as e:
            self._record_failure()
            return None
    
    def _record_success(self):
        """Record successful execution."""
        with self._lock:
            self.failure_count = 0
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
    
    def _record_failure(self):
        """Record failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time
        }


class RAGSystem:
    """Complete RAG system integrating encoding, retrieval, and ranking."""
    
    def __init__(self, max_contexts: int = 10000, retrieval_k: int = 10, 
                 embedding_dim: int = 256, incremental_learning: bool = False,
                 enable_neural_rag: bool = False, neural_rag_budget: int = 100_000):
        self.max_contexts = max_contexts
        self.retrieval_k = retrieval_k
        self.embedding_dim = embedding_dim
        self.incremental_learning = incremental_learning
        self.enable_neural_rag = enable_neural_rag
        
        # Core components
        self.encoder = SemanticEncoder(embedding_dim=embedding_dim, 
                                     regime_aware=True, temporal_aware=True)
        self.retriever = ContextRetriever(max_contexts=max_contexts)
        self.ranker = ContextRanker(adaptive_weights=incremental_learning)
        
        # Neural RAG component (optional, parameter-budgeted)
        if enable_neural_rag:
            self.neural_rag = PatternRAG(
                pattern_dim=128, 
                context_dim=embedding_dim,
                max_params=neural_rag_budget
            )
        else:
            self.neural_rag = None
        
        # Circuit breaker for timeout handling
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            timeout_ms=60.0
        )
        
        # Performance tracking
        self.retrieval_times = []
        self._lock = threading.RLock()
    
    def add_historical_context(self, context: RetrievalContext) -> None:
        """Add historical context to RAG database."""
        with self._lock:
            # Pre-compute embedding for fast retrieval
            if context._embedding is None:
                context._embedding = self.encoder.encode_market_context(context.market_state)
            
            # Add to retriever
            self.retriever.add_context(context)
    
    def retrieve_relevant_contexts(self, current_state: Dict[str, Any]) -> List[RetrievalContext]:
        """Retrieve relevant historical contexts with full RAG pipeline and circuit breaker."""
        # Use circuit breaker to handle timeouts and failures
        result = self.circuit_breaker.call(self._retrieve_with_timeout, current_state)
        
        if result is None:
            # Circuit breaker failed or is open, return empty results
            warnings.warn("RAG system failed or circuit breaker is open, returning empty results")
            return []
        
        return result
    
    def _retrieve_with_timeout(self, current_state: Dict[str, Any]) -> List[RetrievalContext]:
        """Internal retrieval method with timeout handling."""
        start_time = time.time()
        
        with self._lock:
            # Step 1: Retrieve similar contexts
            candidates = self.retriever.retrieve_similar(
                current_state, 
                top_k=min(self.retrieval_k * 3, 50),  # Retrieve more for ranking
                current_time=current_state.get('timestamp', datetime.now())
            )
            
            if not candidates:
                return []
            
            # Step 2: Neural RAG re-ranking (if enabled)
            if self.neural_rag is not None and 'patterns' in current_state:
                candidates = self._neural_rerank(current_state['patterns'], candidates)
            
            # Step 3: Traditional ranking using multiple criteria
            ranked_contexts = self.ranker.rank_contexts(current_state, candidates)
            
            # Step 4: Return top-k final results
            final_results = ranked_contexts[:self.retrieval_k]
            
            # Track performance
            elapsed_ms = (time.time() - start_time) * 1000
            self.retrieval_times.append(elapsed_ms)
            
            # Keep only recent performance stats
            if len(self.retrieval_times) > 100:
                self.retrieval_times = self.retrieval_times[-50:]
            
            # Performance warning (but don't fail - circuit breaker handles timeout)
            if elapsed_ms >= 60:
                warnings.warn(f"RAG retrieval took {elapsed_ms:.2f}ms, approaching timeout limit")
            
            # Compute quality scores
            for ctx in final_results:
                ctx.quality_score = self._compute_context_quality(ctx)
            
            return final_results
    
    def _neural_rerank(self, query_patterns: List, candidates: List[RetrievalContext]) -> List[RetrievalContext]:
        """Re-rank candidates using neural RAG component."""
        if not query_patterns or not candidates:
            return candidates
        
        try:
            # Convert patterns to tensor features
            pattern_features = self._patterns_to_tensor(query_patterns)
            
            # Convert candidates to context tensors
            context_features = self._contexts_to_tensor(candidates)
            
            # Neural scoring
            with torch.no_grad():
                relevance_scores = self.neural_rag(pattern_features, context_features)
                relevance_scores = relevance_scores.cpu().numpy()
            
            # Update candidate scores and re-sort
            for i, candidate in enumerate(candidates):
                if i < len(relevance_scores):
                    # Combine neural score with existing similarity score
                    neural_score = float(relevance_scores[i])
                    combined_score = 0.6 * neural_score + 0.4 * candidate.similarity_score
                    candidate.similarity_score = combined_score
                    candidate.relevance_score = combined_score
            
            # Re-sort by updated scores
            candidates.sort(key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            # Fall back to original ranking if neural component fails
            warnings.warn(f"Neural RAG reranking failed: {e}, falling back to traditional ranking")
        
        return candidates
    
    def _patterns_to_tensor(self, patterns: List) -> torch.Tensor:
        """Convert patterns to tensor features."""
        if not patterns:
            return torch.zeros(1, 128)  # Default pattern dim
        
        # Extract features from patterns
        features = []
        for pattern in patterns[:5]:  # Limit to top 5 patterns for efficiency
            if hasattr(pattern, 'to_feature_vector'):
                feature_vec = pattern.to_feature_vector()
                # Pad or truncate to 128 dimensions
                if len(feature_vec) >= 128:
                    features.append(feature_vec[:128])
                else:
                    padded = np.zeros(128)
                    padded[:len(feature_vec)] = feature_vec
                    features.append(padded)
            else:
                # Fallback for simple pattern representations
                features.append(np.random.normal(0, 0.1, 128))
        
        if not features:
            return torch.zeros(1, 128)
        
        # Average features if multiple patterns
        avg_features = np.mean(features, axis=0)
        return torch.tensor(avg_features, dtype=torch.float32).unsqueeze(0)
    
    def _contexts_to_tensor(self, contexts: List[RetrievalContext]) -> torch.Tensor:
        """Convert context candidates to tensor features."""
        if not contexts:
            return torch.zeros(1, 1, self.embedding_dim)
        
        context_tensors = []
        for ctx in contexts:
            if hasattr(ctx, '_embedding') and ctx._embedding is not None:
                context_tensors.append(ctx._embedding)
            else:
                # Generate embedding using encoder
                embedding = self.encoder.encode_market_context(ctx.market_state)
                context_tensors.append(embedding)
        
        # Stack into tensor [1, num_contexts, embedding_dim]
        context_array = np.stack(context_tensors, axis=0)
        return torch.tensor(context_array, dtype=torch.float32).unsqueeze(0)
    
    def augment_context(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Augment current context with retrieved historical information."""
        # Retrieve relevant contexts
        relevant_contexts = self.retrieve_relevant_contexts(current_context)
        
        if not relevant_contexts:
            return current_context.copy()
        
        # Extract information from retrieved contexts
        augmented = current_context.copy()
        
        # Historical patterns
        all_patterns = []
        for ctx in relevant_contexts:
            all_patterns.extend(ctx.patterns)
        
        augmented['historical_patterns'] = all_patterns
        
        # Expected outcomes based on similar contexts
        outcomes = [ctx.outcome for ctx in relevant_contexts if ctx.outcome]
        if outcomes:
            avg_return = np.mean([o.get('return_1h', 0) for o in outcomes])
            avg_accuracy = np.mean([o.get('accuracy', 0.5) for o in outcomes])
            avg_drawdown = np.mean([o.get('max_drawdown', 0) for o in outcomes])
            
            augmented['expected_outcomes'] = {
                'expected_return_1h': avg_return,
                'expected_accuracy': avg_accuracy,
                'expected_drawdown': avg_drawdown,
                'sample_size': len(outcomes)
            }
        
        # Confidence score based on retrieval quality
        if relevant_contexts:
            confidence = np.mean([ctx.relevance_score for ctx in relevant_contexts])
            augmented['confidence_score'] = confidence
        else:
            augmented['confidence_score'] = 0.0
        
        return augmented
    
    def _compute_context_quality(self, context: RetrievalContext) -> float:
        """Compute overall quality score for a context."""
        quality = 0.0
        
        # Relevance score (from ranking)
        quality += context.relevance_score * 0.4
        
        # Pattern quality
        if context.patterns:
            pattern_strengths = [p.strength for p in context.patterns if hasattr(p, 'strength')]
            if pattern_strengths:
                quality += np.mean(pattern_strengths) * 0.3
        
        # Outcome quality  
        if context.outcome:
            return_quality = np.tanh(context.outcome.get('return_1h', 0) * 100)
            accuracy = context.outcome.get('accuracy', 0.5)
            sample_size = min(1.0, np.log10(context.outcome.get('sample_size', 1)) / 2)
            
            outcome_quality = (return_quality + (accuracy - 0.5) * 2) * sample_size
            quality += np.tanh(outcome_quality) * 0.3
        
        # Normalize to [0, 1]
        return max(0.0, min(1.0, quality))
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the RAG system."""
        if not self.retrieval_times:
            return {'avg_retrieval_time_ms': 0.0, 'max_retrieval_time_ms': 0.0, 'contexts_stored': 0}
        
        return {
            'avg_retrieval_time_ms': np.mean(self.retrieval_times),
            'max_retrieval_time_ms': np.max(self.retrieval_times),
            'p95_retrieval_time_ms': np.percentile(self.retrieval_times, 95),
            'contexts_stored': len(self.retriever._contexts)
        }