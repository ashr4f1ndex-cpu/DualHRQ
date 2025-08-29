"""
rag_system.py - Retrieval-Augmented Generation System
====================================================

Production RAG system for DualHRQ 2.0 with fast historical context retrieval,
semantic encoding, and intelligent context augmentation for dynamic conditioning.

CRITICAL: Replace static puzzle_id conditioning with dynamic RAG retrieval.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import hashlib
import time
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class RAGSystem:
    """Production RAG system for dynamic market context retrieval."""
    
    def __init__(self, max_contexts: int = 10000, retrieval_k: int = 10, embedding_dim: int = 256,
                 incremental_learning: bool = False, temporal_decay: float = 0.1):
        self.max_contexts = max_contexts
        self.retrieval_k = retrieval_k
        self.embedding_dim = embedding_dim
        self.incremental_learning = incremental_learning
        self.temporal_decay = temporal_decay
        
        # Core components
        self.context_retriever = ContextRetriever(max_contexts, temporal_decay)
        self.semantic_encoder = SemanticEncoder(embedding_dim, regime_aware=True, temporal_aware=True)
        self.context_ranker = ContextRanker(adaptive_weights=True)
        
        # Performance tracking
        self._retrieval_times = deque(maxlen=1000)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._lock = threading.Lock()
    
    def add_historical_context(self, context: 'RetrievalContext') -> None:
        """Add historical context to RAG database with automatic indexing."""
        with self._lock:
            # Encode context for retrieval
            context.embedding = self.semantic_encoder.encode_market_context(context.market_state)
            
            # Add to retrieval database
            self.context_retriever.add_context(context)
            
            # Clear cache when adding new data
            if len(self._cache) > 100:
                self._cache.clear()
    
    def retrieve_relevant_contexts(self, current_state: Dict[str, Any], 
                                 top_k: Optional[int] = None) -> List['RetrievalContext']:
        """Retrieve relevant historical contexts with caching."""
        start_time = time.time()
        top_k = top_k or self.retrieval_k
        
        # Create cache key
        state_key = self._create_state_key(current_state)
        cache_key = f"{state_key}_{top_k}"
        
        # Check cache first
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # Retrieve similar contexts
        with self._lock:
            candidate_contexts = self.context_retriever.retrieve_similar(
                current_state, top_k * 2, datetime.now()  # Retrieve more for better ranking
            )
        
        # Rank by relevance
        ranked_contexts = self.context_ranker.rank_contexts(current_state, candidate_contexts)
        
        # Take top-k results
        result = ranked_contexts[:top_k]
        
        # Cache result
        self._cache[cache_key] = result
        
        # Track performance
        retrieval_time = (time.time() - start_time) * 1000
        self._retrieval_times.append(retrieval_time)
        
        if retrieval_time > 50:  # 50ms warning threshold
            warnings.warn(f"RAG retrieval took {retrieval_time:.2f}ms, consider optimization")
        
        return result
    
    def augment_context(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Augment current context with retrieved historical information."""
        # Retrieve relevant contexts
        relevant_contexts = self.retrieve_relevant_contexts(current_context)
        
        if not relevant_contexts:
            return current_context
        
        # Augment with historical patterns
        augmented = current_context.copy()
        
        # Extract historical patterns and outcomes
        historical_patterns = []
        historical_outcomes = []
        
        for ctx in relevant_contexts:
            if ctx.patterns:
                historical_patterns.extend(ctx.patterns)
            if ctx.outcome:
                historical_outcomes.append(ctx.outcome)
        
        # Add historical context to conditioning
        if historical_patterns:
            augmented['historical_patterns'] = historical_patterns[:5]  # Top 5 patterns
        
        if historical_outcomes:
            # Aggregate historical outcomes
            outcome_metrics = {}
            for outcome in historical_outcomes:
                for metric, value in outcome.items():
                    if metric not in outcome_metrics:
                        outcome_metrics[metric] = []
                    outcome_metrics[metric].append(value)
            
            # Calculate summary statistics
            historical_summary = {}
            for metric, values in outcome_metrics.items():
                historical_summary[f'hist_{metric}_mean'] = np.mean(values)
                historical_summary[f'hist_{metric}_std'] = np.std(values)
                historical_summary[f'hist_{metric}_recent'] = values[-1] if values else 0
            
            augmented['historical_outcomes'] = historical_summary
        
        # Add similarity scores for confidence weighting
        similarity_scores = [ctx.similarity_score for ctx in relevant_contexts]
        augmented['retrieval_confidence'] = np.mean(similarity_scores) if similarity_scores else 0.0
        augmented['retrieval_count'] = len(relevant_contexts)
        
        return augmented
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get RAG system performance statistics."""
        return {
            'avg_retrieval_time_ms': np.mean(self._retrieval_times) if self._retrieval_times else 0,
            'cache_hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            'total_contexts': self.context_retriever.get_context_count(),
            'cache_size': len(self._cache)
        }
    
    def _create_state_key(self, state: Dict[str, Any]) -> str:
        """Create deterministic key for state caching."""
        # Extract relevant features for hashing
        key_features = []
        for key in ['price', 'volume', 'volatility', 'momentum', 'regime', 'timestamp']:
            if key in state:
                key_features.append(f"{key}:{state[key]}")
        
        key_str = "|".join(sorted(key_features))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]


class ContextRetriever:
    """High-performance context retrieval with temporal decay and similarity matching."""
    
    def __init__(self, max_contexts: int = 10000, temporal_decay: float = 0.1):
        self.max_contexts = max_contexts
        self.temporal_decay = temporal_decay
        
        # Storage for contexts
        self._contexts: Dict[str, 'RetrievalContext'] = {}
        self._context_embeddings = []
        self._context_ids_ordered = []
        
        # Fast similarity search index
        self._nn_index = None
        self._scaler = StandardScaler()
        self._needs_rebuild = False
        
        # Insertion order for LRU eviction
        self._insertion_order = deque(maxlen=max_contexts)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self._last_rebuild_time = 0
    
    def add_context(self, context: 'RetrievalContext') -> None:
        """Add context to retrieval database with automatic indexing."""
        with self._lock:
            # Check capacity and evict if needed
            if len(self._contexts) >= self.max_contexts and context.context_id not in self._contexts:
                self._evict_oldest_context()
            
            # Store context
            self._contexts[context.context_id] = context
            self._insertion_order.append(context.context_id)
            
            # Mark for index rebuild
            self._needs_rebuild = True
    
    def retrieve_similar(self, query_context: Dict[str, Any], top_k: int = 10,
                        current_time: datetime = None) -> List['RetrievalContext']:
        """Retrieve similar contexts with temporal decay."""
        with self._lock:
            if not self._contexts:
                return []
            
            current_time = current_time or datetime.now()
            
            # Rebuild index if needed
            if self._needs_rebuild or self._nn_index is None:
                self._rebuild_similarity_index()
            
            # Create query embedding from context
            query_embedding = self._create_query_embedding(query_context)
            
            if query_embedding is None or len(self._context_embeddings) == 0:
                return []
            
            # Find similar contexts using k-NN
            query_normalized = self._scaler.transform(query_embedding.reshape(1, -1))
            k = min(top_k * 2, len(self._contexts))  # Get more candidates for temporal filtering
            
            distances, indices = self._nn_index.kneighbors(query_normalized, n_neighbors=k)
            
            # Apply temporal decay and filter
            candidate_contexts = []
            for idx, distance in zip(indices[0], distances[0]):
                context_id = self._context_ids_ordered[idx]
                context = self._contexts[context_id]
                
                # Calculate temporal decay
                time_diff = current_time - context.timestamp
                decay_factor = np.exp(-self.temporal_decay * time_diff.total_seconds() / 3600)  # Hourly decay
                
                # Combine similarity and temporal relevance
                similarity_score = 1.0 - distance  # Convert distance to similarity
                relevance_score = similarity_score * decay_factor
                
                context.similarity_score = similarity_score
                context.relevance_score = relevance_score
                
                candidate_contexts.append(context)
            
            # Sort by relevance and return top-k
            candidate_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
            return candidate_contexts[:top_k]
    
    def retrieve_by_patterns(self, query_patterns: List, top_k: int = 10) -> List['RetrievalContext']:
        """Retrieve contexts by pattern similarity."""
        with self._lock:
            if not self._contexts or not query_patterns:
                return []
            
            # Simple pattern matching based on pattern types and features
            matching_contexts = []
            
            query_pattern_types = {p.pattern_type for p in query_patterns if hasattr(p, 'pattern_type')}
            
            for context in self._contexts.values():
                if not context.patterns:
                    continue
                
                # Check pattern overlap
                context_pattern_types = {p.pattern_type for p in context.patterns if hasattr(p, 'pattern_type')}
                overlap = len(query_pattern_types.intersection(context_pattern_types))
                
                if overlap > 0:
                    # Score based on pattern overlap
                    pattern_score = overlap / max(len(query_pattern_types), len(context_pattern_types))
                    context.similarity_score = pattern_score
                    matching_contexts.append(context)
            
            # Sort by pattern similarity
            matching_contexts.sort(key=lambda x: x.similarity_score, reverse=True)
            return matching_contexts[:top_k]
    
    def get_context_count(self) -> int:
        """Get total number of stored contexts."""
        return len(self._contexts)
    
    def cleanup_old_contexts(self, max_age_hours: int = 24 * 30) -> int:
        """Remove contexts older than specified age."""
        with self._lock:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=max_age_hours)
            
            old_context_ids = [
                ctx_id for ctx_id, ctx in self._contexts.items()
                if ctx.timestamp < cutoff_time
            ]
            
            for ctx_id in old_context_ids:
                del self._contexts[ctx_id]
                if ctx_id in self._insertion_order:
                    # Remove from deque (inefficient but necessary)
                    temp_order = list(self._insertion_order)
                    temp_order.remove(ctx_id)
                    self._insertion_order = deque(temp_order, maxlen=self.max_contexts)
            
            if old_context_ids:
                self._needs_rebuild = True
            
            return len(old_context_ids)
    
    def _evict_oldest_context(self):
        """Evict oldest context (LRU policy)."""
        if self._insertion_order:
            oldest_id = self._insertion_order[0]
            if oldest_id in self._contexts:
                del self._contexts[oldest_id]
            self._needs_rebuild = True
    
    def _create_query_embedding(self, query_context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create embedding vector from query context."""
        try:
            # Extract numerical features
            features = []
            
            # Market state features
            for key in ['price', 'volume', 'volatility', 'momentum', 'spread', 'depth']:
                features.append(query_context.get(key, 0.0))
            
            # Technical indicators
            for key in ['rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_fast', 'ema_slow']:
                features.append(query_context.get(key, 0.0))
            
            # Regime features
            for key in ['regime_volatility', 'regime_trend', 'regime_momentum']:
                features.append(query_context.get(key, 0.0))
            
            # Time-based features
            if 'timestamp' in query_context:
                ts = query_context['timestamp']
                if isinstance(ts, datetime):
                    features.extend([
                        ts.hour / 24.0,
                        ts.weekday() / 7.0,
                        ts.day / 31.0,
                        ts.month / 12.0
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            warnings.warn(f"Error creating query embedding: {e}")
            return None
    
    def _rebuild_similarity_index(self):
        """Rebuild similarity search index."""
        if not self._contexts:
            self._context_embeddings = np.array([]).reshape(0, 17)  # 17 features
            self._context_ids_ordered = []
            return
        
        # Extract embeddings from all contexts
        embeddings = []
        context_ids = []
        
        for ctx_id, context in self._contexts.items():
            if hasattr(context, 'embedding') and context.embedding is not None:
                embeddings.append(context.embedding)
                context_ids.append(ctx_id)
            else:
                # Create embedding from market state
                embedding = self._create_query_embedding(context.market_state)
                if embedding is not None:
                    embeddings.append(embedding)
                    context_ids.append(ctx_id)
                    context.embedding = embedding
        
        if embeddings:
            self._context_embeddings = np.array(embeddings)
            self._context_ids_ordered = context_ids
            
            # Fit scaler and build index
            self._scaler.fit(self._context_embeddings)
            normalized_embeddings = self._scaler.transform(self._context_embeddings)
            
            self._nn_index = NearestNeighbors(
                n_neighbors=min(50, len(embeddings)),
                metric='euclidean',
                algorithm='ball_tree'
            )
            self._nn_index.fit(normalized_embeddings)
        
        self._needs_rebuild = False
        self._last_rebuild_time = time.time()


class SemanticEncoder:
    """Production semantic encoding of market contexts with regime and temporal awareness."""
    
    def __init__(self, embedding_dim: int = 256, regime_aware: bool = False, temporal_aware: bool = False):
        self.embedding_dim = embedding_dim
        self.regime_aware = regime_aware
        self.temporal_aware = temporal_aware
        
        # Feature extractors
        self._pca = PCA(n_components=min(embedding_dim // 4, 64))
        self._scaler = StandardScaler()
        
        # Regime state tracking
        if regime_aware:
            self._regime_states = defaultdict(lambda: {'mean': 0.0, 'std': 1.0, 'count': 0})
            self._current_regime = 'unknown'
        
        # Temporal pattern tracking
        if temporal_aware:
            self._temporal_patterns = defaultdict(list)
            self._time_windows = [300, 900, 3600]  # 5min, 15min, 1hour
        
        self._fitted = False
        self._feature_history = deque(maxlen=10000)  # For fitting
    
    def encode_market_context(self, market_context: Dict[str, Any]) -> np.ndarray:
        """Encode market context into semantic vector with regime and temporal features."""
        try:
            # Extract base features
            base_features = self._extract_base_features(market_context)
            
            # Add regime features if enabled
            if self.regime_aware:
                regime_features = self._extract_regime_features(market_context)
                base_features.extend(regime_features)
            
            # Add temporal features if enabled
            if self.temporal_aware:
                temporal_features = self._extract_temporal_features(market_context)
                base_features.extend(temporal_features)
            
            # Convert to numpy array
            feature_vector = np.array(base_features, dtype=np.float32)
            
            # Store for fitting if not already fitted
            if not self._fitted:
                self._feature_history.append(feature_vector)
                if len(self._feature_history) >= 100:  # Fit after 100 samples
                    self._fit_transformers()
            
            # Apply scaling and dimensionality reduction if fitted
            if self._fitted:
                try:
                    feature_vector = self._scaler.transform(feature_vector.reshape(1, -1))[0]
                    if len(feature_vector) > self.embedding_dim // 4:
                        reduced_features = self._pca.transform(feature_vector.reshape(1, -1))[0]
                    else:
                        reduced_features = feature_vector[:self.embedding_dim // 4]
                except:
                    reduced_features = feature_vector[:self.embedding_dim // 4]
            else:
                reduced_features = feature_vector[:self.embedding_dim // 4]
            
            # Expand to full embedding dimension
            embedding = np.zeros(self.embedding_dim)
            n_reduced = min(len(reduced_features), self.embedding_dim)
            embedding[:n_reduced] = reduced_features[:n_reduced]
            
            # Fill remaining dimensions with engineered features
            self._add_engineered_features(embedding, market_context, n_reduced)
            
            return embedding
            
        except Exception as e:
            warnings.warn(f"Error in semantic encoding: {e}")
            return np.zeros(self.embedding_dim)
    
    def _extract_base_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract base numerical features from market context."""
        features = []
        
        # Price and volume features
        price_features = ['price', 'bid', 'ask', 'volume', 'vwap', 'close', 'high', 'low', 'open']
        for feature in price_features:
            features.append(float(context.get(feature, 0.0)))
        
        # Technical indicators
        tech_features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                        'ema_fast', 'ema_slow', 'sma_short', 'sma_long', 'atr', 'adx']
        for feature in tech_features:
            features.append(float(context.get(feature, 0.0)))
        
        # Market microstructure
        micro_features = ['spread', 'depth', 'imbalance', 'tick_size', 'lot_size']
        for feature in micro_features:
            features.append(float(context.get(feature, 0.0)))
        
        # Volatility measures
        vol_features = ['realized_vol', 'implied_vol', 'vol_smile', 'vol_surface']
        for feature in vol_features:
            features.append(float(context.get(feature, 0.0)))
        
        return features
    
    def _extract_regime_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract regime-aware features."""
        features = []
        
        # Regime identification
        regime_indicators = ['volatility_regime', 'trend_regime', 'momentum_regime', 'correlation_regime']
        for indicator in regime_indicators:
            value = context.get(indicator, 'unknown')
            # Convert categorical to numerical
            if isinstance(value, str):
                regime_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0, 'unknown': 0.25}
                features.append(regime_map.get(value.lower(), 0.25))
            else:
                features.append(float(value))
        
        # Regime transition indicators
        regime_changes = ['regime_stability', 'regime_duration', 'regime_strength']
        for change in regime_changes:
            features.append(float(context.get(change, 0.0)))
        
        # Update regime tracking
        current_regime = context.get('current_regime', 'unknown')
        if current_regime != self._current_regime:
            self._current_regime = current_regime
        
        return features
    
    def _extract_temporal_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract temporal pattern features."""
        features = []
        
        # Time-based features
        timestamp = context.get('timestamp', datetime.now())
        if isinstance(timestamp, datetime):
            features.extend([
                timestamp.hour / 24.0,
                timestamp.minute / 60.0,
                timestamp.weekday() / 7.0,
                timestamp.day / 31.0,
                timestamp.month / 12.0
            ])
            
            # Seasonal features
            day_of_year = timestamp.timetuple().tm_yday
            features.extend([
                np.sin(2 * np.pi * day_of_year / 365.25),  # Seasonal sine
                np.cos(2 * np.pi * day_of_year / 365.25),  # Seasonal cosine
                np.sin(2 * np.pi * timestamp.hour / 24),    # Intraday sine
                np.cos(2 * np.pi * timestamp.hour / 24)     # Intraday cosine
            ])
        else:
            features.extend([0.0] * 9)
        
        # Time window aggregations
        for window in self._time_windows:
            window_key = f'window_{window}'
            window_data = context.get(window_key, {})
            
            # Statistical features for this window
            window_features = ['mean', 'std', 'min', 'max', 'skew', 'kurt']
            for feature in window_features:
                features.append(float(window_data.get(feature, 0.0)))
        
        return features
    
    def _fit_transformers(self):
        """Fit scaling and dimensionality reduction transformers."""
        if len(self._feature_history) < 10:
            return
        
        try:
            feature_matrix = np.array(list(self._feature_history))
            
            # Fit scaler
            self._scaler.fit(feature_matrix)
            
            # Fit PCA if we have enough dimensions
            if feature_matrix.shape[1] > self.embedding_dim // 4:
                scaled_features = self._scaler.transform(feature_matrix)
                self._pca.fit(scaled_features)
            
            self._fitted = True
            
        except Exception as e:
            warnings.warn(f"Error fitting transformers: {e}")
    
    def _add_engineered_features(self, embedding: np.ndarray, context: Dict[str, Any], start_idx: int):
        """Add engineered features to fill remaining embedding dimensions."""
        idx = start_idx
        
        if idx >= self.embedding_dim:
            return
        
        # Price momentum features
        price = context.get('price', 0.0)
        prev_price = context.get('prev_price', price)
        if prev_price != 0:
            price_change = (price - prev_price) / prev_price
            embedding[idx] = price_change
            idx += 1
        
        if idx >= self.embedding_dim:
            return
        
        # Volume momentum features
        volume = context.get('volume', 0.0)
        prev_volume = context.get('prev_volume', volume)
        if prev_volume != 0:
            volume_change = (volume - prev_volume) / prev_volume
            embedding[idx] = volume_change
            idx += 1
        
        if idx >= self.embedding_dim:
            return
        
        # Cross-asset correlations
        corr_features = context.get('correlations', {})
        for asset, corr in corr_features.items():
            if idx >= self.embedding_dim:
                break
            embedding[idx] = float(corr)
            idx += 1
        
        # Fill remaining with pattern indicators
        while idx < self.embedding_dim:
            # Use deterministic pseudo-random values based on context
            seed_value = hash(str(context.get('price', 0))) % 1000
            np.random.seed(seed_value)
            embedding[idx] = np.random.normal(0, 0.01)  # Small noise
            idx += 1


class ContextRanker:
    """Production context ranking with adaptive weights and multi-criteria scoring."""
    
    def __init__(self, criteria: List[str] = None, adaptive_weights: bool = False):
        self.criteria = criteria or ['semantic_similarity', 'temporal_proximity', 'outcome_quality', 'pattern_match']
        self.adaptive_weights = adaptive_weights
        
        # Initialize ranking weights
        self.weights = {criterion: 1.0 for criterion in self.criteria}
        self._normalize_weights()
        
        # Adaptive learning components
        if adaptive_weights:
            self._feedback_history = deque(maxlen=1000)
            self._weight_updates = defaultdict(list)
            self._learning_rate = 0.01
        
        # Performance tracking
        self._ranking_stats = defaultdict(list)
    
    def rank_contexts(self, query_context: Dict[str, Any], 
                     candidate_contexts: List['RetrievalContext']) -> List['RetrievalContext']:
        """Rank contexts by multi-criteria relevance scoring."""
        if not candidate_contexts:
            return []
        
        # Calculate scores for each criterion
        for context in candidate_contexts:
            scores = {}
            
            # Semantic similarity (already computed)
            scores['semantic_similarity'] = getattr(context, 'similarity_score', 0.0)
            
            # Temporal proximity
            scores['temporal_proximity'] = self._calculate_temporal_score(query_context, context)
            
            # Outcome quality
            scores['outcome_quality'] = self._calculate_outcome_score(context)
            
            # Pattern matching
            scores['pattern_match'] = self._calculate_pattern_score(query_context, context)
            
            # Compute weighted relevance score
            relevance_score = sum(
                self.weights[criterion] * scores.get(criterion, 0.0)
                for criterion in self.criteria
            )
            
            context.relevance_score = relevance_score
            context.criterion_scores = scores
        
        # Sort by relevance score
        ranked_contexts = sorted(candidate_contexts, key=lambda x: x.relevance_score, reverse=True)
        
        # Track ranking statistics
        self._update_ranking_stats(ranked_contexts)
        
        return ranked_contexts
    
    def update_weights_from_feedback(self, query: Dict[str, Any], 
                                   ranked_contexts: List['RetrievalContext'],
                                   feedback: Dict[str, float]) -> None:
        """Update ranking weights from user/system feedback using gradient descent."""
        if not self.adaptive_weights or not feedback:
            return
        
        # Store feedback for learning
        feedback_record = {
            'query': query,
            'contexts': ranked_contexts[:5],  # Top 5 contexts
            'feedback': feedback,
            'timestamp': datetime.now()
        }
        self._feedback_history.append(feedback_record)
        
        # Update weights based on feedback
        for context_idx, context in enumerate(ranked_contexts[:5]):
            if hasattr(context, 'criterion_scores'):
                # Get feedback score for this context
                context_feedback = feedback.get(f'context_{context_idx}', 0.0)
                
                # Update weights based on prediction error
                predicted_score = context.relevance_score
                error = context_feedback - predicted_score
                
                # Gradient update for each criterion
                for criterion in self.criteria:
                    criterion_score = context.criterion_scores.get(criterion, 0.0)
                    gradient = error * criterion_score
                    
                    # Apply learning rate and update weight
                    weight_update = self._learning_rate * gradient
                    self.weights[criterion] += weight_update
                    
                    # Track weight updates
                    self._weight_updates[criterion].append(weight_update)
        
        # Normalize and constrain weights
        self._normalize_weights()
        self._constrain_weights()
    
    def get_ranking_performance(self) -> Dict[str, float]:
        """Get ranking system performance metrics."""
        return {
            'avg_top1_relevance': np.mean([contexts[0].relevance_score for contexts in self._ranking_stats['ranked_lists'] if contexts]),
            'weight_stability': self._calculate_weight_stability(),
            'feedback_count': len(self._feedback_history),
            'criteria_weights': self.weights.copy()
        }
    
    def _calculate_temporal_score(self, query_context: Dict[str, Any], context: 'RetrievalContext') -> float:
        """Calculate temporal proximity score."""
        query_time = query_context.get('timestamp', datetime.now())
        if not isinstance(query_time, datetime):
            return 0.0
        
        time_diff = abs((query_time - context.timestamp).total_seconds())
        
        # Exponential decay with 1-hour half-life
        decay_rate = np.log(2) / 3600  # 1-hour half-life
        temporal_score = np.exp(-decay_rate * time_diff)
        
        return temporal_score
    
    def _calculate_outcome_score(self, context: 'RetrievalContext') -> float:
        """Calculate outcome quality score."""
        if not context.outcome:
            return 0.0
        
        # Aggregate outcome metrics
        outcome_scores = []
        
        # Common positive outcome indicators
        positive_indicators = ['profit', 'sharpe_ratio', 'success_rate', 'accuracy']
        for indicator in positive_indicators:
            if indicator in context.outcome:
                value = context.outcome[indicator]
                # Normalize to 0-1 range (assuming reasonable bounds)
                normalized_value = max(0, min(1, value)) if isinstance(value, (int, float)) else 0
                outcome_scores.append(normalized_value)
        
        # Common negative outcome indicators (invert)
        negative_indicators = ['drawdown', 'loss', 'error_rate', 'volatility']
        for indicator in negative_indicators:
            if indicator in context.outcome:
                value = context.outcome[indicator]
                # Invert and normalize
                if isinstance(value, (int, float)) and value != 0:
                    inverted_value = 1.0 / (1.0 + abs(value))
                    outcome_scores.append(inverted_value)
        
        return np.mean(outcome_scores) if outcome_scores else 0.5
    
    def _calculate_pattern_score(self, query_context: Dict[str, Any], context: 'RetrievalContext') -> float:
        """Calculate pattern matching score."""
        query_patterns = query_context.get('patterns', [])
        context_patterns = context.patterns or []
        
        if not query_patterns or not context_patterns:
            return 0.0
        
        # Pattern type matching
        query_pattern_types = {p.pattern_type for p in query_patterns if hasattr(p, 'pattern_type')}
        context_pattern_types = {p.pattern_type for p in context_patterns if hasattr(p, 'pattern_type')}
        
        if not query_pattern_types or not context_pattern_types:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_pattern_types.intersection(context_pattern_types))
        union = len(query_pattern_types.union(context_pattern_types))
        
        jaccard_score = intersection / union if union > 0 else 0.0
        
        # Weight by pattern strength if available
        context_strengths = [getattr(p, 'strength', 1.0) for p in context_patterns if hasattr(p, 'strength')]
        avg_strength = np.mean(context_strengths) if context_strengths else 0.5
        
        return jaccard_score * avg_strength
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for criterion in self.weights:
                self.weights[criterion] /= total_weight
    
    def _constrain_weights(self):
        """Constrain weights to reasonable bounds."""
        for criterion in self.weights:
            # Keep weights between 0.01 and 0.97
            self.weights[criterion] = max(0.01, min(0.97, self.weights[criterion]))
        
        # Renormalize after constraining
        self._normalize_weights()
    
    def _calculate_weight_stability(self) -> float:
        """Calculate stability of weight updates."""
        if not self._weight_updates:
            return 1.0
        
        stabilities = []
        for criterion, updates in self._weight_updates.items():
            if len(updates) > 1:
                # Calculate coefficient of variation
                if np.mean(updates) != 0:
                    cv = np.std(updates) / abs(np.mean(updates))
                    stability = 1.0 / (1.0 + cv)  # Convert to 0-1 scale
                    stabilities.append(stability)
        
        return np.mean(stabilities) if stabilities else 1.0
    
    def _update_ranking_stats(self, ranked_contexts: List['RetrievalContext']):
        """Update ranking performance statistics."""
        if ranked_contexts:
            self._ranking_stats['ranked_lists'].append(ranked_contexts)
            self._ranking_stats['top_scores'].append(ranked_contexts[0].relevance_score)
            
            # Keep only recent stats
            if len(self._ranking_stats['ranked_lists']) > 1000:
                self._ranking_stats['ranked_lists'] = self._ranking_stats['ranked_lists'][-1000:]
                self._ranking_stats['top_scores'] = self._ranking_stats['top_scores'][-1000:]


@dataclass
class RetrievalContext:
    """Production context for RAG retrieval with full metadata support."""
    context_id: str
    market_state: Dict[str, Any]
    patterns: List[Any]
    timestamp: datetime
    outcome: Dict[str, float]
    
    # Computed scores (set by retrieval/ranking systems)
    similarity_score: float = field(default=0.0)
    relevance_score: float = field(default=0.0)
    quality_score: float = field(default=0.0)
    
    # Optional computed features
    embedding: Optional[np.ndarray] = field(default=None)
    criterion_scores: Optional[Dict[str, float]] = field(default=None)
    
    # Metadata for tracking and analysis
    retrieval_count: int = field(default=0)
    last_retrieved: Optional[datetime] = field(default=None)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate required fields
        if not self.context_id:
            raise ValueError("context_id is required")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        
        # Initialize default values
        if self.criterion_scores is None:
            self.criterion_scores = {}
    
    def update_retrieval_stats(self):
        """Update retrieval statistics."""
        self.retrieval_count += 1
        self.last_retrieved = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'context_id': self.context_id,
            'market_state': self.market_state,
            'patterns': [p.to_dict() if hasattr(p, 'to_dict') else str(p) for p in self.patterns],
            'timestamp': self.timestamp.isoformat(),
            'outcome': self.outcome,
            'similarity_score': self.similarity_score,
            'relevance_score': self.relevance_score,
            'quality_score': self.quality_score,
            'retrieval_count': self.retrieval_count,
            'last_retrieved': self.last_retrieved.isoformat() if self.last_retrieved else None
        }


# Export all classes for Phase 1 implementation
__all__ = ['RAGSystem', 'ContextRetriever', 'SemanticEncoder', 'ContextRanker', 'RetrievalContext']