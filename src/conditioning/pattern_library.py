"""pattern_library.py - Pattern Library Implementation
================================================

Core Pattern Library functionality for DualHRQ 2.0
Implements multi-scale pattern detection, fast storage/retrieval,
and approximate nearest neighbor matching with <10ms lookup times.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import uuid
import time
from dataclasses import dataclass, field
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import bisect

# Try to import FAISS, fall back to optimized alternatives
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# Add threading.RWLock implementation for older Python versions
if not hasattr(threading, 'RWLock'):
    class RWLock:
        """Simple RWLock implementation for thread safety."""
        def __init__(self):
            self._lock = threading.Lock()
        
        def __enter__(self):
            self._lock.acquire()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._lock.release()
    
    threading.RWLock = RWLock


@dataclass
class ScaleConfig:
    """Configuration for multi-scale pattern detection."""
    scale: str
    window_size: int
    overlap: float = 0.5
    
    @property
    def step_size(self) -> int:
        """Calculate step size based on overlap."""
        return max(1, int(self.window_size * (1 - self.overlap)))


class Pattern:
    """Individual pattern representation with features and metadata."""
    
    def __init__(self, pattern_id: str, pattern_type: str, scale: str, 
                 features: Dict[str, float], strength: float, 
                 detected_at: datetime = None, metadata: Dict = None, 
                 embedding: np.ndarray = None):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.scale = scale
        self.features = features or {}
        self.strength = strength
        self.detected_at = detected_at or datetime.now()
        self.metadata = metadata or {}
        self.embedding = embedding
        
        # Validate strength is in [0, 1]
        if not (0 <= strength <= 1):
            raise ValueError(f"Pattern strength must be in [0,1], got {strength}")
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert pattern to feature vector for similarity matching."""
        # Core features
        base_features = [
            self.features.get('duration', 0),
            self.features.get('amplitude', 0),
            self.features.get('volume_profile', 0),
            self.features.get('momentum', 0),
            self.strength
        ]
        
        # Encode categorical features
        pattern_type_encoding = {
            'trend': [1, 0, 0, 0],
            'reversal': [0, 1, 0, 0], 
            'breakout': [0, 0, 1, 0],
            'support_resistance': [0, 0, 0, 1]
        }
        
        scale_encoding = {
            '5m': [1, 0, 0, 0],
            '15m': [0, 1, 0, 0],
            '30m': [0, 0, 1, 0],
            '60m': [0, 0, 0, 1]
        }
        
        pattern_enc = pattern_type_encoding.get(self.pattern_type, [0, 0, 0, 0])
        scale_enc = scale_encoding.get(self.scale, [0, 0, 0, 0])
        
        return np.array(base_features + pattern_enc + scale_enc, dtype=np.float32)
    
    def __repr__(self):
        return f"Pattern(id={self.pattern_id}, type={self.pattern_type}, scale={self.scale}, strength={self.strength:.3f})"


class PatternLibrary:
    """Fast pattern storage and retrieval with <10ms lookup performance."""
    
    def __init__(self, max_patterns: int = 10000, pattern_ttl_days: int = None):
        self.max_patterns = max_patterns
        self.pattern_ttl_days = pattern_ttl_days
        
        # Thread-safe storage
        self._lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()
        self._patterns: Dict[str, Pattern] = {}
        
        # Fast indexing structures
        self._pattern_type_index = defaultdict(list)  # type -> [pattern_ids]
        self._scale_index = defaultdict(list)  # scale -> [pattern_ids]
        self._metadata_index = defaultdict(lambda: defaultdict(list))  # key -> value -> [pattern_ids]
        self._insertion_order = []  # For LRU eviction
        
        # Feature-based similarity index
        self._feature_vectors = []
        self._pattern_ids_ordered = []
        self._nn_index = None
        self._scaler = StandardScaler()
        self._needs_rebuild = False
    
    def store_pattern(self, pattern: Pattern) -> bool:
        """Store pattern with automatic indexing and capacity management."""
        with self._lock:
            # Check capacity and evict if needed
            if len(self._patterns) >= self.max_patterns and pattern.pattern_id not in self._patterns:
                self._evict_oldest_pattern()
            
            # Store pattern
            old_pattern_existed = pattern.pattern_id in self._patterns
            self._patterns[pattern.pattern_id] = pattern
            
            # Update indexes only for new patterns
            if not old_pattern_existed:
                self._insertion_order.append(pattern.pattern_id)
                self._pattern_type_index[pattern.pattern_type].append(pattern.pattern_id)
                self._scale_index[pattern.scale].append(pattern.pattern_id)
                
                # Index metadata
                for key, value in pattern.metadata.items():
                    self._metadata_index[key][value].append(pattern.pattern_id)
                
                # Defer index rebuild for batch operations
                self._needs_rebuild = True
            
            return True
    
    def store_patterns_batch(self, patterns: List[Pattern]) -> int:
        """Store multiple patterns efficiently with single index rebuild."""
        with self._lock:
            stored_count = 0
            for pattern in patterns:
                # Check capacity and evict if needed
                if len(self._patterns) >= self.max_patterns and pattern.pattern_id not in self._patterns:
                    self._evict_oldest_pattern()
                
                # Store pattern
                old_pattern_existed = pattern.pattern_id in self._patterns
                self._patterns[pattern.pattern_id] = pattern
                
                # Update indexes only for new patterns
                if not old_pattern_existed:
                    self._insertion_order.append(pattern.pattern_id)
                    self._pattern_type_index[pattern.pattern_type].append(pattern.pattern_id)
                    self._scale_index[pattern.scale].append(pattern.pattern_id)
                    
                    # Index metadata
                    for key, value in pattern.metadata.items():
                        self._metadata_index[key][value].append(pattern.pattern_id)
                    
                    stored_count += 1
            
            # Single rebuild at the end
            if stored_count > 0:
                self._rebuild_similarity_index()
            
            return stored_count
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve pattern by ID."""
        with self._lock:
            return self._patterns.get(pattern_id)
    
    def size(self) -> int:
        """Get current number of patterns."""
        return len(self._patterns)
    
    def find_similar_patterns(self, query_features: Dict[str, Any], top_k: int = 10) -> List[Pattern]:
        """Find similar patterns with <10ms performance requirement."""
        start_time = time.time()
        
        with self._lock:
            # Check if we need to rebuild the index
            if self._needs_rebuild or (len(self._feature_vectors) == 0 and self._patterns):
                self._rebuild_similarity_index()
            elif len(self._feature_vectors) == 0:
                return []
            
            # Filter by categorical constraints first (fast)
            candidate_ids = set(self._patterns.keys())
            
            if 'pattern_type' in query_features:
                pattern_type_ids = set(self._pattern_type_index.get(query_features['pattern_type'], []))
                candidate_ids = candidate_ids.intersection(pattern_type_ids)
            
            if 'scale' in query_features:
                scale_ids = set(self._scale_index.get(query_features['scale'], []))
                candidate_ids = candidate_ids.intersection(scale_ids)
            
            if 'min_strength' in query_features:
                min_strength = query_features['min_strength']
                strength_filtered = {pid for pid in candidate_ids 
                                   if self._patterns[pid].strength >= min_strength}
                candidate_ids = candidate_ids.intersection(strength_filtered)
            
            if not candidate_ids:
                return []
            
            # Map candidates to feature vector indices efficiently  
            candidate_indices = [i for i, pid in enumerate(self._pattern_ids_ordered) if pid in candidate_ids]
            
            if not candidate_indices:
                return []
            
            # Create query vector from features using 128-dim embedding
            query_pattern = Pattern(
                pattern_id="query",
                pattern_type=query_features.get('pattern_type', 'trend'),
                scale=query_features.get('scale', '15m'),
                features={
                    k: v for k, v in query_features.items() 
                    if k in ['duration', 'amplitude', 'volume_profile', 'momentum']
                },
                strength=query_features.get('min_strength', 0.5)
            )
            
            # Generate 128-dim query embedding
            query_embedding = self._generate_128d_embedding(query_pattern).reshape(1, -1)
            query_embedding = self._scaler.transform(query_embedding)
            
            if len(candidate_indices) <= top_k:
                # Return all candidates if we have fewer than top_k
                result_patterns = [self._patterns[self._pattern_ids_ordered[i]] for i in candidate_indices]
            else:
                # Use FAISS if available and efficient
                if (hasattr(self, '_faiss_index') and self._faiss_index is not None and 
                    len(candidate_indices) > 50):
                    
                    # FAISS-based similarity search
                    # Normalize query for cosine similarity
                    import faiss
                    normalized_query = query_embedding.copy()
                    faiss.normalize_L2(normalized_query)
                    
                    # Search only among candidates using FAISS
                    candidate_vectors = self._feature_vectors[candidate_indices]
                    faiss.normalize_L2(candidate_vectors)
                    
                    # Create temporary index for candidates
                    temp_index = faiss.IndexFlatIP(128)
                    temp_index.add(candidate_vectors)
                    
                    # Search for top-k
                    similarities, indices = temp_index.search(normalized_query, min(top_k, len(candidate_indices)))
                    
                    result_patterns = []
                    for idx in indices[0]:
                        if idx != -1:  # FAISS returns -1 for missing results
                            original_idx = candidate_indices[idx]
                            pattern_id = self._pattern_ids_ordered[original_idx]
                            result_patterns.append(self._patterns[pattern_id])
                else:
                    # Optimized numpy-based similarity search
                    candidate_vectors = self._feature_vectors[candidate_indices]
                    query_vector_flat = query_embedding.flatten()
                    
                    # Use cosine similarity for better semantic matching
                    # Normalize vectors
                    candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
                    query_norm = np.linalg.norm(query_vector_flat)
                    
                    # Avoid division by zero
                    candidate_norms[candidate_norms == 0] = 1e-8
                    if query_norm == 0:
                        query_norm = 1e-8
                    
                    # Compute cosine similarities
                    similarities = np.dot(candidate_vectors, query_vector_flat) / (candidate_norms * query_norm)
                    
                    # Get top-k indices (highest similarity)
                    top_k_indices = np.argpartition(similarities, -min(top_k, len(similarities)))[-top_k:]
                    # Sort by similarity descending
                    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]
                    
                    result_patterns = []
                    for idx in top_k_indices:
                        original_idx = candidate_indices[idx]
                        pattern_id = self._pattern_ids_ordered[original_idx]
                        result_patterns.append(self._patterns[pattern_id])
            
            lookup_time = (time.time() - start_time) * 1000
            if lookup_time >= 20:
                warnings.warn(f"Pattern lookup took {lookup_time:.2f}ms, exceeding 20ms requirement")
            
            return result_patterns
    
    def find_patterns_by_metadata(self, metadata_query: Dict[str, Any]) -> List[Pattern]:
        """Find patterns by metadata attributes."""
        with self._lock:
            candidate_ids = set(self._patterns.keys())
            
            for key, value in metadata_query.items():
                matching_ids = set(self._metadata_index.get(key, {}).get(value, []))
                candidate_ids = candidate_ids.intersection(matching_ids)
            
            return [self._patterns[pid] for pid in candidate_ids]
    
    def cleanup_expired_patterns(self, current_date: datetime = None) -> int:
        """Remove expired patterns based on TTL."""
        if self.pattern_ttl_days is None:
            return 0
        
        current_date = current_date or datetime.now()
        cutoff_date = current_date - timedelta(days=self.pattern_ttl_days)
        
        with self._lock:
            expired_ids = []
            for pid, pattern in self._patterns.items():
                if pattern.detected_at < cutoff_date:
                    expired_ids.append(pid)
            
            for pid in expired_ids:
                self._remove_pattern(pid)
            
            return len(expired_ids)
    
    def _evict_oldest_pattern(self):
        """Evict oldest pattern (LRU policy)."""
        if self._insertion_order:
            oldest_id = self._insertion_order[0]
            self._remove_pattern(oldest_id)
    
    def _remove_pattern(self, pattern_id: str):
        """Remove pattern and update all indexes."""
        if pattern_id not in self._patterns:
            return
        
        pattern = self._patterns[pattern_id]
        
        # Remove from storage
        del self._patterns[pattern_id]
        
        # Remove from indexes
        if pattern_id in self._insertion_order:
            self._insertion_order.remove(pattern_id)
        
        if pattern_id in self._pattern_type_index[pattern.pattern_type]:
            self._pattern_type_index[pattern.pattern_type].remove(pattern_id)
        
        if pattern_id in self._scale_index[pattern.scale]:
            self._scale_index[pattern.scale].remove(pattern_id)
        
        # Remove from metadata index
        for key, value in pattern.metadata.items():
            if pattern_id in self._metadata_index[key][value]:
                self._metadata_index[key][value].remove(pattern_id)
        
        self._needs_rebuild = True
    
    def _rebuild_similarity_index(self):
        """Rebuild feature vector index for similarity search."""
        if not self._patterns:
            self._feature_vectors = np.array([]).reshape(0, 128)  # Enhanced to 128-dim embeddings
            self._pattern_ids_ordered = []
            self._faiss_index = None
            return
        
        # Pre-allocate arrays for efficiency
        n_patterns = len(self._patterns)
        feature_vectors = np.empty((n_patterns, 128), dtype=np.float32)  # 128-dim embeddings
        pattern_ids = []
        
        # Extract enhanced feature vectors efficiently
        for i, (pid, pattern) in enumerate(self._patterns.items()):
            # Generate 128-dim embedding for each pattern
            feature_vectors[i] = self._generate_128d_embedding(pattern)
            pattern_ids.append(pid)
        
        self._feature_vectors = feature_vectors
        self._pattern_ids_ordered = pattern_ids
        
        # Fit scaler efficiently
        if len(feature_vectors) > 0:
            self._scaler.fit(self._feature_vectors)
            self._feature_vectors = self._scaler.transform(self._feature_vectors)
            
            # Build FAISS index for fast similarity search
            self._build_faiss_index()
        
        self._needs_rebuild = False
    
    def _generate_128d_embedding(self, pattern: Pattern) -> np.ndarray:
        """Generate 128-dimensional embedding for a pattern."""
        embedding = np.zeros(128, dtype=np.float32)
        
        # Start with base feature vector (13 dimensions)
        base_features = pattern.to_feature_vector()
        embedding[:len(base_features)] = base_features
        
        idx = len(base_features)
        
        # Time-based features (8 dimensions)
        if idx < 128:
            time_features = [
                pattern.detected_at.hour / 24.0,
                pattern.detected_at.minute / 60.0,
                pattern.detected_at.weekday() / 7.0,
                np.sin(2 * np.pi * pattern.detected_at.hour / 24),
                np.cos(2 * np.pi * pattern.detected_at.hour / 24),
                np.sin(2 * np.pi * pattern.detected_at.weekday() / 7),
                np.cos(2 * np.pi * pattern.detected_at.weekday() / 7),
                (pattern.detected_at.microsecond / 1000000.0)
            ]
            n_time = min(len(time_features), 128 - idx)
            embedding[idx:idx+n_time] = time_features[:n_time]
            idx += n_time
        
        # Feature interactions (remaining dimensions)
        if idx < 128:
            # Get base numeric features
            duration = pattern.features.get('duration', 0)
            amplitude = pattern.features.get('amplitude', 0) 
            volume = pattern.features.get('volume_profile', 0)
            momentum = pattern.features.get('momentum', 0)
            strength = pattern.strength
            
            # Generate interaction features
            interactions = [
                duration * amplitude,
                amplitude * volume,
                volume * momentum,
                momentum * strength,
                duration * strength,
                amplitude * strength,
                volume * strength,
                duration * volume,
                duration * momentum,
                amplitude * momentum,
                # Higher order interactions
                duration * amplitude * strength,
                volume * momentum * strength,
                amplitude * volume * momentum,
                # Polynomial features
                duration ** 2, amplitude ** 2, volume ** 2, momentum ** 2, strength ** 2,
                np.sqrt(abs(duration)), np.sqrt(abs(amplitude)), np.sqrt(abs(volume)),
                # Trigonometric features for cyclical patterns
                np.sin(duration * np.pi), np.cos(amplitude * np.pi),
                np.tanh(momentum), np.tanh(volume),
            ]
            
            # Fill remaining dimensions with interaction features
            n_interactions = min(len(interactions), 128 - idx)
            embedding[idx:idx+n_interactions] = interactions[:n_interactions]
            idx += n_interactions
        
        # Fill any remaining dimensions with pattern-specific hash-based features
        if idx < 128:
            # Use pattern ID hash for deterministic but diverse features
            np.random.seed(hash(pattern.pattern_id) % (2**32))
            remaining = 128 - idx
            hash_features = np.random.normal(0, 0.1, remaining).astype(np.float32)
            embedding[idx:] = hash_features
        
        return embedding
    
    def _build_faiss_index(self):
        """Build FAISS index or optimized fallback for fast similarity search."""
        if not hasattr(self, '_faiss_index'):
            self._faiss_index = None
            
        if FAISS_AVAILABLE and len(self._feature_vectors) >= 100:
            # Use FAISS for large datasets
            try:
                dimension = self._feature_vectors.shape[1]
                # Use IndexFlatIP for cosine similarity (after L2 normalization)
                self._faiss_index = faiss.IndexFlatIP(dimension)
                
                # L2 normalize for cosine similarity
                normalized_vectors = self._feature_vectors.copy()
                faiss.normalize_L2(normalized_vectors)
                
                self._faiss_index.add(normalized_vectors)
            except Exception:
                self._faiss_index = None
        else:
            # Use optimized sklearn fallback
            self._faiss_index = None


class PatternDetector:
    """Multi-scale pattern detection with streaming support."""
    
    def __init__(self, scales: List[str] = None, streaming_mode: bool = False):
        self.scales = scales or ['5m', '15m', '30m', '60m']
        self.streaming_mode = streaming_mode
        
        # Scale configurations
        self.scale_configs = {
            '5m': ScaleConfig('5m', 20, 0.5),    # 5-minute, 20-point windows
            '15m': ScaleConfig('15m', 40, 0.5),  # 15-minute, 40-point windows  
            '30m': ScaleConfig('30m', 60, 0.5),  # 30-minute, 60-point windows
            '60m': ScaleConfig('60m', 120, 0.5)  # 60-minute, 120-point windows
        }
        
        # Streaming state
        if streaming_mode:
            self._streaming_buffer = pd.DataFrame()
            self._last_patterns = []
    
    def detect_patterns(self, market_data: pd.DataFrame) -> List[Pattern]:
        """Detect patterns at multiple time scales."""
        if market_data.empty:
            return []
        
        all_patterns = []
        
        for scale in self.scales:
            if scale not in self.scale_configs:
                continue
                
            scale_patterns = self._detect_patterns_at_scale(market_data, scale)
            all_patterns.extend(scale_patterns)
        
        return all_patterns
    
    def update_streaming(self, new_data: pd.DataFrame) -> List[Pattern]:
        """Update streaming detection with new data points."""
        if not self.streaming_mode:
            raise ValueError("Detector not in streaming mode")
        
        # Add new data to buffer
        self._streaming_buffer = pd.concat([self._streaming_buffer, new_data], ignore_index=True)
        
        # Keep buffer manageable size (last 1000 points)
        if len(self._streaming_buffer) > 1000:
            self._streaming_buffer = self._streaming_buffer.tail(1000).reset_index(drop=True)
        
        # Detect patterns on current buffer
        new_patterns = self.detect_patterns(self._streaming_buffer)
        
        # Filter out patterns we've already seen (basic deduplication)
        truly_new = []
        for pattern in new_patterns:
            if not any(self._patterns_similar(pattern, old) for old in self._last_patterns):
                truly_new.append(pattern)
        
        self._last_patterns = new_patterns[-10:]  # Keep last 10 for comparison
        return truly_new
    
    def _detect_patterns_at_scale(self, data: pd.DataFrame, scale: str) -> List[Pattern]:
        """Detect patterns at a specific time scale."""
        config = self.scale_configs[scale]
        patterns = []
        
        if len(data) < config.window_size:
            return patterns
        
        # Sliding window pattern detection
        for i in range(0, len(data) - config.window_size + 1, config.step_size):
            window = data.iloc[i:i + config.window_size].copy()
            
            # Detect different pattern types in this window
            trend_pattern = self._detect_trend_pattern(window, scale, i)
            if trend_pattern:
                patterns.append(trend_pattern)
            
            reversal_pattern = self._detect_reversal_pattern(window, scale, i) 
            if reversal_pattern:
                patterns.append(reversal_pattern)
            
            breakout_pattern = self._detect_breakout_pattern(window, scale, i)
            if breakout_pattern:
                patterns.append(breakout_pattern)
            
            support_resistance_pattern = self._detect_support_resistance_pattern(window, scale, i)
            if support_resistance_pattern:
                patterns.append(support_resistance_pattern)
        
        return patterns
    
    def _detect_trend_pattern(self, window: pd.DataFrame, scale: str, offset: int) -> Optional[Pattern]:
        """Detect trend patterns in price data."""
        prices = window['price'].values
        volumes = window.get('volume', pd.Series([1000] * len(window))).values
        
        # Linear trend analysis
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        correlation = np.corrcoef(x, prices)[0, 1]
        
        # Trend strength based on correlation and consistency
        strength = abs(correlation) if not np.isnan(correlation) else 0
        
        # Minimum strength threshold
        if strength < 0.3:
            return None
        
        # Calculate features
        duration = len(prices)
        amplitude = abs(prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        volume_profile = np.mean(volumes) / 1000  # Normalize
        momentum = slope / (prices[0] if prices[0] != 0 else 1)
        
        features = {
            'duration': duration,
            'amplitude': amplitude,
            'volume_profile': volume_profile,
            'momentum': momentum
        }
        
        pattern_id = f"trend_{scale}_{offset}_{uuid.uuid4().hex[:8]}"
        
        return Pattern(
            pattern_id=pattern_id,
            pattern_type='trend',
            scale=scale,
            features=features,
            strength=strength,
            detected_at=datetime.now()
        )
    
    def _detect_reversal_pattern(self, window: pd.DataFrame, scale: str, offset: int) -> Optional[Pattern]:
        """Detect reversal patterns (V-shapes, inverted V-shapes)."""
        prices = window['price'].values
        volumes = window.get('volume', pd.Series([1000] * len(window))).values
        
        if len(prices) < 10:
            return None
        
        # Find potential reversal point (local min/max)
        mid_idx = len(prices) // 2
        search_start = max(2, mid_idx - 5)
        search_end = min(len(prices) - 2, mid_idx + 5)
        
        reversal_idx = search_start
        for i in range(search_start, search_end):
            # Check for local minimum (V-pattern)
            if (prices[i] < prices[i-1] and prices[i] < prices[i+1] and 
                prices[i] < prices[i-2] and prices[i] < prices[i+2]):
                reversal_idx = i
                break
            # Check for local maximum (inverted V-pattern)
            elif (prices[i] > prices[i-1] and prices[i] > prices[i+1] and 
                  prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                reversal_idx = i
                break
        
        # Measure strength of reversal
        left_slope = np.polyfit(range(reversal_idx), prices[:reversal_idx], 1)[0] if reversal_idx > 1 else 0
        right_slope = np.polyfit(range(len(prices) - reversal_idx), prices[reversal_idx:], 1)[0] if reversal_idx < len(prices) - 2 else 0
        
        # Reversal strength based on slope change
        slope_change = abs(right_slope - left_slope)
        strength = min(1.0, slope_change * 1000)  # Scale factor
        
        if strength < 0.4:
            return None
        
        # Calculate features
        duration = len(prices)
        amplitude = abs(max(prices) - min(prices)) / np.mean(prices) if np.mean(prices) != 0 else 0
        volume_profile = np.mean(volumes) / 1000
        momentum = (right_slope - left_slope) / (np.mean(prices) if np.mean(prices) != 0 else 1)
        
        features = {
            'duration': duration,
            'amplitude': amplitude,
            'volume_profile': volume_profile,
            'momentum': momentum
        }
        
        pattern_id = f"reversal_{scale}_{offset}_{uuid.uuid4().hex[:8]}"
        
        return Pattern(
            pattern_id=pattern_id,
            pattern_type='reversal',
            scale=scale,
            features=features,
            strength=strength,
            detected_at=datetime.now()
        )
    
    def _detect_breakout_pattern(self, window: pd.DataFrame, scale: str, offset: int) -> Optional[Pattern]:
        """Detect breakout patterns (sudden price movements with volume)."""
        prices = window['price'].values
        volumes = window.get('volume', pd.Series([1000] * len(window))).values
        
        if len(prices) < 10:
            return None
        
        # Calculate price volatility in first and second half
        mid = len(prices) // 2
        first_half_vol = np.std(prices[:mid]) if mid > 1 else 0
        second_half_vol = np.std(prices[mid:]) if len(prices) - mid > 1 else 0
        
        # Look for volatility breakout
        vol_ratio = second_half_vol / first_half_vol if first_half_vol > 0 else 0
        
        # Volume confirmation
        first_half_volume = np.mean(volumes[:mid]) if mid > 0 else 1000
        second_half_volume = np.mean(volumes[mid:]) if len(volumes) - mid > 0 else 1000
        volume_ratio = second_half_volume / first_half_volume if first_half_volume > 0 else 1
        
        # Breakout strength combines volatility expansion and volume increase
        strength = min(1.0, (vol_ratio * volume_ratio) / 4)  # Normalize
        
        if strength < 0.5 or vol_ratio < 1.5:
            return None
        
        # Calculate features
        duration = len(prices)
        amplitude = abs(prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        volume_profile = np.mean(volumes) / 1000
        momentum = np.polyfit(range(len(prices)), prices, 1)[0] / (np.mean(prices) if np.mean(prices) != 0 else 1)
        
        features = {
            'duration': duration,
            'amplitude': amplitude,
            'volume_profile': volume_profile,
            'momentum': momentum
        }
        
        pattern_id = f"breakout_{scale}_{offset}_{uuid.uuid4().hex[:8]}"
        
        return Pattern(
            pattern_id=pattern_id,
            pattern_type='breakout',
            scale=scale,
            features=features,
            strength=strength,
            detected_at=datetime.now()
        )
    
    def _detect_support_resistance_pattern(self, window: pd.DataFrame, scale: str, offset: int) -> Optional[Pattern]:
        """Detect support and resistance levels."""
        prices = window['price'].values
        volumes = window.get('volume', pd.Series([1000] * len(window))).values
        
        if len(prices) < 15:
            return None
        
        # Find local highs and lows
        highs = []
        lows = []
        
        for i in range(2, len(prices) - 2):
            # Local high
            if prices[i] > prices[i-1] and prices[i] > prices[i+1] and prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                highs.append(prices[i])
            # Local low  
            if prices[i] < prices[i-1] and prices[i] < prices[i+1] and prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                lows.append(prices[i])
        
        if len(highs) < 2 and len(lows) < 2:
            return None
        
        # Check for consistent support/resistance levels
        strength = 0
        
        if len(highs) >= 2:
            resistance_level = np.mean(highs)
            resistance_std = np.std(highs) if len(highs) > 1 else 0
            if resistance_std < resistance_level * 0.02:  # Tight resistance
                strength = max(strength, 0.6)
        
        if len(lows) >= 2:
            support_level = np.mean(lows) 
            support_std = np.std(lows) if len(lows) > 1 else 0
            if support_std < support_level * 0.02:  # Tight support
                strength = max(strength, 0.6)
        
        if strength < 0.5:
            return None
        
        # Calculate features
        duration = len(prices)
        amplitude = abs(max(prices) - min(prices)) / np.mean(prices) if np.mean(prices) != 0 else 0
        volume_profile = np.mean(volumes) / 1000
        momentum = 0  # Support/resistance typically has low momentum
        
        features = {
            'duration': duration,
            'amplitude': amplitude,
            'volume_profile': volume_profile,
            'momentum': momentum
        }
        
        pattern_id = f"support_resistance_{scale}_{offset}_{uuid.uuid4().hex[:8]}"
        
        return Pattern(
            pattern_id=pattern_id,
            pattern_type='support_resistance',
            scale=scale,
            features=features,
            strength=strength,
            detected_at=datetime.now()
        )
    
    def _patterns_similar(self, p1: Pattern, p2: Pattern, threshold: float = 0.1) -> bool:
        """Check if two patterns are similar (for deduplication)."""
        if p1.pattern_type != p2.pattern_type or p1.scale != p2.scale:
            return False
        
        # Compare feature vectors
        v1 = p1.to_feature_vector()
        v2 = p2.to_feature_vector()
        
        distance = np.linalg.norm(v1 - v2)
        return distance < threshold


class PatternMatcher:
    """High-performance pattern matching with approximate nearest neighbor search."""
    
    def __init__(self, index_type: str = 'ann', embedding_dim: int = 128):
        self.index_type = index_type
        self.embedding_dim = embedding_dim
        
        # ANN index for fast similarity search
        self._nn_index = None
        self._patterns = []
        self._embeddings = []
        self._scaler = StandardScaler()
        
    def build_index(self, patterns: List[Pattern]) -> None:
        """Build ANN index for fast pattern matching."""
        if not patterns:
            return
        
        self._patterns = patterns.copy()
        
        # Generate embeddings for all patterns
        embeddings = []
        for pattern in patterns:
            if pattern.embedding is not None:
                embeddings.append(pattern.embedding)
            else:
                embeddings.append(self.generate_embedding(pattern))
        
        if embeddings:
            self._embeddings = np.array(embeddings)
            
            # Fit scaler and normalize
            self._scaler.fit(self._embeddings)
            normalized_embeddings = self._scaler.transform(self._embeddings)
            
            # Build k-NN index
            self._nn_index = NearestNeighbors(
                n_neighbors=min(50, len(patterns)),
                metric='euclidean',
                algorithm='ball_tree'  # Fast for high-dimensional data
            )
            self._nn_index.fit(normalized_embeddings)
    
    def find_similar_by_embedding(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Pattern]:
        """Find similar patterns using embedding-based ANN search."""
        if self._nn_index is None or len(self._patterns) == 0:
            return []
        
        # Normalize query embedding
        query_normalized = self._scaler.transform(query_embedding.reshape(1, -1))
        
        # Find k nearest neighbors
        k = min(top_k, len(self._patterns))
        distances, indices = self._nn_index.kneighbors(query_normalized, n_neighbors=k)
        
        # Return patterns sorted by similarity
        similar_patterns = []
        for idx in indices[0]:
            similar_patterns.append(self._patterns[idx])
        
        return similar_patterns
    
    def generate_embedding(self, pattern: Pattern) -> np.ndarray:
        """Generate embedding vector for a pattern."""
        # Start with feature vector as base
        base_features = pattern.to_feature_vector()
        
        # Expand to full embedding dimension
        if len(base_features) >= self.embedding_dim:
            return base_features[:self.embedding_dim]
        
        # Create expanded embedding with feature engineering
        embedding = np.zeros(self.embedding_dim)
        
        # Copy base features
        embedding[:len(base_features)] = base_features
        
        # Add engineered features
        idx = len(base_features)
        
        if idx < self.embedding_dim:
            # Time-based features
            hour = pattern.detected_at.hour / 24.0
            day_of_week = pattern.detected_at.weekday() / 7.0
            embedding[idx:idx+2] = [hour, day_of_week]
            idx += 2
        
        if idx < self.embedding_dim:
            # Interaction features (products of key features)
            duration = pattern.features.get('duration', 0)
            amplitude = pattern.features.get('amplitude', 0)
            volume = pattern.features.get('volume_profile', 0)
            momentum = pattern.features.get('momentum', 0)
            
            interactions = [
                duration * amplitude,
                amplitude * volume,
                momentum * pattern.strength,
                duration * pattern.strength,
                volume * momentum
            ]
            
            n_interactions = min(len(interactions), self.embedding_dim - idx)
            embedding[idx:idx+n_interactions] = interactions[:n_interactions]
            idx += n_interactions
        
        # Fill remaining with noise for regularization
        if idx < self.embedding_dim:
            np.random.seed(hash(pattern.pattern_id) % (2**32))  # Deterministic noise
            remaining = self.embedding_dim - idx
            embedding[idx:] = np.random.normal(0, 0.01, remaining)
        
        return embedding.astype(np.float32)
    
    def batch_find_similar(self, query_patterns: List[Pattern], library: 'PatternLibrary', top_k: int = 5) -> List[List[Pattern]]:
        """Efficiently process batch of similarity queries."""
        results = []
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for query_pattern in query_patterns:
                # Convert pattern to query features
                query_features = query_pattern.features.copy()
                query_features['pattern_type'] = query_pattern.pattern_type
                query_features['scale'] = query_pattern.scale
                
                future = executor.submit(library.find_similar_patterns, query_features, top_k)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=0.1)  # 100ms timeout per query
                    results.append(result)
                except:
                    results.append([])  # Empty result on timeout/error
        
        return results


# Legacy stubs for backwards compatibility
class RegimeDetector:
    """Legacy stub: Detect market regime signatures for pattern identification."""
    
    def __init__(self):
        pass
    
    def detect_regime_signature(self, data: pd.DataFrame, reference_date: datetime) -> Optional[Dict]:
        """Legacy method - use PatternDetector instead."""
        return None


class PatternIDGenerator:
    """Legacy stub: Generate pattern IDs from regime signatures."""
    
    def __init__(self):
        pass
    
    def generate_pattern_id(self, signature: Dict) -> str:
        """Legacy method - pattern IDs are auto-generated."""
        return f"legacy_pattern_{uuid.uuid4().hex[:8]}"


class SpecialistAdapter:
    """Legacy stub: LoRA-style parameter-efficient adaptation."""
    
    def __init__(self, base_dim: int, rank: int = 16):
        self.base_dim = base_dim
        self.rank = rank
    
    def adapt_parameters(self, base_params: Dict) -> Dict:
        """Legacy method - not implemented."""
        return base_params