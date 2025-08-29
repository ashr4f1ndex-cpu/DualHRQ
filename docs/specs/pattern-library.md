# Pattern Library Specification

**Team:** ML Infrastructure  
**Sprint:** Phase 1-2 (Weeks 5-12)  
**Storage Budget:** 500GB patterns + metadata  
**Latency Budget:** <60ms retrieval with fail-open  

## Scope

Implement scalable pattern detection, storage, and retrieval system for dynamic conditioning without puzzle_id dependencies.

## Architecture Overview

```
Pattern Detection → Pattern Encoding → Storage → Indexing → Retrieval
      ↓                    ↓              ↓         ↓         ↓
  Multi-scale          VQ-VAE/AE    Vector Store   FAISS    Context
  Detectors            Codebooks     + Metadata    Search   Fusion
```

## Core Components

### 1. Pattern Detection (`src/models/pattern_library.py`)

#### Multi-Scale Pattern Detector
```python
class MultiScalePatternDetector:
    def __init__(self, scales: List[int] = [5, 15, 30, 60]):
        self.scales = scales  # minutes
        self.detectors = {scale: ScaleDetector(scale) for scale in scales}
        
    def detect_patterns(self, market_data: MarketData) -> List[Pattern]:
        """
        Detect patterns across multiple time scales
        Returns: List of detected patterns with metadata
        """
        patterns = []
        for scale in self.scales:
            scale_patterns = self.detectors[scale].detect(market_data)
            patterns.extend(scale_patterns)
        return patterns
```

#### Pattern Types
```python
@dataclass
class Pattern:
    id: str                    # Unique pattern identifier
    pattern_type: PatternType  # Enum: momentum, reversal, breakout, etc.
    scale_minutes: int         # Time scale (5, 15, 30, 60 minutes)
    embedding: np.ndarray      # 128-dimensional pattern embedding
    features: Dict[str, float] # Raw feature values
    metadata: PatternMetadata  # Quality score, detection time, etc.
    
class PatternType(Enum):
    MOMENTUM_UP = "momentum_up"
    MOMENTUM_DOWN = "momentum_down"  
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    VOLUME_SPIKE = "volume_spike"
    SPREAD_WIDENING = "spread_widening"
    VOLATILITY_REGIME = "volatility_regime"
```

#### Technical Pattern Detectors
```python
class TechnicalPatternDetector:
    def detect_momentum(self, prices: pd.Series, volume: pd.Series) -> Optional[Pattern]:
        """Detect momentum patterns using multiple indicators"""
        # RSI, MACD, Volume-Price Trend
        rsi = compute_rsi(prices, window=14)
        macd = compute_macd(prices)
        vpt = compute_vpt(prices, volume)
        
        if self._is_momentum_pattern(rsi, macd, vpt):
            return Pattern(
                id=generate_pattern_id(),
                pattern_type=PatternType.MOMENTUM_UP if rsi > 70 else PatternType.MOMENTUM_DOWN,
                embedding=self._encode_momentum(rsi, macd, vpt),
                features={"rsi": rsi, "macd": macd, "vpt": vpt}
            )
        return None
        
    def detect_mean_reversion(self, prices: pd.Series) -> Optional[Pattern]:
        """Detect mean reversion using Bollinger Bands, Z-score"""
        bb_upper, bb_lower, bb_mean = compute_bollinger_bands(prices)
        z_score = (prices.iloc[-1] - bb_mean) / prices.std()
        
        if abs(z_score) > 2.0:  # Outside 2-sigma bands
            return Pattern(
                id=generate_pattern_id(),
                pattern_type=PatternType.MEAN_REVERSION,
                embedding=self._encode_mean_reversion(bb_upper, bb_lower, z_score),
                features={"z_score": z_score, "bb_position": (prices.iloc[-1] - bb_lower) / (bb_upper - bb_lower)}
            )
        return None
```

#### Microstructure Pattern Detectors  
```python
class MicrostructurePatternDetector:
    def detect_order_flow_patterns(self, order_book: OrderBookData) -> List[Pattern]:
        """Detect patterns in order flow and market microstructure"""
        patterns = []
        
        # Volume imbalance patterns
        imbalance = self._compute_volume_imbalance(order_book)
        if abs(imbalance) > 0.3:  # 30% imbalance threshold
            patterns.append(Pattern(
                pattern_type=PatternType.VOLUME_SPIKE,
                embedding=self._encode_volume_imbalance(imbalance, order_book),
                features={"volume_imbalance": imbalance}
            ))
            
        # Spread dynamics patterns
        spread_pattern = self._detect_spread_pattern(order_book)
        if spread_pattern:
            patterns.append(spread_pattern)
            
        return patterns
        
    def _compute_volume_imbalance(self, order_book: OrderBookData) -> float:
        """Compute bid-ask volume imbalance"""
        bid_volume = order_book.bid_sizes.sum()
        ask_volume = order_book.ask_sizes.sum()
        total_volume = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0
```

### 2. Pattern Encoding & Storage

#### Vector Quantization Codebook
```python
class PatternCodebook:
    def __init__(self, codebook_size: int = 1024, embedding_dim: int = 128):
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook = nn.Parameter(torch.randn(codebook_size, embedding_dim))
        
    def encode(self, patterns: List[Pattern]) -> torch.Tensor:
        """Encode patterns into discrete codes"""
        embeddings = torch.stack([torch.from_numpy(p.embedding) for p in patterns])
        # Vector quantization
        distances = torch.cdist(embeddings, self.codebook)
        codes = distances.argmin(dim=-1)
        return codes
        
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode discrete codes back to embeddings"""
        return self.codebook[codes]
```

#### Pattern Storage Engine
```python
class PatternStorageEngine:
    def __init__(self, storage_path: str = "data/patterns/"):
        self.storage_path = Path(storage_path)
        self.metadata_db = self._init_metadata_db()
        self.vector_store = self._init_vector_store()
        
    def store_pattern(self, pattern: Pattern) -> str:
        """Store pattern and return unique ID"""
        # Store embedding in vector database
        self.vector_store.add(pattern.id, pattern.embedding)
        
        # Store metadata in relational database
        self.metadata_db.execute("""
            INSERT INTO patterns (id, pattern_type, scale_minutes, features, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (pattern.id, pattern.pattern_type.value, pattern.scale_minutes,
              json.dumps(pattern.features), json.dumps(pattern.metadata), datetime.now()))
              
        return pattern.id
        
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve pattern by ID"""
        # Get metadata from database
        metadata_row = self.metadata_db.execute(
            "SELECT * FROM patterns WHERE id = ?", (pattern_id,)
        ).fetchone()
        
        if not metadata_row:
            return None
            
        # Get embedding from vector store
        embedding = self.vector_store.get(pattern_id)
        
        return Pattern(
            id=pattern_id,
            pattern_type=PatternType(metadata_row['pattern_type']),
            scale_minutes=metadata_row['scale_minutes'],
            embedding=embedding,
            features=json.loads(metadata_row['features']),
            metadata=json.loads(metadata_row['metadata'])
        )
```

### 3. Pattern Retrieval & Indexing

#### FAISS-based Similarity Search
```python
class PatternIndex:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product similarity
        self.id_map = {}  # FAISS index -> pattern ID mapping
        
    def add_patterns(self, patterns: List[Pattern]):
        """Add patterns to search index"""
        embeddings = np.array([p.embedding for p in patterns]).astype('float32')
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Update ID mapping
        for i, pattern in enumerate(patterns):
            self.id_map[start_idx + i] = pattern.id
            
    def search_similar(self, query_embedding: np.ndarray, 
                      k: int = 10) -> List[Tuple[str, float]]:
        """Search for k most similar patterns"""
        query = query_embedding.astype('float32').reshape(1, -1)
        similarities, indices = self.index.search(query, k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx in self.id_map:
                results.append((self.id_map[idx], float(sim)))
        return results
```

#### Context-Aware Retrieval
```python
class ContextAwareRetrieval:
    def __init__(self, pattern_index: PatternIndex, 
                 storage_engine: PatternStorageEngine):
        self.pattern_index = pattern_index
        self.storage_engine = storage_engine
        
    def retrieve_patterns(self, query_features: torch.Tensor,
                         context: Dict[str, Any],
                         max_patterns: int = 5,
                         timeout_ms: int = 60) -> Optional[torch.Tensor]:
        """
        Retrieve relevant patterns with timeout and fail-open
        """
        start_time = time.time()
        
        try:
            # Convert features to query embedding
            query_embedding = self._features_to_embedding(query_features)
            
            # Search for similar patterns
            similar_patterns = self.pattern_index.search_similar(
                query_embedding, k=max_patterns * 2  # Over-retrieve for filtering
            )
            
            # Context-based filtering
            filtered_patterns = self._filter_by_context(similar_patterns, context)
            
            # Check timeout
            if (time.time() - start_time) * 1000 > timeout_ms:
                logger.warning("Pattern retrieval timeout, failing open")
                return None
                
            # Encode retrieved patterns for conditioning
            pattern_context = self._encode_pattern_context(filtered_patterns[:max_patterns])
            return pattern_context
            
        except Exception as e:
            logger.error(f"Pattern retrieval failed: {e}, failing open")
            return None
            
    def _filter_by_context(self, patterns: List[Tuple[str, float]], 
                          context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Filter patterns based on market context"""
        filtered = []
        for pattern_id, similarity in patterns:
            pattern = self.storage_engine.get_pattern(pattern_id)
            if pattern and self._is_context_relevant(pattern, context):
                filtered.append((pattern_id, similarity))
        return filtered
        
    def _is_context_relevant(self, pattern: Pattern, context: Dict[str, Any]) -> bool:
        """Check if pattern is relevant to current context"""
        # Filter by time scale
        if 'preferred_scales' in context:
            if pattern.scale_minutes not in context['preferred_scales']:
                return False
                
        # Filter by market regime
        if 'market_regime' in context:
            # Pattern should be from similar market conditions
            pattern_regime = pattern.metadata.get('market_regime')
            if pattern_regime != context['market_regime']:
                return False
                
        # Filter by recency (patterns shouldn't be too old)
        if 'max_age_hours' in context:
            pattern_age = (datetime.now() - pattern.metadata['created_at']).total_seconds() / 3600
            if pattern_age > context['max_age_hours']:
                return False
                
        return True
```

## Pattern Lifecycle Management

### Pattern Quality Scoring
```python
class PatternQualityScorer:
    def compute_quality_score(self, pattern: Pattern, 
                            performance_history: List[float]) -> float:
        """Compute quality score based on historical performance"""
        if not performance_history:
            return 0.5  # Neutral score for new patterns
            
        # Weighted average of recent performance
        weights = np.exp(-np.arange(len(performance_history)) * 0.1)  # Exponential decay
        weighted_performance = np.average(performance_history, weights=weights)
        
        # Normalize to [0, 1] range
        quality_score = max(0.0, min(1.0, (weighted_performance + 1.0) / 2.0))
        
        return quality_score
```

### Pattern Expiration & Cleanup
```python  
class PatternLifecycleManager:
    def __init__(self, max_patterns: int = 10000, 
                 expiration_hours: int = 168):  # 1 week
        self.max_patterns = max_patterns
        self.expiration_hours = expiration_hours
        
    def cleanup_expired_patterns(self):
        """Remove expired and low-quality patterns"""
        cutoff_time = datetime.now() - timedelta(hours=self.expiration_hours)
        
        # Get expired patterns
        expired_patterns = self.metadata_db.execute("""
            SELECT id FROM patterns 
            WHERE created_at < ? OR quality_score < 0.2
            ORDER BY quality_score ASC, created_at ASC
        """, (cutoff_time,)).fetchall()
        
        # Remove from storage and index
        for pattern_id in expired_patterns:
            self._remove_pattern(pattern_id)
            
    def _remove_pattern(self, pattern_id: str):
        """Remove pattern from all storage systems"""
        # Remove from metadata database
        self.metadata_db.execute("DELETE FROM patterns WHERE id = ?", (pattern_id,))
        
        # Remove from vector store
        self.vector_store.remove(pattern_id)
        
        # Rebuild FAISS index if necessary
        if self._should_rebuild_index():
            self._rebuild_faiss_index()
```

## Configuration

```yaml
pattern_library:
  detection:
    scales: [5, 15, 30, 60]  # minutes
    pattern_types: ["momentum", "reversal", "breakout", "volume", "spread"]
    
  storage:
    max_patterns: 10000
    embedding_dim: 128
    codebook_size: 1024
    storage_path: "data/patterns/"
    
  retrieval:
    timeout_ms: 60
    max_retrieved: 5
    similarity_threshold: 0.7
    context_filtering: true
    
  lifecycle:
    expiration_hours: 168  # 1 week
    min_quality_score: 0.2
    cleanup_interval_hours: 24
```

## Performance Requirements

### Latency SLOs
- **Pattern detection**: <30ms per symbol
- **Pattern storage**: <10ms per pattern  
- **Pattern retrieval**: <60ms with fail-open to 0ms
- **Index search**: <20ms for 10K patterns

### Storage Requirements
- **Vector embeddings**: 128 * 4 bytes * 10K = ~5MB
- **Metadata database**: ~50MB for 10K patterns
- **FAISS index**: ~10MB for 10K patterns
- **Total**: ~500GB with growth buffer

## Testing Strategy

### Unit Tests (Write First)
```python
def test_pattern_detection_accuracy():
    # Test pattern detectors identify known patterns
    
def test_pattern_storage_retrieval():
    # Test round-trip storage and retrieval
    
def test_similarity_search():
    # Test FAISS index returns most similar patterns
    
def test_retrieval_timeout():
    # Test fail-open behavior on timeout
    
def test_pattern_quality_scoring():
    # Test quality score computation
```

### Integration Tests
```python
def test_end_to_end_pattern_pipeline():
    # Test full detection → storage → retrieval pipeline
    
def test_pattern_library_scaling():
    # Test performance with 10K+ patterns
    
def test_concurrent_access():
    # Test thread-safe access to pattern library
```

## Success Criteria

- [ ] Pattern detection covers major technical patterns
- [ ] Storage handles 10K+ patterns efficiently  
- [ ] Retrieval meets <60ms SLO with fail-open
- [ ] Pattern quality scoring improves over time
- [ ] No correlation with puzzle_id in pattern features
- [ ] Integration with conditioning system functional