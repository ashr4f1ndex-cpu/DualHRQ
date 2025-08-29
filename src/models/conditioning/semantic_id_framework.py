"""
Semantic ID (SID) Framework - JSON Specification Implementation
============================================================

CRITICAL: Replace static puzzle_id with dynamic conditioning using vector quantization.

JSON Requirements:
- Scalable vector quantization-based approach for compact, generalizable conditioning
- Explicit methods for adapting SIDs via hashing and sub-sequence encoding  
- Industry-scale ranking model compatibility
- Direct replacement of static puzzle_id with dynamic conditioning alternatives

This is the PRIMARY solution for the HRM generalization crisis identified in JSON analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import hashlib
from collections import defaultdict
import warnings


@dataclass
class SemanticIDConfig:
    """Configuration for Semantic ID Framework."""
    codebook_size: int = 1024          # Number of semantic vectors in codebook
    embedding_dim: int = 256           # Dimension of semantic embeddings
    commitment_cost: float = 0.25      # VQ commitment loss weight
    decay: float = 0.99               # EMA update rate for codebook
    epsilon: float = 1e-5             # Numerical stability
    hash_length: int = 16             # Hash-based encoding length
    max_sequence_length: int = 512    # Maximum sub-sequence length
    num_hash_functions: int = 4       # Number of hash functions for LSH
    similarity_threshold: float = 0.8  # Similarity threshold for clustering
    enable_ema_updates: bool = True   # Enable Exponential Moving Average updates
    enable_hash_encoding: bool = True  # Enable hash-based encoding
    enable_subsequence_encoding: bool = True  # Enable sub-sequence encoding


class VectorQuantizer(nn.Module):
    """Vector Quantization component for Semantic ID."""
    
    def __init__(self, config: SemanticIDConfig):
        super().__init__()
        self.config = config
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.commitment_cost = config.commitment_cost
        self.decay = config.decay
        self.epsilon = config.epsilon
        
        # Codebook embeddings
        self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1/self.codebook_size, 1/self.codebook_size)
        
        # EMA tracking for codebook updates
        if config.enable_ema_updates:
            self.register_buffer('cluster_counts', torch.zeros(self.codebook_size))
            self.register_buffer('cluster_sum', torch.zeros(self.codebook_size, self.embedding_dim))
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vector quantization forward pass.
        
        Args:
            inputs: Input tensor [batch_size, ..., embedding_dim]
            
        Returns:
            quantized: Quantized output
            quantize_loss: VQ loss (commitment + codebook)
            encoding_indices: Discrete codebook indices
        """
        # Flatten input for quantization
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                   torch.sum(self.codebook.weight**2, dim=1) - \
                   2 * torch.matmul(flat_input, self.codebook.weight.t())
        
        # Get closest codebook entries
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and get quantized output
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
        
        # Calculate VQ losses
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        quantize_loss = self.commitment_cost * commitment_loss + codebook_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # EMA codebook updates during training
        if self.training and self.config.enable_ema_updates:
            self._update_codebook_ema(flat_input, encoding_indices.squeeze())
        
        return quantized, quantize_loss, encoding_indices.squeeze().view(input_shape[:-1])
    
    def _update_codebook_ema(self, flat_input: torch.Tensor, encoding_indices: torch.Tensor):
        """Update codebook using Exponential Moving Average."""
        encodings = F.one_hot(encoding_indices, self.codebook_size).float()
        
        # Update cluster counts and sums
        self.cluster_counts = self.cluster_counts * self.decay + \
                             torch.sum(encodings, dim=0) * (1 - self.decay)
        
        dw = torch.matmul(encodings.t(), flat_input)
        self.cluster_sum = self.cluster_sum * self.decay + dw * (1 - self.decay)
        
        # Update codebook weights
        n = torch.sum(self.cluster_counts)
        cluster_counts = (self.cluster_counts + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
        
        embed_normalized = self.cluster_sum / cluster_counts.unsqueeze(1)
        self.codebook.weight.data.copy_(embed_normalized)


class HashEncoder(nn.Module):
    """Hash-based encoding for dynamic conditioning adaptation."""
    
    def __init__(self, config: SemanticIDConfig):
        super().__init__()
        self.config = config
        self.hash_length = config.hash_length
        self.num_hash_functions = config.num_hash_functions
        self.embedding_dim = config.embedding_dim
        
        # Hash projection matrices
        self.hash_projections = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.hash_length, bias=False)
            for _ in range(self.num_hash_functions)
        ])
        
        # Initialize with random orthogonal matrices for better hash distribution
        for projection in self.hash_projections:
            nn.init.orthogonal_(projection.weight)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Generate hash-based encodings.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, embedding_dim]
            
        Returns:
            hash_codes: Binary hash codes [batch_size, seq_len, num_hash_functions, hash_length]
        """
        batch_size, seq_len = inputs.shape[:2]
        hash_codes = []
        
        for projection in self.hash_projections:
            # Project to hash space
            projected = projection(inputs)  # [B, S, hash_length]
            
            # Convert to binary codes using sign
            binary_codes = (projected > 0).float()  # Binary hash codes
            hash_codes.append(binary_codes)
        
        # Stack all hash functions
        hash_codes = torch.stack(hash_codes, dim=-2)  # [B, S, num_hash_functions, hash_length]
        
        return hash_codes
    
    def compute_hash_similarity(self, hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
        """Compute Hamming similarity between hash codes."""
        # XOR for Hamming distance, then average over hash functions and bits
        hamming_distance = torch.mean((hash1 != hash2).float(), dim=(-2, -1))
        hamming_similarity = 1.0 - hamming_distance
        return hamming_similarity


class SubsequenceEncoder(nn.Module):
    """Sub-sequence encoding for temporal pattern capture."""
    
    def __init__(self, config: SemanticIDConfig):
        super().__init__()
        self.config = config
        self.max_sequence_length = config.max_sequence_length
        self.embedding_dim = config.embedding_dim
        
        # Multi-scale temporal encoders
        self.temporal_encoders = nn.ModuleList([
            nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]  # Multiple temporal scales
        ])
        
        # Position encoding for temporal awareness
        self.position_encoding = nn.Parameter(
            torch.randn(self.max_sequence_length, self.embedding_dim) * 0.02
        )
        
        # Attention mechanism for sub-sequence importance
        self.attention = nn.MultiheadAttention(
            self.embedding_dim, num_heads=8, batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            len(self.temporal_encoders) * self.embedding_dim, 
            self.embedding_dim
        )
    
    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode sub-sequences with temporal awareness.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, embedding_dim]
            mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            encoded: Sub-sequence encoded tensor [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = inputs.shape
        
        # Add positional encoding (truncated if sequence is longer)
        pos_encoding = self.position_encoding[:seq_len, :]
        inputs_with_pos = inputs + pos_encoding.unsqueeze(0)
        
        # Multi-scale temporal encoding
        encoded_scales = []
        inputs_transposed = inputs_with_pos.transpose(1, 2)  # [B, E, S] for conv1d
        
        for conv in self.temporal_encoders:
            encoded = conv(inputs_transposed)  # [B, E, S]
            encoded = F.relu(encoded)
            encoded = encoded.transpose(1, 2)  # [B, S, E]
            encoded_scales.append(encoded)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(encoded_scales, dim=-1)  # [B, S, 4*E]
        
        # Project back to original dimension
        projected = self.output_projection(multi_scale)  # [B, S, E]
        
        # Apply attention for importance weighting
        if mask is not None:
            # Convert mask for attention (True = attend, False = ignore)
            attention_mask = mask.bool()
        else:
            attention_mask = None
        
        attended, _ = self.attention(projected, projected, projected, key_padding_mask=attention_mask)
        
        # Residual connection
        output = projected + attended
        
        return output


class SemanticIDFramework(nn.Module):
    """
    Complete Semantic ID Framework for dynamic conditioning.
    
    This replaces static puzzle_id with dynamic, learnable conditioning
    as specified in the JSON plan.
    """
    
    def __init__(self, config: SemanticIDConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.vector_quantizer = VectorQuantizer(config)
        
        if config.enable_hash_encoding:
            self.hash_encoder = HashEncoder(config)
        else:
            self.hash_encoder = None
            
        if config.enable_subsequence_encoding:
            self.subsequence_encoder = SubsequenceEncoder(config)
        else:
            self.subsequence_encoder = None
        
        # Input projection to ensure correct dimensions
        self.input_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Semantic clustering for regime detection
        self.regime_classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embedding_dim // 2, config.codebook_size),
            nn.Softmax(dim=-1)
        )
        
        # Statistics tracking
        self.register_buffer('usage_statistics', torch.zeros(config.codebook_size))
        self.register_buffer('semantic_similarities', torch.zeros(config.codebook_size, config.codebook_size))
    
    def forward(self, market_data: torch.Tensor, 
                context_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate Semantic IDs for dynamic conditioning.
        
        Args:
            market_data: Market data tensor [batch_size, seq_len, feature_dim]
            context_length: Optional context length for sub-sequence encoding
            
        Returns:
            Dictionary containing:
            - semantic_ids: Quantized semantic representations
            - discrete_codes: Discrete codebook indices  
            - hash_codes: Hash-based encodings (if enabled)
            - regime_probabilities: Market regime classifications
            - quantization_loss: VQ training loss
        """
        # Project input to correct dimensions
        projected = self.input_projection(market_data)
        
        # Sub-sequence encoding (if enabled)
        if self.subsequence_encoder is not None:
            if context_length is not None:
                # Truncate or pad to context length
                if projected.size(1) > context_length:
                    projected = projected[:, :context_length, :]
                elif projected.size(1) < context_length:
                    padding = torch.zeros(
                        projected.size(0), context_length - projected.size(1), 
                        projected.size(2), device=projected.device
                    )
                    projected = torch.cat([projected, padding], dim=1)
            
            encoded = self.subsequence_encoder(projected)
        else:
            encoded = projected
        
        # Vector quantization for semantic IDs
        semantic_ids, quantization_loss, discrete_codes = self.vector_quantizer(encoded)
        
        # Hash-based encoding (if enabled)
        hash_codes = None
        if self.hash_encoder is not None:
            hash_codes = self.hash_encoder(semantic_ids)
        
        # Regime classification
        # Use mean pooling over sequence for regime detection
        pooled_semantics = torch.mean(semantic_ids, dim=1)  # [B, E]
        regime_probabilities = self.regime_classifier(pooled_semantics)  # [B, codebook_size]
        
        # Update usage statistics during training
        if self.training:
            self._update_statistics(discrete_codes, regime_probabilities)
        
        return {
            'semantic_ids': semantic_ids,
            'discrete_codes': discrete_codes,
            'hash_codes': hash_codes,
            'regime_probabilities': regime_probabilities,
            'quantization_loss': quantization_loss,
            'pooled_semantics': pooled_semantics
        }
    
    def _update_statistics(self, discrete_codes: torch.Tensor, regime_probs: torch.Tensor):
        """Update usage and similarity statistics."""
        # Update codebook usage statistics
        unique_codes, counts = torch.unique(discrete_codes, return_counts=True)
        for code, count in zip(unique_codes, counts):
            self.usage_statistics[code] += count.float()
        
        # Update semantic similarities (batch-wise approximation)
        batch_similarities = torch.matmul(regime_probs, regime_probs.t())
        self.semantic_similarities += torch.mean(batch_similarities, dim=0)
    
    def get_similar_semantics(self, query_semantic: torch.Tensor, 
                            top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find most similar semantic IDs for retrieval.
        
        Args:
            query_semantic: Query semantic vector [embedding_dim]
            top_k: Number of similar semantics to return
            
        Returns:
            similar_ids: Top-k similar semantic ID indices
            similarities: Similarity scores
        """
        # Compute similarities with all codebook entries
        similarities = F.cosine_similarity(
            query_semantic.unsqueeze(0), 
            self.vector_quantizer.codebook.weight, 
            dim=1
        )
        
        # Get top-k most similar
        top_similarities, top_indices = torch.topk(similarities, top_k)
        
        return top_indices, top_similarities
    
    def adapt_to_new_regime(self, new_market_data: torch.Tensor, 
                           learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Adapt Semantic IDs to new market regime (online learning).
        
        Args:
            new_market_data: New market data for adaptation
            learning_rate: Adaptation learning rate
            
        Returns:
            Adaptation statistics and updated representations
        """
        with torch.no_grad():
            # Get current semantic representation
            current_output = self.forward(new_market_data)
            regime_probs = current_output['regime_probabilities']
            
            # Identify dominant regime
            dominant_regime = torch.argmax(regime_probs, dim=-1)
            
            # Adapt codebook entries (simplified online learning)
            semantic_ids = current_output['semantic_ids']
            discrete_codes = current_output['discrete_codes']
            
            # Update codebook with new patterns
            for batch_idx in range(discrete_codes.size(0)):
                for seq_idx in range(discrete_codes.size(1)):
                    code_idx = discrete_codes[batch_idx, seq_idx]
                    semantic_vector = semantic_ids[batch_idx, seq_idx]
                    
                    # EMA update for the corresponding codebook entry
                    current_vector = self.vector_quantizer.codebook.weight[code_idx]
                    updated_vector = (1 - learning_rate) * current_vector + \
                                   learning_rate * semantic_vector
                    self.vector_quantizer.codebook.weight[code_idx] = updated_vector
            
            return {
                'adapted_regimes': dominant_regime,
                'regime_confidence': torch.max(regime_probs, dim=-1)[0],
                'adaptation_magnitude': torch.norm(semantic_ids - current_output['pooled_semantics'].unsqueeze(1))
            }
    
    def export_semantic_dictionary(self) -> Dict[str, Any]:
        """Export learned semantic dictionary for analysis."""
        return {
            'codebook_vectors': self.vector_quantizer.codebook.weight.detach().cpu().numpy(),
            'usage_statistics': self.usage_statistics.detach().cpu().numpy(),
            'semantic_similarities': self.semantic_similarities.detach().cpu().numpy(),
            'config': self.config.__dict__
        }


def create_semantic_id_framework(market_data_dim: int = 128, 
                                codebook_size: int = 1024,
                                enable_all_features: bool = True) -> SemanticIDFramework:
    """
    Factory function to create Semantic ID Framework.
    
    Args:
        market_data_dim: Dimension of input market data features
        codebook_size: Size of semantic codebook
        enable_all_features: Whether to enable all advanced features
        
    Returns:
        Configured SemanticIDFramework
    """
    config = SemanticIDConfig(
        codebook_size=codebook_size,
        embedding_dim=market_data_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        hash_length=16,
        max_sequence_length=512,
        num_hash_functions=4,
        similarity_threshold=0.8,
        enable_ema_updates=enable_all_features,
        enable_hash_encoding=enable_all_features,
        enable_subsequence_encoding=enable_all_features
    )
    
    return SemanticIDFramework(config)


# Export for integration
__all__ = ['SemanticIDFramework', 'SemanticIDConfig', 'VectorQuantizer', 
           'HashEncoder', 'SubsequenceEncoder', 'create_semantic_id_framework']