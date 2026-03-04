# transformer/policy_model.py
"""
Custom Transformer-based Feature Extractor for Stable Baselines 3.

This implements a history-aware policy that uses a Transformer Encoder
to process sequences of (state, action) pairs and make context-aware decisions.

Architecture:
- Input: Batch of padded history sequences (batch_size, max_history_len, embedding_dim)
- Positional Encoding: Sinusoidal embeddings to encode sequence order
- Transformer Encoder with [CLS] token strategy
- Output: Single feature vector per sequence (batch_size, features_dim)

CRITICAL FIX (v2.0):
- Added PositionalEncoding to fix permutation-invariance issue
- Reduced model complexity for faster training (2 layers, 4 heads, 512 FFN)
- Fixed padding mask logic to work correctly with positional encodings
"""

import torch
import torch.nn as nn
import math
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for Transformer.
    
    Injects information about the position of tokens in the sequence.
    Uses sine and cosine functions of different frequencies.
    
    Reference: "Attention is All You Need" (Vaswani et al., 2017)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        # Shape: (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Add batch dimension: (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of module state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # x shape: (batch_size, seq_len, d_model)
        # self.pe shape: (1, max_len, d_model)
        # Add positional encoding (broadcasts across batch dimension)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using Transformer Encoder for sequence processing.
    
    Uses [CLS] token strategy with positional encoding:
    - Prepend a learnable [CLS] token to each sequence
    - Add sinusoidal positional encodings to capture sequence order
    - Pass through Transformer Encoder
    - Extract the [CLS] token output as the sequence representation
    - Project to features_dim via linear layer
    
    Args:
        observation_space: Gym observation space (Box with shape (max_history_len, embedding_dim))
        features_dim: Dimension of output features (default: 256)
        n_heads: Number of attention heads (default: 4, reduced from 8)
        n_layers: Number of transformer layers (default: 2, reduced from 4)
        dim_feedforward: Dimension of feedforward network (default: 512, reduced from 1024)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        n_heads: int = 4,        # REDUCED: 8 → 4
        n_layers: int = 2,       # REDUCED: 4 → 2
        dim_feedforward: int = 512,  # REDUCED: 1024 → 512
        dropout: float = 0.1
    ):
        # Must call parent constructor first
        super().__init__(observation_space, features_dim)
        
        # Extract dimensions from observation space
        # observation_space.shape = (max_history_len, embedding_dim)
        assert len(observation_space.shape) == 2, \
            f"Expected 2D observation space (seq_len, embed_dim), got {observation_space.shape}"
        
        self.max_history_len, self.embedding_dim = observation_space.shape
        
        print(f"\n🔧 Initializing TransformerFeaturesExtractor (v2.0 - FIXED):")
        print(f"   Input shape: ({self.max_history_len}, {self.embedding_dim})")
        print(f"   Output features_dim: {features_dim}")
        print(f"   Transformer: {n_layers} layers, {n_heads} heads (OPTIMIZED)")
        print(f"   Feedforward dim: {dim_feedforward} (REDUCED)")
        print(f"   Dropout: {dropout}")
        
        # Learnable [CLS] token - this will be prepended to each sequence
        # Shape: (1, 1, embedding_dim) for broadcasting
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        
        # CRITICAL FIX: Add Positional Encoding
        # Max length = max_history_len + 1 (to account for [CLS] token)
        self.pos_encoder = PositionalEncoding(
            d_model=self.embedding_dim,
            max_len=self.max_history_len + 1,
            dropout=dropout
        )
        print(f"   ✓ Positional Encoding added (max_len={self.max_history_len + 1})")
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # CRITICAL: Input shape (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(self.embedding_dim)
        )
        
        # Final projection layer: [CLS] output -> features_dim
        self.output_projection = nn.Linear(self.embedding_dim, features_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(features_dim)
        
        print(f"   ✓ TransformerFeaturesExtractor initialized")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}\n")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer with positional encoding.
        
        Args:
            observations: Tensor of shape (batch_size, max_history_len, embedding_dim)
                         Padded history sequences from environment
        
        Returns:
            features: Tensor of shape (batch_size, features_dim)
                     Single feature vector per sequence for policy/value heads
        """
        batch_size = observations.shape[0]
        
        # Validate input shape
        assert observations.shape[1:] == (self.max_history_len, self.embedding_dim), \
            f"Expected shape (batch, {self.max_history_len}, {self.embedding_dim}), got {observations.shape}"
        
        # 1. Create padding mask BEFORE adding positional encodings
        # CRITICAL FIX: Mask must be created from original observations (where padding = all zeros)
        # Shape: (batch_size, max_history_len)
        obs_padding_mask = (observations.sum(dim=-1) == 0)  # True for padded positions
        
        # Create mask for [CLS] token (always False = never masked)
        # Shape: (batch_size, 1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=observations.device)
        
        # Concatenate to create final mask
        # Shape: (batch_size, max_history_len + 1)
        padding_mask = torch.cat([cls_mask, obs_padding_mask], dim=1)
        
        # 2. Prepend [CLS] token to each sequence in the batch
        # cls_token: (1, 1, embedding_dim)
        # Expand to: (batch_size, 1, embedding_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate [CLS] with observations
        # Result shape: (batch_size, max_history_len + 1, embedding_dim)
        sequences_with_cls = torch.cat([cls_tokens, observations], dim=1)
        
        # 3. Add positional encodings (CRITICAL FIX - THIS WAS MISSING!)
        # This encodes the sequential order of the history
        # Shape: (batch_size, max_history_len + 1, embedding_dim)
        sequences_with_pos = self.pos_encoder(sequences_with_cls)
        
        # 4. Pass through Transformer Encoder
        # Output shape: (batch_size, max_history_len + 1, embedding_dim)
        transformer_output = self.transformer_encoder(
            sequences_with_pos,
            src_key_padding_mask=padding_mask
        )
        
        # 5. Extract [CLS] token output (first position)
        # Shape: (batch_size, embedding_dim)
        cls_output = transformer_output[:, 0, :]
        
        # 6. Project to features_dim
        # Shape: (batch_size, features_dim)
        features = self.output_projection(cls_output)
        
        # 7. Layer normalization for stability
        features = self.layer_norm(features)
        
        return features


# Test function (optional, for debugging)
if __name__ == "__main__":
    print("Testing TransformerFeaturesExtractor...")
    
    # Create dummy observation space
    max_history = 20
    embed_dim = 768
    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(max_history, embed_dim),
        dtype=np.float32
    )
    
    # Create extractor (using new default optimized hyperparameters)
    extractor = TransformerFeaturesExtractor(
        observation_space=obs_space,
        features_dim=256,
        n_heads=4,  # Updated from 8
        n_layers=2   # Updated from 4
    )
    
    # Create dummy batch
    batch_size = 4
    dummy_obs = torch.randn(batch_size, max_history, embed_dim)
    
    # Forward pass
    features = extractor(dummy_obs)
    
    print(f"Input shape: {dummy_obs.shape}")
    print(f"Output shape: {features.shape}")
    print(f"✓ Test passed!")
