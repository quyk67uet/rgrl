# transformer/policy_model.py
"""
Custom Transformer-based Feature Extractor for Stable Baselines 3.

History-aware policy extractor that uses a Transformer Encoder to process
sequences of (state, action) pairs.

Architecture:
- Input: Batch of padded history sequences (batch_size, max_history_len, embedding_dim)
- Positional Encoding: Sinusoidal embeddings to encode sequence order
- Transformer Encoder with [CLS] token strategy
- Output: Single feature vector per sequence (batch_size, features_dim)

v2.0 updates:
- Added positional encoding
- Reduced model size for faster training
- Fixed padding mask handling
"""

import torch
import torch.nn as nn
import math
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for the Transformer.
    
    Encodes token order with sine and cosine functions.
    
    Reference: "Attention is All You Need" (Vaswani et al., 2017)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Build the positional encoding matrix.
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension for broadcasting.
        pe = pe.unsqueeze(0)
        
        # Store as a buffer, not a trainable parameter.
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for sequence processing.
    
    Uses a [CLS] token with positional encoding:
    - Prepend a learnable [CLS] token
    - Add sinusoidal positional encodings
    - Pass through Transformer Encoder
    - Use the [CLS] output as the sequence representation
    - Project to features_dim via linear layer
    
    Args:
        observation_space: Gym observation space with shape (max_history_len, embedding_dim).
        features_dim: Output feature size.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        # Call the parent constructor first.
        super().__init__(observation_space, features_dim)
        
        # Read dimensions from the observation space.
        assert len(observation_space.shape) == 2, \
            f"Expected 2D observation space (seq_len, embed_dim), got {observation_space.shape}"
        
        self.max_history_len, self.embedding_dim = observation_space.shape
        
        print(f"\nInitializing TransformerFeaturesExtractor (v2.0):")
        print(f"   Input shape: ({self.max_history_len}, {self.embedding_dim})")
        print(f"   Features dim: {features_dim}")
        print(f"   Transformer: {n_layers} layers, {n_heads} heads")
        print(f"   Feedforward dim: {dim_feedforward}")
        print(f"   Dropout: {dropout}")
        
        # Learnable [CLS] token.
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        
        # Add positional encoding.
        self.pos_encoder = PositionalEncoding(
            d_model=self.embedding_dim,
            max_len=self.max_history_len + 1,
            dropout=dropout
        )
        print(f"   Positional encoding added (max_len={self.max_history_len + 1})")
        
        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(self.embedding_dim)
        )
        
        # Final projection layer.
        self.output_projection = nn.Linear(self.embedding_dim, features_dim)
        
        # Layer normalization for stability.
        self.layer_norm = nn.LayerNorm(features_dim)
        
        print(f"   TransformerFeaturesExtractor ready")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}\n")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        
        Args:
            observations: Tensor of shape (batch_size, max_history_len, embedding_dim).
        
        Returns:
            Tensor of shape (batch_size, features_dim).
        """
        batch_size = observations.shape[0]
        
        # Validate the input shape.
        assert observations.shape[1:] == (self.max_history_len, self.embedding_dim), \
            f"Expected shape (batch, {self.max_history_len}, {self.embedding_dim}), got {observations.shape}"
        
        # 1. Build the padding mask before adding positional encodings.
        obs_padding_mask = (observations.sum(dim=-1) == 0)  # True for padded positions
        
        # Mask for the [CLS] token, which is never masked.
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=observations.device)
        
        # Combine masks for the full sequence.
        padding_mask = torch.cat([cls_mask, obs_padding_mask], dim=1)
        
        # 2. Prepend the [CLS] token to each sequence.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate [CLS] with the observations.
        sequences_with_cls = torch.cat([cls_tokens, observations], dim=1)
        
        # 3. Add positional encodings.
        sequences_with_pos = self.pos_encoder(sequences_with_cls)
        
        # 4. Run the Transformer encoder.
        transformer_output = self.transformer_encoder(
            sequences_with_pos,
            src_key_padding_mask=padding_mask
        )
        
        # 5. Extract the [CLS] output.
        cls_output = transformer_output[:, 0, :]
        
        # 6. Project to the output feature size.
        features = self.output_projection(cls_output)
        
        # 7. Normalize for stability.
        features = self.layer_norm(features)
        
        return features


# Optional test.
if __name__ == "__main__":
    print("Testing TransformerFeaturesExtractor...")
    
    # Create a dummy observation space.
    max_history = 20
    embed_dim = 768
    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(max_history, embed_dim),
        dtype=np.float32
    )
    
    # Create the extractor.
    extractor = TransformerFeaturesExtractor(
        observation_space=obs_space,
        features_dim=256,
        n_heads=4,
        n_layers=2
    )
    
    # Create a dummy batch.
    batch_size = 4
    dummy_obs = torch.randn(batch_size, max_history, embed_dim)
    
    # Run a forward pass.
    features = extractor(dummy_obs)
    
    print(f"Input shape: {dummy_obs.shape}")
    print(f"Output shape: {features.shape}")
    print("Test passed!")
