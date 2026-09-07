"""Transformer decoder with cross-attention over encoder output.

* :class:`TransformerDecoderLayer` -- one pre-norm decoder block
  (masked self-attention -> cross-attention -> FFN).
* :class:`TransformerDecoder` -- a stack of decoder layers + final LayerNorm.
"""

import torch
import torch.nn as nn
from typing import Optional

from .PositionalEmbeddings import PositionalEmbedding
from .transformer_encoder import MultiHeadAttention, FeedForward


class TransformerDecoderLayer(nn.Module):
    """Single pre-norm decoder block.

    Sub-layers:
        1. Masked self-attention  (with causal / future / bidirectional mask)
        2. Cross-attention over encoder output
        3. Position-wise FFN
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        cross_attn_bias: Optional[torch.Tensor] = None,
        pe: Optional[PositionalEmbedding] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:               ``[batch, tgt_len, d_model]``
            encoder_output:  ``[batch, src_len, d_model]``
            self_attn_bias:  additive bias for decoder self-attention, combining the
                             decoder mask with the *target* padding mask.
            cross_attn_bias: additive bias for cross-attention, carrying the *source*
                             padding mask so the decoder never reads padded encoder
                             positions.
            pe:              positional embedding (RoPE applied in self-attention only;
                             cross-attention is skipped).
        """
        # 1. Self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_bias=self_attn_bias, pe=pe)
        x = self.dropout1(x) + residual

        # 2. Cross-attention
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(
            x, encoder_output, encoder_output,
            attn_bias=cross_attn_bias,
            pe=pe,
            is_cross_attention=True,
        )
        x = self.dropout2(x) + residual

        # 3. Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = self.dropout3(x) + residual
        return x


class TransformerDecoder(nn.Module):
    """Stack of :class:`TransformerDecoderLayer` blocks + final LayerNorm.

    Args:
        num_layers: number of decoder layers.
        d_model:    model / embedding dimension.
        num_heads:  attention heads per layer.
        d_ff:       inner dimension of the feed-forward network.
        dropout:    dropout rate applied throughout.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        cross_attn_bias: Optional[torch.Tensor] = None,
        pe: Optional[PositionalEmbedding] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_bias, cross_attn_bias, pe)
        return self.final_norm(x)
