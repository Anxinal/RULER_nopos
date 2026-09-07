"""Transformer encoder with pluggable positional embeddings and masks.

Building blocks
~~~~~~~~~~~~~~~~
* :class:`MultiHeadAttention` -- multi-head attention that delegates to a
  :class:`~PositionalEmbeddings.PositionalEmbedding` for RoPE / ALiBi
  integration at the attention-score level.
* :class:`FeedForward` -- position-wise FFN (Linear -> GELU -> Linear).
* :class:`TransformerEncoderLayer` -- one pre-norm encoder block.
* :class:`TransformerEncoder` -- a stack of encoder layers + final LayerNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .PositionalEmbeddings import PositionalEmbedding


# ---------------------------------------------------------------------------
# Shared primitives (also imported by transformer_decoder)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    Handles three positional-embedding strategies transparently:

    * **additive** -- PE already summed into the input; nothing extra here.
    * **rotary**   -- Q and K are rotated before the dot product.
    * **alibi**    -- a per-head linear bias is folded into ``attn_bias`` by the caller.

    Attention is computed with :func:`torch.nn.functional.scaled_dot_product_attention`,
    so the ``[batch, heads, q_len, k_len]`` weight matrix is never materialised or stored
    for the backward pass. Everything that would previously have been added to the logits
    -- the causal / future-only mask, the ALiBi bias, and the key padding mask -- arrives
    pre-combined in ``attn_bias``, built once per forward pass by
    :class:`~transformer.MaskedTransformer` rather than once per layer.

    Args:
        d_model:   model / embedding dimension.
        num_heads: number of attention heads (must divide *d_model*).
        dropout:   dropout on attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        pe: Optional[PositionalEmbedding] = None,
        is_cross_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query: ``[batch, q_len, d_model]``
            key:   ``[batch, k_len, d_model]``
            value: ``[batch, k_len, d_model]``
            attn_bias: additive bias broadcastable to ``[batch, heads, q_len, k_len]``,
                       already combining the attention mask, the ALiBi bias and the key
                       padding mask. ``0`` = attend, large negative = block.
            pe: optional positional embedding. Only RoPE is applied here; ALiBi is
                folded into *attn_bias* by the caller.
            is_cross_attention: if ``True``, skip RoPE (the two sequences have
                                incompatible position spaces).
        """
        bsz, q_len, _ = query.shape
        k_len = key.shape[1]

        # Project and reshape -> [bsz, heads, seq, head_dim]
        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE (self-attention only)
        if pe is not None and pe.pe_type == "rotary" and not is_cross_attention:
            q, k = pe.rotate_queries_and_keys(q, k)

        # SDPA requires the mask dtype to match the query dtype. The caller builds the
        # bias in the right dtype already, so this is normally a no-op; it is kept as a
        # guard for direct callers and for autocast edge cases.
        if attn_bias is not None and attn_bias.dtype != q.dtype:
            attn_bias = attn_bias.to(q.dtype)

        # SDPA applies the 1/sqrt(head_dim) scaling itself, so self.scale is not applied.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        # Combine heads
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Single pre-norm encoder block: self-attention -> FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        pe: Optional[PositionalEmbedding] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_bias=attn_bias, pe=pe)
        x = self.dropout1(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout2(x) + residual
        return x


class TransformerEncoder(nn.Module):
    """Stack of :class:`TransformerEncoderLayer` blocks + final LayerNorm.

    Args:
        num_layers: number of encoder layers.
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
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        pe: Optional[PositionalEmbedding] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, pe=pe)
        return self.final_norm(x)
