"""Positional embedding implementations for the masked transformer.

Each subclass sets ``pe_type`` to declare how it integrates with attention:

* ``"additive"`` -- added to token embeddings before the transformer layers.
* ``"rotary"``  -- applied to Q and K inside every attention head (RoPE).
* ``"alibi"``   -- produces a per-head bias added to attention logits.

Use :func:`build_positional_embedding` to instantiate by name.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PositionalEmbedding(nn.Module):
    """Abstract base for all positional-embedding strategies."""

    pe_type: str = "additive"

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return a ``[1, seq_len, d_model]`` tensor to add to embeddings.

        Non-additive types return zeros (the real work happens in
        :meth:`rotate_queries_and_keys` or :meth:`attention_bias`).
        """
        raise NotImplementedError

    def rotate_queries_and_keys(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to *q* and *k*.

        Args:
            q, k: ``[batch, heads, seq_len, head_dim]``
            offset: starting position index (useful for incremental decoding).

        Default: identity (no-op).
        """
        return q, k

    def attention_bias(
        self, q_len: int, k_len: int, num_heads: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return additive bias ``[num_heads, q_len, k_len]`` or ``None``."""
        return None


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class NoPositionalEmbedding(PositionalEmbedding):
    """No positional information -- pure bag-of-tokens baseline."""

    pe_type = "additive"

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, seq_len, self.d_model, device=device)


class SinusoidalPositionalEmbedding(PositionalEmbedding):
    """Fixed sinusoidal encodings from *Attention Is All You Need*."""

    pe_type = "additive"

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.pe[:, :seq_len, :].to(device)


class LearnedPositionalEmbedding(PositionalEmbedding):
    """Learned absolute position embeddings (one vector per position)."""

    pe_type = "additive"

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions).unsqueeze(0)  # [1, seq_len, d_model]


class RotaryPositionalEmbedding(PositionalEmbedding):
    """Rotary Position Embedding (RoPE).

    Encodes position by rotating pairs of dimensions in query/key vectors,
    so that the dot product between q_i and k_j depends on *i - j*.
    """

    pe_type = "rotary"

    def __init__(self, d_model: int, num_heads: int, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        head_dim = d_model // num_heads

        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)                # [max_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)          # [max_len, head_dim]
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, seq_len, self.d_model, device=device)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def rotate_queries_and_keys(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_len = q.shape[2]
        k_len = k.shape[2]

        cos_q = self.cos_cached[offset:offset + q_len].to(q.device)[None, None]
        sin_q = self.sin_cached[offset:offset + q_len].to(q.device)[None, None]
        q_rot = q * cos_q + self._rotate_half(q) * sin_q

        cos_k = self.cos_cached[offset:offset + k_len].to(k.device)[None, None]
        sin_k = self.sin_cached[offset:offset + k_len].to(k.device)[None, None]
        k_rot = k * cos_k + self._rotate_half(k) * sin_k

        return q_rot, k_rot


class ALiBiPositionalEmbedding(PositionalEmbedding):
    """Attention with Linear Biases (ALiBi).

    Adds ``-slope * |i - j|`` to every attention logit, where the slope is
    a fixed geometric sequence that differs per head.
    """

    pe_type = "alibi"

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", torch.tensor(slopes, dtype=torch.float))

    @staticmethod
    def _get_slopes(num_heads: int) -> List[float]:
        """Geometric slope schedule from the ALiBi paper."""

        def _power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            return _power_of_2(num_heads)

        closest = 2 ** math.floor(math.log2(num_heads))
        base = _power_of_2(closest)
        extra = _power_of_2(2 * closest)
        base.extend(extra[0::2][: num_heads - closest])
        return base

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, seq_len, self.d_model, device=device)

    def attention_bias(
        self, q_len: int, k_len: int, num_heads: int, device: torch.device,
    ) -> torch.Tensor:
        """``-slope_h * |q_pos - k_pos|``  ->  ``[num_heads, q_len, k_len]``."""
        q_pos = torch.arange(k_len - q_len, k_len, device=device, dtype=torch.float)
        k_pos = torch.arange(k_len, device=device, dtype=torch.float)
        distance = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()   # [q, k]
        slopes = self.slopes.to(device)[:, None, None]                # [h, 1, 1]
        return -slopes * distance.unsqueeze(0)                        # [h, q, k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_positional_embedding(
    pe_type: str,
    d_model: int,
    num_heads: int = 8,
    max_len: int = 8192,
) -> PositionalEmbedding:
    """Instantiate a positional embedding by name (case-insensitive).

    Accepted names:
        ``"none"`` | ``"no"``  ,  ``"sinusoidal"`` | ``"sin"``  ,
        ``"learned"`` | ``"learn"``  ,  ``"rope"`` | ``"rotary"``  ,
        ``"alibi"``
    """
    name = pe_type.lower()
    if name in ("none", "no"):
        return NoPositionalEmbedding(d_model)
    if name in ("sinusoidal", "sin"):
        return SinusoidalPositionalEmbedding(d_model, max_len)
    if name in ("learned", "learn"):
        return LearnedPositionalEmbedding(d_model, max_len)
    if name in ("rope", "rotary"):
        return RotaryPositionalEmbedding(d_model, num_heads, max_len)
    if name == "alibi":
        return ALiBiPositionalEmbedding(d_model, num_heads)
    raise ValueError(f"Unknown positional embedding type: {pe_type!r}")
