"""tmodel -- Masked Transformer with pluggable positional embeddings.

Quick start::

    from scripts.tmodel import MaskedTransformer

    model = MaskedTransformer(
        vocab_size=32000,
        pe_type="rope",             # none | sinusoidal | learned | rope | alibi
        encoder_mask_spec="CCCCFFFF",  # one code per head: B | C | F
                                       # (the decoder is always causal)
    )
"""

from .masks import (
    Mask,
    CausalMask,
    FutureOnlyMask,
    BidirectionalMask,
    build_additive_mask,
    build_head_mask_bias,
    parse_mask_spec,
    clear_mask_cache,
)
from .PositionalEmbeddings import (
    PositionalEmbedding,
    NoPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    ALiBiPositionalEmbedding,
    build_positional_embedding,
)
from .transformer_encoder import (
    MultiHeadAttention,
    FeedForward,
    TransformerEncoderLayer,
    TransformerEncoder,
)
from .transformer_decoder import TransformerDecoderLayer, TransformerDecoder
from .transformer import MaskedTransformer
