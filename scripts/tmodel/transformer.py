"""Full encoder-decoder transformer with configurable positional embeddings
and attention masks.

Usage example::

    model = MaskedTransformer(
        vocab_size=32000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=8,
        num_decoder_layers=8,
        pe_type="sinusoidal",        # none | sinusoidal | learned | rope | alibi
        encoder_mask_type="B",       # B(idirectional) | C(ausal) | F(uture)
        decoder_mask_type="C",       # C(ausal) is standard for decoders
    )

    logits = model(src_tokens, tgt_tokens)          # training
    generated = model.generate(src_tokens)           # inference
"""

import torch
import torch.nn as nn
from typing import Optional

from .PositionalEmbeddings import build_positional_embedding
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder
from .masks import CausalMask, FutureOnlyMask


class MaskedTransformer(nn.Module):
    """Encoder-decoder transformer with pluggable PE and masks.

    Args:
        vocab_size:          vocabulary size (shared between encoder/decoder).
        d_model:             model / embedding dimension.
        num_heads:           attention heads per layer.
        num_encoder_layers:  depth of the encoder stack.
        num_decoder_layers:  depth of the decoder stack.
        d_ff:                inner dimension of the feed-forward sublayers.
        dropout:             dropout rate used throughout.
        max_len:             maximum sequence length (for PE buffers).
        pe_type:             positional embedding type
                             (``"none"``, ``"sinusoidal"``, ``"learned"``,
                             ``"rope"``, ``"alibi"``).
        encoder_mask_type:   ``"B"`` bidirectional, ``"C"`` causal,
                             ``"F"`` future-only.
        decoder_mask_type:   same codes; ``"C"`` is the standard choice.
        pad_token_id:        token id used for padding.
        tie_weights:         if ``True`` (default), the decoder input
                             embedding and output projection share weights.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 8192,
        pe_type: str = "sinusoidal",
        encoder_mask_type: str = "B",
        decoder_mask_type: str = "C",
        pad_token_id: int = 0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.encoder_mask_type = encoder_mask_type
        self.decoder_mask_type = decoder_mask_type
        self.embed_scale = d_model ** 0.5

        # Token embeddings (shared between encoder and decoder)
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embed_dropout = nn.Dropout(dropout)

        # Positional embedding
        self.pe = build_positional_embedding(pe_type, d_model, num_heads, max_len)

        # Encoder / decoder stacks
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)

        # Output projection (vocab logits)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.output_proj.weight = self.embed_tokens.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "embed" not in name:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mask(
        mask_type: str, dim: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return a ``[dim, dim]`` additive mask or ``None`` (bidirectional)."""
        if mask_type == "C":
            return CausalMask(dim).tensor.to(device)
        if mask_type == "F":
            return FutureOnlyMask(dim).tensor.to(device)
        return None  # "B" -- bidirectional: no mask needed

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens, scale, add additive PE, apply dropout."""
        x = self.embed_tokens(tokens) * self.embed_scale
        if self.pe.pe_type == "additive":
            x = x + self.pe(tokens.shape[1], tokens.device)
        return self.embed_dropout(x)

    def encode(self, src_tokens: torch.Tensor) -> torch.Tensor:
        """Run the encoder on *src_tokens* and return hidden states.

        Args:
            src_tokens: ``[batch, src_len]``

        Returns:
            ``[batch, src_len, d_model]``
        """
        x = self._embed(src_tokens)
        attn_mask = self._build_mask(
            self.encoder_mask_type, src_tokens.shape[1], src_tokens.device,
        )
        return self.encoder(x, attn_mask=attn_mask, pe=self.pe)

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decoder and return vocab logits.

        Args:
            tgt_tokens:     ``[batch, tgt_len]``
            encoder_output: ``[batch, src_len, d_model]``

        Returns:
            ``[batch, tgt_len, vocab_size]``
        """
        x = self._embed(tgt_tokens)
        self_attn_mask = self._build_mask(
            self.decoder_mask_type, tgt_tokens.shape[1], tgt_tokens.device,
        )
        hidden = self.decoder(x, encoder_output, self_attn_mask=self_attn_mask, pe=self.pe)
        return self.output_proj(hidden)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass (training).

        Args:
            src_tokens: ``[batch, src_len]``
            tgt_tokens: ``[batch, tgt_len]``

        Returns:
            Logits ``[batch, tgt_len, vocab_size]``
        """
        return self.decode(tgt_tokens, self.encode(src_tokens))

    # ------------------------------------------------------------------
    # Generation (greedy / sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        src_tokens: torch.Tensor,
        max_new_tokens: int = 64,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Auto-regressive generation from encoder output.

        Args:
            src_tokens:     ``[batch, src_len]``
            max_new_tokens: maximum tokens to generate.
            bos_token_id:   id of the beginning-of-sequence token.
            eos_token_id:   id of the end-of-sequence token.
            temperature:    softmax temperature (0 -> greedy argmax).
            top_k:          if > 0, only sample from the *top_k* logits.

        Returns:
            ``[batch, generated_len]`` including the initial BOS token.
        """
        encoder_output = self.encode(src_tokens)
        bsz = src_tokens.shape[0]
        generated = torch.full(
            (bsz, 1), bos_token_id,
            dtype=torch.long, device=src_tokens.device,
        )

        for _ in range(max_new_tokens):
            logits = self.decode(generated, encoder_output)
            next_logits = logits[:, -1, :]  # last position

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < topk_vals[:, [-1]]] = float("-inf")
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated
