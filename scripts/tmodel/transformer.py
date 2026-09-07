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
from .masks import CausalMask, FutureOnlyMask, build_additive_mask


def _compute_dtype(device: torch.device) -> torch.dtype:
    """Dtype the attention will actually run in, honouring autocast.

    Attention masks must match the query dtype that
    :func:`torch.nn.functional.scaled_dot_product_attention` sees. Under autocast the
    projections emit half precision even though the parameters are float32, so reading
    the dtype off the embedding output would give the wrong answer and force a costly
    per-layer cast of a tensor that can be gigabytes at long context.
    """
    if device.type == "cuda" and torch.is_autocast_enabled():
        # torch >= 2.4 prefers get_autocast_dtype(device_type); the older spelling is
        # what the NGC 23.10 image (torch 2.1) provides.
        if hasattr(torch, "get_autocast_dtype"):
            return torch.get_autocast_dtype("cuda")
        return torch.get_autocast_gpu_dtype()
    return torch.get_default_dtype()


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
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_len = max_len
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

    def _padding_mask(self, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """``[batch, seq]`` bool mask, ``True`` where the token is padding.

        Returns ``None`` when the batch contains no padding, so the common
        single-sequence inference path allocates nothing.
        """
        pad = tokens.eq(self.pad_token_id)
        return pad if pad.any() else None

    def _build_attn_bias(
        self,
        mask_type: Optional[str],
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
        key_padding_mask: Optional[torch.Tensor] = None,
        use_alibi: bool = False,
    ) -> Optional[torch.Tensor]:
        """Combine attention mask, ALiBi bias and key padding into one additive tensor.

        Built once per forward pass and shared by every layer, rather than reconstructed
        inside each attention call. Shapes are kept broadcastable so nothing is expanded
        to the full ``[batch, heads, q_len, k_len]`` unless ALiBi genuinely requires a
        per-head term.

        Args:
            mask_type: ``"B"``, ``"C"``, ``"F"``, or ``None`` for cross-attention.
            q_len, k_len: query and key lengths.
            device, dtype: must match the attention query tensor.
            key_padding_mask: ``[batch, k_len]`` bool, ``True`` at padded keys.
            use_alibi: whether to fold in the ALiBi per-head bias.

        Returns:
            Additive bias broadcastable to ``[batch, heads, q_len, k_len]``, or ``None``
            when nothing needs masking at all.
        """
        neg = torch.finfo(dtype).min
        bias = None

        # Per-head ALiBi term -> [1, heads, q_len, k_len]
        if use_alibi:
            alibi = self.pe.attention_bias(q_len, k_len, self.num_heads, device, dtype)
            if alibi is not None:
                bias = alibi.unsqueeze(0)

        # Square causal / future-only mask -> [1, 1, q_len, k_len]
        if mask_type is not None:
            mask = build_additive_mask(mask_type, q_len, device, dtype)
            if mask is not None:
                mask = mask[:q_len, :k_len].view(1, 1, q_len, k_len)
                bias = mask if bias is None else bias + mask

        # Key padding -> [batch, 1, 1, k_len]
        if key_padding_mask is not None:
            pad = torch.zeros(
                key_padding_mask.shape, device=device, dtype=dtype,
            ).masked_fill(key_padding_mask, neg).view(-1, 1, 1, k_len)
            bias = pad if bias is None else bias + pad

        if bias is not None:
            # Two finite minima can sum below the dtype's range and become -inf, which
            # would reintroduce the NaN that finite masking exists to avoid.
            bias = bias.clamp_min(neg)
        return bias

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens, scale, add additive PE, apply dropout."""
        x = self.embed_tokens(tokens) * self.embed_scale
        if self.pe.pe_type == "additive":
            x = x + self.pe(tokens.shape[1], tokens.device)
        return self.embed_dropout(x)

    def encode(
        self,
        src_tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the encoder on *src_tokens* and return hidden states.

        Args:
            src_tokens: ``[batch, src_len]``
            src_key_padding_mask: ``[batch, src_len]`` bool, ``True`` at padded
                positions. Derived from ``pad_token_id`` when omitted.

        Returns:
            ``[batch, src_len, d_model]``
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = self._padding_mask(src_tokens)

        x = self._embed(src_tokens)
        src_len = src_tokens.shape[1]
        attn_bias = self._build_attn_bias(
            mask_type=self.encoder_mask_type,
            q_len=src_len,
            k_len=src_len,
            device=src_tokens.device,
            dtype=_compute_dtype(src_tokens.device),
            key_padding_mask=src_key_padding_mask,
            use_alibi=self.pe.pe_type == "alibi",
        )
        return self.encoder(x, attn_bias=attn_bias, pe=self.pe)

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the decoder and return vocab logits.

        Args:
            tgt_tokens:     ``[batch, tgt_len]``
            encoder_output: ``[batch, src_len, d_model]``
            tgt_key_padding_mask: ``[batch, tgt_len]`` bool for decoder self-attention.
                Derived from ``pad_token_id`` when omitted.
            src_key_padding_mask: ``[batch, src_len]`` bool for cross-attention. Must be
                passed explicitly, since the source tokens are not available here and
                the encoder output cannot be inspected for padding.

        Returns:
            ``[batch, tgt_len, vocab_size]``
        """
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._padding_mask(tgt_tokens)

        x = self._embed(tgt_tokens)
        device = tgt_tokens.device
        dtype = _compute_dtype(device)
        tgt_len = tgt_tokens.shape[1]

        self_attn_bias = self._build_attn_bias(
            mask_type=self.decoder_mask_type,
            q_len=tgt_len,
            k_len=tgt_len,
            device=device,
            dtype=dtype,
            key_padding_mask=tgt_key_padding_mask,
            use_alibi=self.pe.pe_type == "alibi",
        )
        # Cross-attention carries only the source padding: RoPE and ALiBi are skipped
        # across the two sequences because their position spaces are incompatible.
        cross_attn_bias = self._build_attn_bias(
            mask_type=None,
            q_len=tgt_len,
            k_len=encoder_output.shape[1],
            device=device,
            dtype=dtype,
            key_padding_mask=src_key_padding_mask,
            use_alibi=False,
        )

        hidden = self.decoder(
            x, encoder_output,
            self_attn_bias=self_attn_bias,
            cross_attn_bias=cross_attn_bias,
            pe=self.pe,
        )
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
        src_key_padding_mask = self._padding_mask(src_tokens)
        encoder_output = self.encode(src_tokens, src_key_padding_mask=src_key_padding_mask)
        return self.decode(
            tgt_tokens, encoder_output,
            src_key_padding_mask=src_key_padding_mask,
        )

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
        src_key_padding_mask = self._padding_mask(src_tokens)
        encoder_output = self.encode(src_tokens, src_key_padding_mask=src_key_padding_mask)
        bsz = src_tokens.shape[0]
        generated = torch.full(
            (bsz, 1), bos_token_id,
            dtype=torch.long, device=src_tokens.device,
        )
        # Track which sequences have already emitted EOS so that, in a batch, a finished
        # sequence is padded rather than continuing to sample past its stop token.
        finished = torch.zeros(bsz, dtype=torch.bool, device=src_tokens.device)

        for _ in range(max_new_tokens):
            # Everything generated so far is a real token, so the decoder needs no target
            # padding mask; the source mask still gates cross-attention.
            logits = self.decode(
                generated, encoder_output,
                tgt_key_padding_mask=None,
                src_key_padding_mask=src_key_padding_mask,
            )
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

            # Once a sequence has emitted EOS, keep it at pad so later steps cannot
            # append spurious tokens to an answer that already ended.
            next_token = next_token.masked_fill(finished.unsqueeze(1), self.pad_token_id)
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | next_token.squeeze(1).eq(eos_token_id)

            if finished.all():
                break

        return generated
