"""Token-level wrappers that give every tmodel model the same interface.

Training treats a model as ``logits = model(src, tgt_in)``: ``src`` is the
right-padded prompt ``[batch, src_len]``, ``tgt_in`` is the right-padded answer
``[batch, tgt_len]`` starting with BOS, and ``logits`` is ``[batch, tgt_len, vocab]``,
aligned so that position ``t`` predicts ``tgt_out[:, t]``. Prediction uses
``model.generate(src, ...)`` (greedy). Every wrapper also carries ``pad_token_id`` and a
one-line ``description`` for the log.

Each model is trained in its native form:

* ``transformer_mask`` is an encoder-decoder, so the prompt goes to the encoder and the
  answer to the decoder.
* ``roformer`` and ``alibi`` are decoder-only, so each sample becomes one sequence,
  prompt followed by answer, and only the answer positions are scored.
"""

import torch
import torch.nn as nn

from .alibi import ALiBiConfig, ALiBiTransformer
from .maskedVanilla import TransformerMask

MODEL_TYPES = ("transformer_mask", "roformer", "alibi")
# Positional encodings for transformer_mask. nn.Transformer ships none, so "sinusoidal"
# comes from torch_geometric; roformer and alibi carry their own.
PE_TYPES = ("none", "sinusoidal")


class TokenLM(nn.Module):
    """Greedy decoding shared by every wrapper.

    Subclasses provide ``prepare(src)``, the per-prompt work done once (the encoder
    pass, for the encoder-decoder), and ``next_token_logits(state, tgt_in)``.
    """

    pad_token_id: int
    description: str

    def prepare(self, src: torch.Tensor):
        return src

    def next_token_logits(self, state, tgt_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_new_tokens: int, bos_token_id: int,
                 eos_token_id: int) -> torch.Tensor:
        """Greedy-decode up to *max_new_tokens* after a BOS token.

        Returns ``[batch, 1 + n]``: BOS followed by the generated tokens, with EOS kept
        where a row stopped and padding after it while other rows continue.
        """
        state = self.prepare(src)
        out = src.new_full((src.size(0), 1), bos_token_id)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_new_tokens):
            nxt = self.next_token_logits(state, out).argmax(dim=-1)
            nxt = nxt.masked_fill(done, self.pad_token_id)
            out = torch.cat([out, nxt.unsqueeze(1)], dim=1)
            done |= nxt.eq(eos_token_id)
            if done.all():
                break
        return out


class TransformerMaskLM(TokenLM):
    """:class:`TransformerMask` with token embeddings and an output projection.

    ``nn.Transformer`` works on vectors, not token ids, so this adds the embedding in
    front and the vocabulary projection behind it.

    ``pe="sinusoidal"`` adds the sinusoidal encoding of "Attention Is All You Need",
    from ``torch_geometric.nn.encoding.PositionalEncoding``, to both the encoder and
    the decoder input, as the original Transformer does. It is the only difference
    from ``pe="none"``: the embeddings are neither scaled by sqrt(d_model) nor followed
    by an extra dropout, so the two arms differ in the encoding alone. With
    ``pe="none"`` position reaches the encoder only through ``mask_spec``, and the
    decoder only through its causal mask.
    """

    def __init__(self, vocab_size: int, pad_token_id: int, mask_spec: str,
                 pe: str = "none"):
        super().__init__()
        if pe not in PE_TYPES:
            raise ValueError(f"unknown pe {pe!r}; expected one of {PE_TYPES}")
        self.pad_token_id = pad_token_id
        self.transformer = TransformerMask(mask_spec)
        d_model = self.transformer.d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = None
        if pe == "sinusoidal":
            # Imported only for this arm: torch_geometric is a heavy import the other
            # models do not need.
            from torch_geometric.nn.encoding import PositionalEncoding
            self.pos_encoding = PositionalEncoding(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.description = (f"transformer_mask (encoder-decoder, nn.Transformer defaults) "
                            f"| enc_mask={mask_spec} | pe={pe}")

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        """``[batch, len]`` ids -> ``[len, batch, d_model]`` (nn.Transformer is seq-first).

        Used for both the encoder and the decoder input, so both get the encoding.
        """
        x = self.embed(ids)
        if self.pos_encoding is not None:
            # Sequences are right-padded, so every real token's position counts from 0.
            positions = torch.arange(ids.size(1), device=ids.device)
            x = x + self.pos_encoding(positions)   # [len, d_model], broadcast over batch
        return x.transpose(0, 1)

    def encode(self, src: torch.Tensor):
        src_pad = src.eq(self.pad_token_id)
        memory = self.transformer.encoder(self._embed(src), src_key_padding_mask=src_pad)
        return memory, src_pad

    def decode(self, state, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_pad = state
        tgt_len = tgt_in.size(1)
        # A bool mask to match the bool padding masks; nn.MultiheadAttention warns when
        # the two kinds are mixed. True = blocked.
        tgt_mask = torch.ones(tgt_len, tgt_len, dtype=torch.bool,
                              device=tgt_in.device).triu(1)
        out = self.transformer.decoder(
            self._embed(tgt_in), memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_in.eq(self.pad_token_id),
            memory_key_padding_mask=src_pad,
        )
        return self.lm_head(out.transpose(0, 1))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(src), tgt_in)

    # Generation encodes the prompt once and re-runs only the decoder per token.
    def prepare(self, src: torch.Tensor):
        return self.encode(src)

    def next_token_logits(self, state, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode(state, tgt_in)[:, -1]


def pack_prompt_and_answer(src: torch.Tensor, tgt_in: torch.Tensor, pad_token_id: int):
    """Join each prompt and its answer into one right-padded sequence.

    ``src`` is right-padded, so simply concatenating ``[src, tgt_in]`` would leave a
    run of padding between a short prompt and its answer. Instead each answer starts
    straight after its own prompt, and all padding ends up at the right.

    Right padding is what makes this safe without a padding mask (the ALiBi model has
    none): under a causal mask a real token only attends to earlier positions, and
    those are all real.

    Returns:
        ``(seq, answer_pos)``: ``seq`` is ``[batch, src_len + tgt_len]`` and
        ``answer_pos[b, t]`` is where ``tgt_in[b, t]`` sits in ``seq``.
    """
    bsz, src_len = src.shape
    tgt_len = tgt_in.size(1)
    prompt_len = src.ne(pad_token_id).sum(dim=1, keepdim=True)
    answer_pos = prompt_len + torch.arange(tgt_len, device=src.device)
    seq = src.new_full((bsz, src_len + tgt_len), pad_token_id)
    seq[:, :src_len] = src
    seq.scatter_(1, answer_pos, tgt_in)
    return seq, answer_pos


class DecoderOnlyLM(TokenLM):
    """Base for decoder-only models: score the answer given the prompt before it.

    Subclasses provide ``hidden(seq)`` and ``head(hidden)``. The head runs only on the
    answer positions: at ``src_len=2048`` the prompt is ~95% of the sequence, and
    projecting it onto a 50k vocabulary would cost gigabytes for logits nothing reads.
    """

    pad_token_id: int

    def hidden(self, seq: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        seq, answer_pos = pack_prompt_and_answer(src, tgt_in, self.pad_token_id)
        hidden = self.hidden(seq)
        index = answer_pos.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        return self.head(hidden.gather(1, index))

    def next_token_logits(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        # No KV cache: each step re-runs prompt + answer so far. Answers are a few dozen
        # tokens, so this costs a few dozen forward passes per prompt.
        return self(src, tgt_in)[:, -1]


class RoFormerLM(DecoderOnlyLM):
    """``RoFormerForCausalLM`` from ``tmodel.roformer`` (rotary position embeddings).

    Uses ``RoFormerConfig``'s defaults (768 wide, 12 heads, 12 layers, feed-forward
    3072, dropout 0.1). Only what the data or the decoder-only setup requires is set:
    the vocabulary, the pad id, ``is_decoder`` (for the causal mask) and ``max_len``.
    """

    def __init__(self, vocab_size: int, pad_token_id: int, max_len: int):
        super().__init__()
        # Imported here, not at module level: it needs transformers>=5, and the other
        # two models should not.
        from .roformer import RoFormerConfig, RoFormerForCausalLM

        self.pad_token_id = pad_token_id
        config = RoFormerConfig(vocab_size=vocab_size, pad_token_id=pad_token_id,
                                max_position_embeddings=max_len, is_decoder=True,
                                use_cache=False)
        self.model = RoFormerForCausalLM(config)
        self.description = (f"roformer (decoder-only, RoPE) | RoFormerConfig defaults: "
                            f"{config.hidden_size} wide, {config.num_attention_heads} heads, "
                            f"{config.num_hidden_layers} layers, ff {config.intermediate_size}, "
                            f"dropout {config.hidden_dropout_prob} | max_len={max_len}")

    def hidden(self, seq: torch.Tensor) -> torch.Tensor:
        # No attention_mask: the causal mask alone keeps real tokens off the padding,
        # which is all on the right (see pack_prompt_and_answer).
        return self.model.roformer(input_ids=seq, use_cache=False).last_hidden_state

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.cls(hidden)


class ALiBiLM(DecoderOnlyLM):
    """:class:`ALiBiTransformer` with token embeddings, a final norm and an output head.

    Uses ``ALiBiConfig``'s defaults (see ``tmodel/alibi/config.py``); only ``max_len``
    is set. The upstream model works
    on vectors and its pre-norm layers end without a final LayerNorm, so the embedding,
    the final norm and the output head are added here.
    """

    def __init__(self, vocab_size: int, pad_token_id: int, max_len: int):
        super().__init__()
        self.pad_token_id = pad_token_id
        config = ALiBiConfig(max_len=max_len)
        if not config.causal:
            raise ValueError("ALiBiConfig must be causal: the model is trained decoder-only.")
        self.embed = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_token_id)
        self.transformer = ALiBiTransformer(config)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size)
        self.description = (f"alibi (decoder-only, ALiBi) | ALiBiConfig defaults: "
                            f"{config.d_model} wide, {config.num_heads} heads, "
                            f"{config.num_layers} layers, ff x{config.expansion_factor}, "
                            f"dropout {config.dropout} | max_len={max_len}")

    def hidden(self, seq: torch.Tensor) -> torch.Tensor:
        return self.norm(self.transformer(self.embed(seq)))

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


def build_model(model_type: str, vocab_size: int, pad_token_id: int, *,
                mask_spec: str = "B", pe: str = "none", max_len: int = 2176) -> TokenLM:
    """Build one of :data:`MODEL_TYPES` with the shared ``model(src, tgt_in)`` interface.

    Every model uses its own library's default hyperparameters: ``nn.Transformer``'s for
    ``transformer_mask``, ``RoFormerConfig``'s for ``roformer`` and ``ALiBiConfig``'s for
    ``alibi``. The only arguments are what the data dictates:

    * ``mask_spec`` -- ``transformer_mask`` only, the per-head encoder mask.
    * ``pe``        -- ``transformer_mask`` only, ``"none"`` or ``"sinusoidal"``.
    * ``max_len``   -- ``roformer`` / ``alibi`` only, the longest prompt + answer.
      Their defaults are shorter than a RULER sample (2048-token prompts), so it has
      to be set; ``train.py`` passes ``src_len + tgt_len``. The ALiBi model allocates a
      ``[max_len, max_len]`` causal mask in every layer.
    """
    if model_type == "transformer_mask":
        return TransformerMaskLM(vocab_size, pad_token_id, mask_spec, pe)
    if pe != "none":
        raise ValueError(f"pe={pe!r} applies to transformer_mask only; {model_type} "
                         f"has its own position encoding.")
    if model_type == "roformer":
        return RoFormerLM(vocab_size, pad_token_id, max_len)
    if model_type == "alibi":
        return ALiBiLM(vocab_size, pad_token_id, max_len)
    raise ValueError(f"unknown model {model_type!r}; expected one of {MODEL_TYPES}")
