"""Extract one layer's hidden state from a model on one input.

The hidden state of layer ``i`` is the residual stream of that layer right after its
attention has been added, BEFORE the LayerNorm and the feed-forward block that follow.
Every model here has exactly one LayerNorm sitting at that point, so the state is read
as that LayerNorm's input:

    transformer_mask, encoder layer    input of ``norm2``
    transformer_mask, decoder layer    input of ``norm3`` (after self- and cross-attention)
    alibi layer                        input of ``ffn_norm``
    roformer layer                     input of ``attention.output.LayerNorm``
"""

import torch

from ..models import DecoderOnlyLM


def _layers_and_norm(model, stack):
    """The layer list of *stack*, and the name of the LayerNorm the state is read at."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "decoder"):
        # transformer_mask (nn.Transformer). Reading the norm's INPUT is only "before
        # LayerNorm and FFN" when the layers are pre-norm, which they are.
        tr = model.transformer
        if stack == "encoder":
            return tr.encoder.layers, "norm2"
        return tr.decoder.layers, "norm3"
    if stack != "decoder":
        raise ValueError("this model is decoder-only; use stack='decoder'")
    if hasattr(model, "transformer"):                      # alibi
        return model.transformer.layers, "ffn_norm"
    return model.model.roformer.encoder.layer, "attention.output.LayerNorm"   # roformer


@torch.no_grad()
def extract_hidden_state(model, src, tgt_in, layer, stack="encoder"):
    """Return layer *layer*'s hidden state for one forward pass of ``model(src, tgt_in)``.

    The model is run in eval mode (and put back as it was), under bf16 autocast when the
    input is on a GPU, as prediction runs it.

    Args:
        model:  a model built by ``tmodel.models.build_model``, with trained weights loaded.
        src:    ``[batch, src_len]`` prompt ids.
        tgt_in: ``[batch, tgt_len]`` decoder input ids, starting with BOS. Pass BOS alone
                for the state at the first decoding step.
        layer:  layer index in the stack, 0-based; negative counts from the last.
        stack:  ``"encoder"`` or ``"decoder"``. The decoder-only models (roformer, alibi)
                have only the latter.

    Returns:
        ``[batch, seq, d_model]`` float32 on the CPU. ``seq`` is ``src_len`` for the
        transformer_mask encoder, ``tgt_len`` for its decoder, and ``src_len + tgt_len``
        for the decoder-only models, whose sequence is each prompt followed directly by
        its answer (``tmodel.models.pack_prompt_and_answer``).
    """
    layers, norm_name = _layers_and_norm(model, stack)
    norm = layers[layer].get_submodule(norm_name)

    captured = []
    handle = norm.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    was_training = model.training
    model.eval()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=src.is_cuda):
            model(src, tgt_in)
    finally:
        handle.remove()
        model.train(was_training)
    if len(captured) != 1:
        raise RuntimeError(f"expected the hook on {norm_name} to fire once, got "
                           f"{len(captured)}; the layer did not run its Python forward.")

    hidden = captured[0]
    if not isinstance(model, DecoderOnlyLM):
        hidden = hidden.transpose(0, 1)        # nn.Transformer is [seq, batch, d_model]
    return hidden.float().cpu()
