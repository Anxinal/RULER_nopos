
import torch
from abc import ABC, abstractmethod


def fill_with_neg_inf(t: torch.Tensor) -> torch.Tensor:
    """Fill a tensor with -inf (in-place) and return it."""
    return t.fill_(float("-inf"))




class Mask(ABC):
    """Base class for a square additive attention mask.

    Subclasses implement ``_build`` to return a ``[dim, dim]`` float tensor
    where masked positions hold ``-inf`` and unmasked positions hold ``0``.
    ``apply`` adds the mask to an attention-weight tensor in place of the
    standard ``attn_weights += attn_mask`` pattern used throughout fairseq.
    """

    def __init__(self, dim: int):
        self._mask = self._build(dim)

    @abstractmethod
    def _build(self, dim: int) -> torch.Tensor: ...

    @property
    def tensor(self) -> torch.Tensor:
        """The raw ``[dim, dim]`` mask tensor (0 = attend, -inf = block)."""
        return self._mask

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Add the mask to *x*, broadcasting over any leading batch/head dims."""
        return x + self._mask.to(x)

    @classmethod
    def convert_from_config(cls, config: str):
        """Return the Mask **subclass** for a one-letter config code.

        ``"C"`` → CausalMask, ``"F"`` → FutureOnlyMask, anything else →
        BidirectionalMask.  The caller is responsible for instantiating the
        returned class with the appropriate ``dim``.
        """
        if config == "C":
            return CausalMask
        elif config == "F":
            return FutureOnlyMask
        return BidirectionalMask

class CausalMask(Mask):
    """Standard autoregressive (causal) mask.

    Each token may attend to itself and all *earlier* tokens; future
    positions (j > i) are set to -inf.

    Example (dim=4):
        [  0  -∞  -∞  -∞ ]
        [  0   0  -∞  -∞ ]
        [  0   0   0  -∞ ]
        [  0   0   0   0 ]
    """

    def _build(self, dim: int) -> torch.Tensor:
        return torch.triu(fill_with_neg_inf(torch.zeros(dim, dim)), 1)


class FutureOnlyMask(Mask):
    """Anti-causal (future-only) mask.

    Each token may attend to itself and all *later* tokens; past
    positions (j < i) are set to -inf.  Used in encoder self-attention
    experiments — it must not be applied to the decoder, where it would
    break the autoregressive property.

    Args:
        dim:        Sequence length (mask will be ``[dim, dim]``).
        allow_self: If ``True`` (default) the diagonal is kept at 0 so
                    each token can still attend to itself.  If ``False``
                    the diagonal is also masked, which produces empty
                    attention rows for the last token and should be used
                    with care.

    """

    def __init__(self, dim: int, allow_self: bool = True):
        self.allow_self = allow_self
        super().__init__(dim)

    def _build(self, dim: int) -> torch.Tensor:
        diagonal = -1 if self.allow_self else 0
        return torch.tril(fill_with_neg_inf(torch.zeros(dim, dim)), diagonal)


class BidirectionalMask(Mask):
    """No-op mask: every position may attend to every other position.

    Equivalent to passing ``attn_mask=None`` but lets all three mask types
    be handled uniformly in per-head mask specs.
    """

    def _build(self, dim: int) -> torch.Tensor:
        return torch.zeros(dim, dim)


# ---------------------------------------------------------------------------
# Cached additive masks
# ---------------------------------------------------------------------------

_MASK_CACHE = {}


def build_additive_mask(mask_type: str, dim: int, device, dtype) -> torch.Tensor:
    """Return a cached ``[dim, dim]`` additive mask, or ``None`` for bidirectional.

    Two differences from instantiating a :class:`Mask` directly, both of which matter
    in the training/inference hot path:

    * **Cached.** Building the tensor fresh on every forward pass costs a 268 MB
      allocation at ``dim=8192``. The cache is keyed on shape, device and dtype, and
      the number of distinct keys is bounded by the experiment grid.
    * **Finite.** Masked positions hold ``torch.finfo(dtype).min`` rather than ``-inf``.
      A row that ends up fully masked -- which happens with the future-only mask when a
      padded query position can only see later positions that are themselves padding --
      would otherwise softmax to NaN. A NaN at a padded position is not harmless: the
      decoder's cross-attention multiplies it by a zero weight, and ``0 * NaN`` is NaN,
      so it would propagate into real positions. Finite values soften such a row to a
      uniform distribution instead, and its output is discarded downstream anyway.

    Args:
        mask_type: ``"B"`` bidirectional, ``"C"`` causal, ``"F"`` future-only.
        dim:       sequence length.
        device:    target device.
        dtype:     target floating dtype (must match the attention query dtype).

    Returns:
        ``[dim, dim]`` additive mask, or ``None`` when ``mask_type`` is ``"B"``.
    """
    if mask_type == "B":
        return None  # bidirectional: nothing to add

    key = (mask_type, dim, str(device), dtype)
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    mask_cls = Mask.convert_from_config(mask_type)
    tensor = mask_cls(dim).tensor  # float32, -inf in masked positions
    neg = torch.finfo(dtype).min
    tensor = torch.nan_to_num(tensor, neginf=neg).to(device=device, dtype=dtype)
    tensor = tensor.clamp_min(neg)

    _MASK_CACHE[key] = tensor
    return tensor


def clear_mask_cache() -> None:
    """Drop every cached mask. Mainly useful in tests and to release device memory."""
    _MASK_CACHE.clear()
