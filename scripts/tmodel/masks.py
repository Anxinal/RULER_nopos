
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
