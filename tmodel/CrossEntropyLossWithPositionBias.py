

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossWithPositionBias(nn.Module):
    """Cross-entropy over positions against a Gaussian soft label around the true one.

    The target is not one-hot: position ``j`` gets the weight

        q_j  proportional to  exp(-(j - t)^2 / (2 * sigma^2))

    normalised to sum to 1 over the positions, where ``t`` is the true position, and

        loss = mean_i( -sum_j q_ij * log p_ij )

    A prediction a few positions off still earns most of the credit. As ``sigma`` goes
    to 0 the label becomes one-hot and this is ``nn.CrossEntropyLoss``. The smallest
    possible loss is the label's entropy, not 0 (about 3.5 at ``sigma=8``).

    Args:
        sigma:        standard deviation of the soft label, in positions.
        ignore_index: targets with this value are left out.
    """

    def __init__(self, sigma: float = 8.0, ignore_index: int = -100):
        super().__init__()
        self.sigma = sigma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: ``[N, P]``, one finite score per position.
            target: ``[N]`` true position indices in ``[0, P)``, or ``ignore_index``.

        Returns:
            Scalar: the mean over the targets that are not ignored (``nan`` if all are,
            as ``nn.CrossEntropyLoss`` gives).
        """
        num_positions = logits.size(-1)
        keep = target != self.ignore_index
        logits, target = logits[keep].float(), target[keep]

        positions = torch.arange(num_positions, device=logits.device)
        # [n, P]: how far each position is from that row's true position.
        distance = (positions.unsqueeze(0) - target.unsqueeze(1)).abs()
        # softmax of the Gaussian exponent: each row of the label sums to 1.
        soft = F.softmax(-distance.float().pow(2) / (2 * self.sigma ** 2), dim=-1)
        return F.cross_entropy(logits, soft)
