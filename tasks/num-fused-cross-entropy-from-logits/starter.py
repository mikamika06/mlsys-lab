import math

def fused_cross_entropy(logits: list[list[float]], targets: list[int]) -> list[float]:
    """Per-example cross-entropy loss ell_i = logsumexp(logits[i]) - logits[i, targets[i]],
    computed via the numerically-stable log-sum-exp trick (fully vectorised)."""
    raise NotImplementedError('your code here')
