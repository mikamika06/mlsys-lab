import numpy as np


def gumbel_softmax_relaxed(logits: np.ndarray, g: np.ndarray, tau: float) -> np.ndarray:
    """
    Return the Gumbel-softmax relaxation softmax((logits + g) / tau,
    axis=-1) for externally supplied (fixed) Gumbel noise g. See
    task.md.
    """
    raise NotImplementedError('your code here')
