import numpy as np


def topk_gating(logits: np.ndarray, k: int):
    """Top-k MoE router gating. See task.md for the exact selection and
    weighting rule.

    Returns
    -------
    indices : (N, k) int64 -- selected expert index per token, per rank.
    weights : (N, k) float64 -- softmax gate weight per selected expert.
    """
    raise NotImplementedError('your code here')
