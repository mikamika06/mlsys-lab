import numpy as np


def naive_merge(chunk_scores, chunk_values):
    """
    Merge C chunks of RAW (unstabilized) attention scores/values into the
    combined softmax-weighted average, WITHOUT any max-subtraction:
        output = sum_i sum_j exp(L_i[j]) * V_i[j] / sum_i sum_j exp(L_i[j])
    Returns a (d,) vector. See task.md.
    """
    raise NotImplementedError('your code here')


def lse_merge(chunk_scores, chunk_values):
    """
    Same target quantity as naive_merge, computed via the numerically
    stable log-sum-exp merge (local per-chunk stabilization, then a
    running-max rescale across chunks). Returns a (d,) vector. See task.md.
    """
    raise NotImplementedError('your code here')
