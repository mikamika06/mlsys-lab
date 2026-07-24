import numpy as np


def streaming_logsumexp(chunks):
    """Per-row log-sum-exp computed by streaming over column chunks.

    chunks: list of (N, w_k) float arrays tiling a full (N, D) matrix along
    columns. Process them one at a time with the running max/sum recurrence
    -- never concatenate them into the full matrix. Returns a (N,) float64
    array of per-row LSE values.
    """
    raise NotImplementedError('your code here')
