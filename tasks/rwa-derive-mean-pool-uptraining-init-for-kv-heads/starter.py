import numpy as np


def mean_pool_gqa_logit_mse(Q: np.ndarray, K: np.ndarray, n_rep: int) -> float:
    """
    Q: (n_heads, seq_q, d) original per-head queries.
    K: (n_heads, seq_k, d) original per-head keys (one MHA checkpoint).
    n_rep: number of original heads collapsed into each new shared GQA
    key head (n_heads must be divisible by n_rep).

    Uptrain the MHA checkpoint into GQA and return the mean squared error
    between the reconstructed (GQA) attention logits and the original
    (MHA) attention logits.
    """
    raise NotImplementedError('your code here')
