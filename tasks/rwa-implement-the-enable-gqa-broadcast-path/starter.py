import numpy as np


def enable_gqa_broadcast_attention(Q, K, V):
    """
    Q: (batch, n_q, seq_q, d) queries, PyTorch SDPA axis order.
    K, V: (batch, n_kv, seq_k, d) with n_q a multiple of n_kv.

    Broadcast K/V from n_kv to n_q heads using blocked grouping (query
    head h reads KV head h // r, where r = n_q // n_kv -- NOT a cyclic
    h % n_kv assignment), then run scaled dot-product attention per query
    head with scale 1/sqrt(d).

    Returns a (batch, n_q, seq_q, d_v) float64 array.
    """
    raise NotImplementedError('your code here')
