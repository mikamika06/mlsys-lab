import numpy as np

def mha_single_kv_head(Q, K, V):
    """Compute scaled dot-product attention with a single shared KV head.

    Q : (B, H, S, D)  – queries, H heads
    K : (B, 1, S, D)  – keys, one head
    V : (B, 1, S, D)  – values, one head

    Returns (B, H, S, D).
    """
    raise NotImplementedError("your code here")
