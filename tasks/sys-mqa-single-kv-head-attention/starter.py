import math

def mha_single_kv_head(Q: list[list[list[list[float]]]], K: list[list[list[list[float]]]], V: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
    """Compute scaled dot-product attention with a single shared KV head.

    Q : (B, H, S, D)  – queries, H heads
    K : (B, 1, S, D)  – keys, one head
    V : (B, 1, S, D)  – values, one head

    Returns (B, H, S, D).
    """
    raise NotImplementedError('your code here')
