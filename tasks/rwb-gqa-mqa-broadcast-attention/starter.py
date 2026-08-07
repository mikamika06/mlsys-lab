import math

def gqa_broadcast_attention(q: list[list[list[float]]], k: list[list[list[float]]], v: list[list[list[float]]]) -> list[list[list[float]]]:
    """GQA/MQA attention: broadcast each KV head across its query-head group.

    q: (H_q, n, d) float64 per-head queries.
    k, v: (H_kv, n, d) float64 per-head keys/values, H_q % H_kv == 0.

    n_rep = H_q // H_kv. Query head h reads KV head h // n_rep (repeat
    each KV head n_rep times along the head axis), then runs standard
    (non-causal) scaled dot-product attention per query head.

    Returns the (H_q, n, d) attention output.
    """
    raise NotImplementedError('your code here')
