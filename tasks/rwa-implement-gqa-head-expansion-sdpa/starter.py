import math

def gqa_head_expansion_attention(Q: list[list[list[list[float]]]], K: list[list[list[list[float]]]], V: list[list[list[list[float]]]]) -> tuple[list[list[list[list[float]]]], float]:
    """
    Q: (batch, seq_q, n_q, d) queries.
    K, V: (batch, seq_k, n_kv, d) with n_q a multiple of n_kv.

    Expand K/V from n_kv to n_q heads via repeat_interleave (each KV head
    repeated n_rep = n_q // n_kv times consecutively along the head axis),
    then run standard scaled dot-product attention.

    Returns (output, memory_ratio):
      output: (batch, seq_q, n_q, d) float64 array.
      memory_ratio: float, n_kv / n_q.
    """
    raise NotImplementedError('your code here')
