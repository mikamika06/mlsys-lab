import math

def mha_gqa_mqa_reconstruct(Q: list[list[list[list[float]]]], K: list[list[list[list[float]]]], V: list[list[list[list[float]]]], group_sizes):
    """Reconstruct attention output under several KV-grouping arities.

    Q: (batch, seq_q, n_heads, head_dim)
    K, V: (batch, seq_k, n_heads, head_dim) -- one KV per head, as in MHA.
    group_sizes: iterable of ints, each dividing n_heads. For each g:
      - mean-pool K and V within every group of g adjacent heads,
      - broadcast the pooled K/V back out to n_heads heads,
      - run standard scaled dot-product attention with the original Q.
    g == 1 reproduces exact MHA; g == n_heads is MQA; values in between are
    GQA(g).

    Returns a list, one entry per group_size, of
    (output: ndarray shape (batch, seq_q, n_heads, head_dim), size_ratio: float)
    where size_ratio is the grouped KV cache size relative to full per-head
    MHA KV cache, i.e. n_kv_heads / n_heads.
    """
    raise NotImplementedError('your code here')
