import numpy as np


def chunked_causal_prefill(q, k, v, chunk_sizes):
    """Causal self-attention over the whole prompt, computed by processing
    `chunk_sizes` in sequential order and carrying a running online-softmax
    state across the KV chunks that precede or equal each query chunk.

    q, k, v: (n, d) float64 arrays for the whole prompt.
    chunk_sizes: list of positive ints summing to n; the prefill schedule.

    Returns the (n, d) causal attention output. Must be identical to
    single-shot dense causal attention regardless of the chunk schedule
    used.
    """
    raise NotImplementedError('your code here')
