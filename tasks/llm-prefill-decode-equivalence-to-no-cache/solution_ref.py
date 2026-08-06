import math
import random

random.seed(0)
d = 16
W_ih = [[random.gauss(0, 1) for _ in range(d)] for _ in range(d)]
W_hh = [[random.gauss(0, 1) for _ in range(d)] for _ in range(d)]
b = [random.gauss(0, 1) for _ in range(d)]

def prefill_decode_equiv(inputs: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Compute hidden states for a toy RNN using two strategies:
      * no_cache  – recompute the whole prefix from scratch at each step.
      * cache     – maintain and update the previous hidden state.

    Parameters
    ----------
    inputs : list of list of float of shape (seq_len, d)
        Sequence of token embeddings.

    Returns
    -------
    tuple
        (no_cache, cache) where each is a list of shape (seq_len, d).
    """
    seq_len = len(inputs)

    no_cache_list = []
    for L in range(1, seq_len + 1):
        h_prev = [0.0] * d
        for i in range(L):
            x_i = inputs[i]
            h_next = [0.0] * d
            for r in range(d):
                acc = 0.0
                for c in range(d):
                    acc += W_ih[r][c] * float(x_i[c])
                for c in range(d):
                    acc += W_hh[r][c] * h_prev[c]
                acc += b[r]
                h_next[r] = math.tanh(acc)
            h_prev = h_next
        no_cache_list.append(h_prev)

    cache_list = []
    h_prev = [0.0] * d
    for i in range(seq_len):
        x_i = inputs[i]
        h_next = [0.0] * d
        for r in range(d):
            acc = 0.0
            for c in range(d):
                acc += W_ih[r][c] * float(x_i[c])
            for c in range(d):
                acc += W_hh[r][c] * h_prev[c]
            acc += b[r]
            h_next[r] = math.tanh(acc)
        h_prev = h_next
        cache_list.append(h_prev)

    return no_cache_list, cache_list
