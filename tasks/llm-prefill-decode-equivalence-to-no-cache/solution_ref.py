import math
import numpy as np

np.random.seed(0)
d = 16
W_ih = np.random.randn(d, d).astype(np.float64)
W_hh = np.random.randn(d, d).astype(np.float64)
b = np.random.randn(d).astype(np.float64)

def prefill_decode_equiv(inputs: np.ndarray):
    """
    Compute hidden states for a toy RNN using two strategies:
      * no_cache  – recompute the whole prefix from scratch at each step.
      * cache     – maintain and update the previous hidden state.

    Parameters
    ----------
    inputs : np.ndarray of shape (seq_len, d)
        Sequence of token embeddings.

    Returns
    -------
    tuple
        (no_cache, cache) where each is an array of shape (seq_len, d).
    """
    seq_len = inputs.shape[0]

    no_cache_list = []
    for L in range(1, seq_len + 1):
        h_prev = [0.0] * d
        for i in range(L):
            x_i = inputs[i]
            h_next = [0.0] * d
            for r in range(d):
                acc = 0.0
                for c in range(d):
                    acc += W_ih[r, c] * float(x_i[c])
                for c in range(d):
                    acc += W_hh[r, c] * h_prev[c]
                acc += b[r]
                h_next[r] = math.tanh(acc)
            h_prev = h_next
        no_cache_list.append(h_prev)
    no_cache = np.array(no_cache_list, dtype=np.float64)

    cache_list = []
    h_prev = [0.0] * d
    for i in range(seq_len):
        x_i = inputs[i]
        h_next = [0.0] * d
        for r in range(d):
            acc = 0.0
            for c in range(d):
                acc += W_ih[r, c] * float(x_i[c])
            for c in range(d):
                acc += W_hh[r, c] * h_prev[c]
            acc += b[r]
            h_next[r] = math.tanh(acc)
        h_prev = h_next
        cache_list.append(h_prev)
    cache = np.array(cache_list, dtype=np.float64)

    return no_cache, cache
