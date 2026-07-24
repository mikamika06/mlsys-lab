import numpy as np

# deterministic weights for reproducibility
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

    # no‑cache strategy
    no_cache = []
    for L in range(1, seq_len + 1):
        h_prev = np.zeros(d, dtype=np.float64)
        for i in range(L):
            h_prev = np.tanh(W_ih @ inputs[i] + W_hh @ h_prev + b)
        no_cache.append(h_prev)
    no_cache = np.stack(no_cache)

    # cache strategy
    cache = []
    h_prev = np.zeros(d, dtype=np.float64)
    for i in range(seq_len):
        h_prev = np.tanh(W_ih @ inputs[i] + W_hh @ h_prev + b)
        cache.append(h_prev)
    cache = np.stack(cache)

    return no_cache, cache
