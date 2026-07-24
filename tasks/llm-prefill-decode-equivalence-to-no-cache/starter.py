import numpy as np

# deterministic weights for reproducibility
np.random.seed(0)
d = 16
W_ih = np.random.randn(d, d).astype(np.float64)
W_hh = np.random.randn(d, d).astype(np.float64)
b = np.random.randn(d).astype(np.float64)

def prefill_decode_equiv(inputs: np.ndarray):
    """
    Broken implementation – returns incorrect no_cache.
    """
    seq_len = inputs.shape[0]

    # cache strategy (correct)
    h_prev = np.zeros(d, dtype=np.float64)
    cache = []
    for i in range(seq_len):
        h_prev = np.tanh(W_ih @ inputs[i] + W_hh @ h_prev + b)
        cache.append(h_prev)
    cache = np.stack(cache)

    # no_cache incorrectly: all zeros
    no_cache = np.zeros_like(cache)  # wrong

    return no_cache, cache
