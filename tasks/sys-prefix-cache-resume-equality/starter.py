import numpy as np


def run_with_cache(Wq, Wk, Wv, X, kv_cache=None):
    """Single-head causal self-attention over new tokens X, using (and
    extending) an optional KV cache built from an already-processed prefix.

    See task.md for the exact caching and masking rules.

    Returns
    -------
    out : (L, d) float64 -- attention output for the L new tokens.
    new_cache : {'K': (P+L, d), 'V': (P+L, d)} -- the extended cache.
    """
    raise NotImplementedError('your code here')
