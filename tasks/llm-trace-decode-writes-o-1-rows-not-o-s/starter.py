import numpy as np


def decode_step(k_cache, v_cache, q, k_new, v_new):
    """One autoregressive decode step.

    k_cache, v_cache : float arrays of shape (S, d) -- the keys/values already
                       cached for the S previous tokens (S may be 0).
    q                : float array (d,)  -- query for the new token.
    k_new, v_new     : float arrays (d,) -- key/value for the new token.

    Return (out, k_cache2, v_cache2):
      out       -- (d,) attention output softmax(q @ K.T / sqrt(d)) @ V over all
                   S+1 cached tokens (the new one included).
      k_cache2  -- (S+1, d) cache with k_new appended as its last row.
      v_cache2  -- (S+1, d) cache with v_new appended as its last row.

    Append exactly one new K row and one new V row -- O(1) Python-level work per
    step, independent of S. Do NOT rebuild the cache with a Python loop over the
    S existing rows.
    """
    raise NotImplementedError("your code here")
