import numpy as np

def scatter_add(dst, idx, src, out):
    """Accumulate src into out at indices idx, handling duplicates.
    Currently overwrites — fix it to accumulate byte-exact."""
    out[:] = dst
    for i in range(len(idx)):
        out[idx[i]] = src[i]  # BUG: overwrites on duplicate indices
