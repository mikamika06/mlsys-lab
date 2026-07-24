import numpy as np

def scatter_add(dst, idx, src, out):
    """Accumulate src into out at indices idx, handling duplicates."""
    np.add.at(out, idx, src)
