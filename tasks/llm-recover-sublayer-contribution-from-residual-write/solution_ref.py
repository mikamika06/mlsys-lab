import numpy as np

def recover_sublayer_contribution(in_, out_):
    """Return the sublayer contribution from residual write."""
    in_arr = np.asarray(in_)
    out_arr = np.asarray(out_)
    return (out_arr - in_arr).astype(np.float64)
