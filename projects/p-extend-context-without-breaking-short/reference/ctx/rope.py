import numpy as np

def apply_rope_scaling(q, k, scale_type, factor):
    if scale_type == "linear":
        return q / factor, k / factor
    elif scale_type == "ntk":
        return q * np.log(factor), k * np.log(factor)
    return q, k
