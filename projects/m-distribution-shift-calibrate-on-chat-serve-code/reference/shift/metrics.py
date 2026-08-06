import numpy as np


def relative_error(ref_out, approx_out):
    diff = np.linalg.norm(ref_out - approx_out)
    norm = np.linalg.norm(ref_out) + 1e-8
    return float(diff / norm)
