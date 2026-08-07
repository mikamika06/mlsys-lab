import numpy as np

def relative_error(ref_out, test_out):
    diff = np.linalg.norm(ref_out - test_out)
    norm = np.linalg.norm(ref_out)
    if norm == 0:
        return 0.0
    return float(diff / norm)
