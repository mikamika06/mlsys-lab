import numpy as np

def compute_relative_error(output_ref, output_merged):
    diff = np.linalg.norm(output_merged - output_ref)
    denom = np.linalg.norm(output_ref)
    if denom == 0.0:
        return float(diff)
    return float(diff / denom)
