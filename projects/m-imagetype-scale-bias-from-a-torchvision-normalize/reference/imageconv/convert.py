import numpy as np

def verify_drift(ref_out, got_out, threshold):
    num = np.linalg.norm(ref_out - got_out)
    den = np.linalg.norm(ref_out) + 1e-12
    rel = float(num / den)
    return rel <= threshold
