import numpy as np


def compute_rel_err(quantized, reference):
    num = np.linalg.norm(quantized - reference)
    den = np.linalg.norm(reference) + 1e-8
    return float(num / den)
