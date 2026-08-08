import numpy as np


def verify_drift(reference_output, converted_output):
    diff = np.abs(reference_output - converted_output)
    rel_err = np.max(diff / (np.abs(reference_output) + 1e-7))
    return float(rel_err)
