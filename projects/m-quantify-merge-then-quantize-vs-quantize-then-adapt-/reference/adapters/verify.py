import numpy as np


def verify_output_shift(base_out, adapted_out, expected_shift, rel_err):
    actual_shift = adapted_out - base_out
    diff = np.abs(actual_shift - expected_shift)
    error = np.mean(diff / (np.abs(expected_shift) + 1e-5))
    return bool(error <= rel_err)
