import numpy as np


def compute_relative_drift(torch_out, coreml_out):
    torch_arr = np.array(torch_out, dtype=np.float64)
    coreml_arr = np.array(coreml_out, dtype=np.float64)
    diff_norm = np.linalg.norm(torch_arr - coreml_arr)
    ref_norm = np.linalg.norm(torch_arr)
    if ref_norm == 0.0:
        return 0.0
    return float(diff_norm / ref_norm)
