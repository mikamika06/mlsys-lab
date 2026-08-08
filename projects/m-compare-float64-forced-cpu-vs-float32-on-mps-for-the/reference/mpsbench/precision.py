import numpy as np


def compute_relative_error(a, b):
    diff_norm = float(np.linalg.norm(a - b))
    ref_norm = float(np.linalg.norm(a))
    if ref_norm == 0.0:
        return 0.0
    return diff_norm / ref_norm


def simulate_execution(matrix_a, matrix_b):
    cpu_a = matrix_a.astype(np.float64)
    cpu_b = matrix_b.astype(np.float64)
    y_cpu = cpu_a @ cpu_b

    mps_a = matrix_a.astype(np.float32)
    mps_b = matrix_b.astype(np.float32)
    y_mps = (mps_a @ mps_b).astype(np.float64)

    rel_err = compute_relative_error(y_cpu, y_mps)
    return {"cpu_result": y_cpu, "mps_result": y_mps, "rel_err": rel_err}
