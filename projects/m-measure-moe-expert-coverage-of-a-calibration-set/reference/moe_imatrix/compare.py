import numpy as np


def compare_imatrices(matrix_a, matrix_b):
    common_keys = set(matrix_a.keys()).intersection(set(matrix_b.keys()))
    diffs = {}
    for k in sorted(common_keys):
        a = np.array(matrix_a[k], dtype=np.float64)
        b = np.array(matrix_b[k], dtype=np.float64)
        min_len = min(a.size, b.size)
        if min_len == 0:
            diffs[k] = 0.0
            continue
        a = a[:min_len]
        b = b[:min_len]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0:
            a = a / norm_a
        if norm_b > 0:
            b = b / norm_b
        diff = float(np.linalg.norm(a - b))
        diffs[k] = diff
    overall = float(np.mean(list(diffs.values()))) if diffs else 0.0
    return {"tensor_diffs": diffs, "mean_diff": overall}
