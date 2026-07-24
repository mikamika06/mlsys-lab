import numpy as np

def _compute_reference_errors(w):
    """Return mean relative errors for AWQ and RTN on weight matrix w."""
    C = w.shape[0]
    rel_awq_list = []
    rel_rtn_list = []
    for i in range(C):
        row = w[i]
        s0 = np.max(np.abs(row)) / 7.0
        q = np.clip(np.round(row / s0), -8, 7)
        deq_rtn = q * s0
        rel_err_rtn = np.linalg.norm(deq_rtn - row) / (np.linalg.norm(row) + 1e-12)

        denom = np.sum(q ** 2) + 1e-12
        s_opt = np.sum(row * q) / denom
        deq_awq = q * s_opt
        rel_err_awq = np.linalg.norm(deq_awq - row) / (np.linalg.norm(row) + 1e-12)

        rel_rtn_list.append(rel_err_rtn)
        rel_awq_list.append(rel_err_awq)
    return np.mean(rel_awq_list), np.mean(rel_rtn_list)

def grade(sol, fx) -> dict:
    """Grade the student's implementation."""
    np.random.seed(0)
    max_diff_awq = 0.0
    max_diff_rtn = 0.0
    for _ in range(5):
        w = np.random.randn(8, 128).astype(np.float64)
        ref_awq, ref_rtn = _compute_reference_errors(w)

        try:
            got = sol.awq_vs_rtn_quality(w)
            if not (isinstance(got, tuple) and len(got) == 2):
                max_diff_awq = float("inf")
                max_diff_rtn = float("inf")
                break
            diff_awq = abs(got[0] - ref_awq)
            diff_rtn = abs(got[1] - ref_rtn)
            if diff_awq > max_diff_awq:
                max_diff_awq = diff_awq
            if diff_rtn > max_diff_rtn:
                max_diff_rtn = diff_rtn
        except Exception:
            max_diff_awq = float("inf")
            max_diff_rtn = float("inf")
    return {"rel_err_awq": max_diff_awq, "rel_err_rtn": max_diff_rtn}
