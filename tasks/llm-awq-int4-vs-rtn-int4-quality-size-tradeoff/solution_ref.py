import numpy as np

def awq_vs_rtn_quality(w: np.ndarray) -> tuple[float, float]:
    """Correct implementation of AWQ vs RTN quality comparison."""
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
