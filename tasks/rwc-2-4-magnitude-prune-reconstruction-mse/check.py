import numpy as np
from mlsys import scorers

def _oracle_prune(W):
    W = np.asarray(W, dtype=np.float64)
    n_rows, n_cols = W.shape
    pruned = np.zeros_like(W)
    for i in range(n_rows):
        row = W[i]
        for j in range(0, n_cols, 4):
            group = row[j:j+4]
            if group.size == 0:
                continue
            idx = np.argpartition(np.abs(group), -2)[-2:]
            mask = np.zeros_like(group, dtype=bool)
            mask[idx] = True
            pruned[i,j:j+4][mask] = group[mask]
    return pruned

def grade(sol, fx) -> dict:
    cases = [
        (np.random.randn(5, 12), "random"),
        (np.array([[1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12]]), "pattern"),
        (np.zeros((3, 8)), "zeros"),
        (np.random.randn(7, 15), "non-multiple-of-4 columns"),
    ]
    max_rel_err = 0.0
    for W, name in cases:
        try:
            ref_pruned = _oracle_prune(W)
            ref_mse = np.mean((W - ref_pruned) ** 2)
            student_val = sol.magnitude_prune_mse(W)
            rel_err = abs(student_val - ref_mse) / (ref_mse + 1e-12)
        except Exception:
            return {"mse": 1.0}
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"mse": max_rel_err}
