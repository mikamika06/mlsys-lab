def compare_imatrices(imatrix_a, imatrix_b):
    """Compare two imatrices and return a divergence metric."""
    import numpy as np
    keys = sorted(set(imatrix_a.keys()) & set(imatrix_b.keys()))
    if not keys:
        return 0.0
    diffs = []
    for k in keys:
        a = np.array(imatrix_a[k], dtype=float)
        b = np.array(imatrix_b[k], dtype=float)
        if a.shape != b.shape:
            continue
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            diffs.append(0.0 if np.array_equal(a, b) else 1.0)
        else:
            cos_sim = np.dot(a, b) / (norm_a * norm_b)
            diffs.append(float(cos_sim))
    return float(np.mean(diffs)) if diffs else 0.0
