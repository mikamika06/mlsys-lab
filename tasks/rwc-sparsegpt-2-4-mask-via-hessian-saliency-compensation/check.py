import numpy as np


def _oracle(W, X):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    s = X.shape[0]
    H = (X.T @ X) / s + 1e-4 * np.eye(X.shape[1], dtype=np.float64)
    Hinv = np.linalg.inv(H)

    mask = np.zeros_like(W, dtype=np.int64)
    out = W.copy()

    for r in range(W.shape[0]):
        for start in range(0, W.shape[1], 4):
            cols = list(range(start, start + 4))
            scores = []
            for c in cols:
                scores.append((W[r, c] ** 2) / Hinv[c, c])
            keep = set(cols)
            for c, _ in sorted(zip(cols, scores), key=lambda x: x[1])[:2]:
                keep.remove(c)
            for c in keep:
                mask[r, c] = 1

            pruned = [c for c in cols if mask[r, c] == 0]
            for c in pruned:
                old = out[r, c]
                for k in cols:
                    if mask[r, k] == 1:
                        out[r, k] -= old * Hinv[k, c] / Hinv[c, c]
                out[r, c] = 0.0

    return mask, out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.8, -1.2, 0.5, 2.0, -0.7, 0.4, 1.5, -0.3],
                      [1.1, 0.2, -0.9, 0.6, 1.7, -1.4, 0.3, 0.9]]),
            np.array([[1.0, 0.2, -0.5, 1.3, 0.7, -1.1, 0.4, 0.8],
                      [-0.3, 1.5, 0.6, -0.7, 1.2, 0.1, -0.9, 0.5],
                      [0.4, -0.8, 1.1, 0.3, -1.2, 0.9, 0.6, -0.4]])
        ),
        (
            np.array([[2.0, 0.1, 0.2, -1.0]]),
            np.array([[1.0, 2.0, 3.0, 4.0],
                      [0.5, -1.0, 2.0, 1.5]])
        )
    ]

    mask_ok = 1.0
    err = 0.0

    for W, X in cases:
        ref_mask, ref_W = _oracle(W, X)
        try:
            got_mask, got_W = sol.sparsegpt_2_4(W.copy(), X.copy())
            got_mask = np.asarray(got_mask)
            got_W = np.asarray(got_W, dtype=np.float64)
        except Exception:
            return {"mask_exact": 0.0, "rel_err": 1e9}

        if not np.array_equal(got_mask, ref_mask):
            mask_ok = 0.0
        denom = np.linalg.norm(ref_W) + 1e-12
        err = max(err, float(np.linalg.norm(got_W - ref_W) / denom))

    return {"mask_exact": mask_ok, "rel_err": err}
