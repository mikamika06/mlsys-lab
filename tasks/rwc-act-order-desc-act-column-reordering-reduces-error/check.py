import numpy as np


def _quantize_column(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x)) / 7.0
    if scale == 0:
        return np.zeros_like(x)
    return np.clip(np.round(x / scale), -8, 7) * scale


def _oracle(W, H, desc_act=True):
    n = W.shape[1]
    if desc_act:
        perm = np.argsort(-np.diag(H), kind="stable")
    else:
        perm = np.arange(n)

    inv_h = np.linalg.inv(H)
    work = W[:, perm].copy()
    out = np.zeros_like(work)

    for j in range(n):
        q = _quantize_column(work[:, j])
        err = work[:, j] - q
        out[:, j] = q
        if j + 1 < n:
            for k in range(j + 1, n):
                work[:, k] -= err * (inv_h[perm[j], perm[k]] / inv_h[perm[j], perm[j]])

    restored = np.zeros_like(out)
    restored[:, perm] = out
    return perm, restored


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.2, 1.1, -0.4, 0.7],
                      [0.7, 0.3, 0.9, -0.8],
                      [-0.2, 0.4, 0.5, 1.2]], dtype=np.float64),
            np.array([[5.0, 0.1, 0.0, 0.2],
                      [0.1, 2.0, 0.2, 0.0],
                      [0.0, 0.2, 1.0, 0.1],
                      [0.2, 0.0, 0.1, 3.0]], dtype=np.float64),
        ),
        (
            np.array([[1.4, -0.6, 0.8],
                      [0.1, 0.9, -1.2]], dtype=np.float64),
            np.array([[1.0, 0.05, 0.0],
                      [0.05, 4.0, 0.1],
                      [0.0, 0.1, 2.0]], dtype=np.float64),
        ),
    ]

    ok_perm = 1.0
    max_err = 0.0

    for W, H in cases:
        ref_perm, ref_hat = _oracle(W, H, True)
        try:
            got_perm, got_hat = sol.gptq_act_order(W.copy(), H.copy())
        except Exception:
            return {"exact_perm": 0.0, "rel_err": 1.0}

        if not np.array_equal(np.asarray(got_perm), ref_perm):
            ok_perm = 0.0

        got_hat = np.asarray(got_hat, dtype=np.float64)
        err = np.linalg.norm(got_hat - ref_hat) / (np.linalg.norm(ref_hat) + 1e-12)
        max_err = max(max_err, float(err))

    return {"exact_perm": ok_perm, "rel_err": max_err}
