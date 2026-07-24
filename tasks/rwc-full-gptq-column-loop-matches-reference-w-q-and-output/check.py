import numpy as np


def _oracle_gptq(W, X, bits=3, group_size=2, damp=0.01):
    W_work = np.asarray(W, dtype=np.float64).copy()
    m, n = W_work.shape
    H = X @ X.T
    H = H + np.eye(n) * damp * np.mean(np.diag(H))
    Hinv = np.linalg.inv(H)

    W_q = np.zeros_like(W_work)
    maxq = (1 << (bits - 1)) - 1

    scales = {}
    for start in range(0, n, group_size):
        end = min(n, start + group_size)
        block = W_work[:, start:end]
        s = np.max(np.abs(block), axis=1) / maxq
        s[s == 0] = 1.0
        scales[start] = s

    for i in range(n):
        start = (i // group_size) * group_size
        scale = scales[start]
        q = np.clip(np.round(W_work[:, i] / scale), -maxq, maxq) * scale
        W_q[:, i] = q
        err = q - W_work[:, i]
        if i + 1 < n:
            coeff = Hinv[i, i + 1:] / Hinv[i, i]
            W_work[:, i + 1:] -= np.outer(err, coeff)

    return W_q, W_q @ X


def grade(sol, fx) -> dict:
    W = np.array(
        [
            [0.71, -0.44, 0.18, 0.93, -0.61, 0.25],
            [-0.32, 0.84, -0.55, 0.12, 0.77, -0.29],
            [0.45, 0.16, 0.62, -0.74, 0.31, 0.58],
        ],
        dtype=np.float64,
    )
    X = np.array(
        [
            [1.0, 0.2, -0.4, 0.7],
            [-0.5, 1.1, 0.3, -0.2],
            [0.8, -0.6, 0.9, 0.1],
            [0.4, 0.5, -0.7, 1.2],
            [-0.3, 0.8, 0.6, -0.9],
            [0.2, -0.1, 0.4, 0.5],
        ],
        dtype=np.float64,
    )

    ref_w, ref_y = _oracle_gptq(W, X)

    try:
        got_w, got_y = sol.gptq_quantize(W, X)
        got_w = np.asarray(got_w, dtype=np.float64)
        got_y = np.asarray(got_y, dtype=np.float64)
    except Exception:
        return {"wq_rel_err": 1.0, "output_rel_err": 1.0}

    w_err = np.linalg.norm(got_w - ref_w) / (np.linalg.norm(ref_w) + 1e-12)
    y_err = np.linalg.norm(got_y - ref_y) / (np.linalg.norm(ref_y) + 1e-12)
    return {
        "wq_rel_err": float(w_err),
        "output_rel_err": float(y_err),
    }
