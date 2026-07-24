import numpy as np
from mlsys import scorers


def _quantize_col(col, bits):
    qmax = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(col)) / qmax
    if scale == 0:
        return np.zeros_like(col)
    codes = np.clip(np.round(col / scale), -qmax, qmax)
    return codes * scale


def _oracle(W, H_inv, bits):
    work = np.array(W, dtype=np.float64, copy=True)
    out = np.zeros_like(work)
    n = work.shape[1]
    for j in range(n):
        old = work[:, j].copy()
        q = _quantize_col(old, bits)
        out[:, j] = q
        err = q - old
        denom = H_inv[j, j]
        for k in range(j + 1, n):
            work[:, k] += err * (H_inv[j, k] / denom)
    return out


def _rtn(W, bits):
    out = np.empty_like(W, dtype=np.float64)
    for j in range(W.shape[1]):
        out[:, j] = _quantize_col(W[:, j], bits)
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [
                    [1.1, -0.7, 0.4, -1.5],
                    [0.3, 1.2, -0.8, 0.9],
                    [-0.6, 0.2, 1.4, -0.4],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [1.0, 0.2, 0.1, 0.05],
                    [0.2, 1.0, 0.3, 0.1],
                    [0.1, 0.3, 1.0, 0.25],
                    [0.05, 0.1, 0.25, 1.0],
                ],
                dtype=np.float64,
            ),
            4,
        ),
        (
            np.array(
                [
                    [0.8, -1.1, 0.5],
                    [-0.2, 0.7, -1.3],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [1.0, 0.4, 0.2],
                    [0.4, 1.0, 0.35],
                    [0.2, 0.35, 1.0],
                ],
                dtype=np.float64,
            ),
            3,
        ),
    ]

    ref_errors = []
    rtn_gaps = []

    for W, H_inv, bits in cases:
        try:
            got = np.asarray(sol.gptq_quantize(W.copy(), H_inv.copy(), bits), dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0, "rtn_gap": 0.0}

        ref = _oracle(W, H_inv, bits)
        rtn = _rtn(W, bits)

        ref_errors.append(scorers.rel_err(ref, got))
        rtn_gaps.append(float(np.linalg.norm(got - rtn)))

    return {
        "rel_err": float(max(ref_errors)),
        "rtn_gap": float(min(rtn_gaps)),
    }
