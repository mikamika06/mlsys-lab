import numpy as np


def _quantize_column(x, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(x)) / qmax
    if scale == 0:
        return np.zeros_like(x)
    codes = np.clip(np.round(x / scale), -qmax, qmax)
    return codes * scale


def _oracle_gptq(W, H, bits):
    work = np.asarray(W, dtype=np.float64).copy()
    out = np.zeros_like(work)
    n = work.shape[1]
    for j in range(n):
        original = work[:, j].copy()
        quantized = _quantize_column(original, bits)
        out[:, j] = quantized
        error = original - quantized
        for k in range(j + 1, n):
            work[:, k] -= error * (H[j, k] / H[j, j])
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 0.7, -1.2], [-0.5, 1.4, 0.3]], dtype=np.float64),
            np.array([[2.0, 0.5, -0.2], [0.5, 3.0, 0.4], [-0.2, 0.4, 2.5]], dtype=np.float64),
            4,
        ),
        (
            np.array([[2.3, -1.1], [0.4, 0.9], [-1.7, 2.2]], dtype=np.float64),
            np.array([[1.5, 0.3], [0.3, 2.0]], dtype=np.float64),
            3,
        ),
        (
            np.arange(20, dtype=np.float64).reshape(4, 5) / 3.0 - 2.0,
            np.array(
                [
                    [3.0, 0.2, 0.1, -0.1, 0.0],
                    [0.2, 2.5, 0.4, 0.0, 0.1],
                    [0.1, 0.4, 2.2, 0.3, -0.2],
                    [-0.1, 0.0, 0.3, 1.8, 0.5],
                    [0.0, 0.1, -0.2, 0.5, 2.7],
                ],
                dtype=np.float64,
            ),
            4,
        ),
    ]

    worst = 0.0
    for W, H, bits in cases:
        ref = _oracle_gptq(W, H, bits)
        try:
            got = sol.gptq_quantize(W, H, bits)
        except Exception:
            return {"rel_err": float("inf")}
        got = np.asarray(got, dtype=np.float64)
        err = np.linalg.norm(got.ravel() - ref.ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12)
        worst = max(worst, float(err))
    return {"rel_err": worst}
